import asyncio
import json
import logging
import math
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

import pimm
from positronic import geom, utils
from positronic.drivers.webrtc import VideoPeer

logger = logging.getLogger(__name__)

# The names of the controllers in the positions and the buttons that ``WebXR`` emits.
LEFT, RIGHT = 'left', 'right'

# The routes of the page state and of the controller stream.
STATE_PATH, CONTROLLER_PATH, INPUT_PATH = '/state', '/ws', '/input'

# The fields of a message of the page on ``/ws``. The orientation is a wxyz quaternion. The timestamp is the time in
# milliseconds on the monotonic clock of the page at which the page sent the message.
_CONTROLLERS, _POSITION, _ORIENTATION, _BUTTONS = 'controllers', 'position', 'orientation', 'buttons'
_PRESSES, _TIMESTAMP = 'presses', 'timestamp'
# The field of a message on ``/ws`` that carries a WebRTC setup message of the video, from the page or to it.
_SIGNAL = 'webrtc'
# The button values of a controller in the xr-standard gamepad layout, up to B or Y at index 5
_CONTROLLER_BUTTONS = 6


@dataclass(frozen=True)
class ControllerButtons:
    """The value of each button of a controller, and the number of presses of each button that the page counted since
    it loaded."""

    values: tuple[float, ...]
    presses: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.presses) != len(self.values):
            raise ValueError(
                f'a controller has a press count for each button value, got {len(self.presses)} counts and '
                f'{len(self.values)} values'
            )
        if not all(isinstance(count, int) and count >= 0 for count in self.presses):
            raise ValueError(f'a press count is an integer of at least 0, got {list(self.presses)}')


def parse_controller_message(
    data: Mapping[str, Any],
) -> tuple[dict[str, geom.Transform3D | None], dict[str, ControllerButtons | None]]:
    """The poses and the buttons of the controllers in a message of the page on ``/ws``, the inverse of
    ``controller_message``. A controller that the page does not track has ``None`` for both, and a tracked
    controller has at least ``_CONTROLLER_BUTTONS`` button values."""
    controller_positions: dict[str, geom.Transform3D | None] = {LEFT: None, RIGHT: None}
    buttons_dict: dict[str, ControllerButtons | None] = {LEFT: None, RIGHT: None}
    for side in [RIGHT, LEFT]:
        controller = data[_CONTROLLERS][side]
        # A hand that the headset tracks with no controller in it has fewer values than a controller, or none. The pose
        # and the buttons go out on two signals, so only here is the pose known to be of the hand.
        if controller is not None and len(controller[_BUTTONS]) >= _CONTROLLER_BUTTONS:
            translation = np.array(controller[_POSITION], dtype=np.float64)
            rotation = np.array(controller[_ORIENTATION], dtype=np.float64)
            controller_positions[side] = geom.Transform3D(translation, geom.Rotation.from_quat(rotation))
            values = tuple(float(value) for value in controller[_BUTTONS])
            buttons_dict[side] = ControllerButtons(values, tuple(controller[_PRESSES]))

    return controller_positions, buttons_dict


def controller_message(
    positions: Mapping[str, geom.Transform3D], buttons: Mapping[str, ControllerButtons], sent_ms: float | None = None
) -> dict[str, Any]:
    """The message of the page on ``/ws`` for the controllers that ``positions`` and ``buttons`` name, sent at
    ``sent_ms`` on a monotonic clock in milliseconds, or now. The page does not track the others."""
    if positions.keys() != buttons.keys():
        raise ValueError(
            f'positions and buttons must name the same controllers, got {list(positions)}, {list(buttons)}'
        )
    if short := {side: len(state.values) for side, state in buttons.items() if len(state.values) < _CONTROLLER_BUTTONS}:
        raise ValueError(f'a controller has at least {_CONTROLLER_BUTTONS} button values, got {short}')
    tracked = {
        side: {
            _POSITION: pose.translation.tolist(),
            _ORIENTATION: pose.rotation.as_quat.tolist(),
            _BUTTONS: list(buttons[side].values),
            _PRESSES: list(buttons[side].presses),
        }
        for side, pose in positions.items()
    }
    sent = time.monotonic() * 1000 if sent_ms is None else sent_ms
    return {_CONTROLLERS: {side: tracked.get(side) for side in (LEFT, RIGHT)}, _TIMESTAMP: sent}


def _get_or_create_ssl_files(port: int, keyfile: str, certfile: str) -> tuple[str, str]:
    """Return paths to SSL key/cert, creating basic self-signed ones if missing.

    Note: For iPhone/XR Browser development, prefer HTTP (use_https=False).
    """
    if os.path.exists(keyfile) and os.path.exists(certfile):
        return keyfile, certfile

    tmp_dir = tempfile.gettempdir()
    tmp_key = os.path.join(tmp_dir, f'webxr_key_{port}.pem')
    tmp_cert = os.path.join(tmp_dir, f'webxr_cert_{port}.pem')

    if os.path.exists(tmp_key) and os.path.exists(tmp_cert):
        return tmp_key, tmp_cert

    try:
        cl = (
            [
                'openssl',
                'req',
                '-x509',
                '-nodes',
                '-days',
                '365',
                '-newkey',
                'rsa:2048',
                '-keyout',
                tmp_key,
                '-out',
                tmp_cert,
                '-subj',
                '/CN=localhost',
            ],
        )
        subprocess.run(*cl, check=True, capture_output=True)
        logger.info(f'Generated self-signed SSL certs in {tmp_dir}')
    except Exception as e:
        logger.warning('Failed to generate SSL certificates via openssl. Provide cert/key files or use HTTP.')
        raise e

    return tmp_key, tmp_cert


class Frontend(StrEnum):
    """The page that ``WebXR`` serves."""

    HEADSET = 'headset'
    PHONE = 'phone'


class SessionMode(StrEnum):
    """The WebXR session that the page requests."""

    AR = 'immersive-ar'
    VR = 'immersive-vr'


def _build_access_url(host: str, port: int, use_https: bool) -> str:
    scheme = 'https' if use_https else 'http'
    return f'{scheme}://{host}:{port}/'


# How fast the delay base of a connection on ``/ws`` may rise, in ms per ms of arrival time. It is far above the drift
# of a clock, so the base follows a page clock that runs slow.
_DELAY_BASE_RISE = 1e-3


class _ConnectionPresses:
    """The press counts of one connection on ``/ws``, from a base for each controller.

    The first message of a controller sets its base, so a press that the page counted before it connected does
    nothing. The delay of a message is its arrival time less its timestamp, so it holds the offset of the two clocks
    too. The delay base is the least delay of the connection. It rises at ``_DELAY_BASE_RISE`` with the arrival time,
    and never above the delay of the newest fresh message. A message whose delay is more than ``stale_limit`` seconds
    above the delay base is stale, as in a backlog after a stall of the network: its counts become the base, so the
    presses of the stall do nothing. Only differences of delays count, so the clocks need not agree."""

    def __init__(self, stale_limit: float) -> None:
        self._stale_ms = stale_limit * 1000
        # Infinite until the first message, whose delay then becomes the delay base
        self._delay_base_ms = math.inf
        self._fresh_delay_ms = math.inf
        self._last_received_ms = 0.0
        self._base: dict[str, tuple[int, ...]] = {}

    def counted(
        self, buttons: Mapping[str, ControllerButtons | None], sent_ms: float, received_ms: float
    ) -> dict[str, ControllerButtons | None]:
        """``buttons`` of a message sent at ``sent_ms`` by the page clock and received at ``received_ms``, with the
        presses since the base of each controller."""
        delay_ms = received_ms - sent_ms
        risen_ms = self._delay_base_ms + _DELAY_BASE_RISE * (received_ms - self._last_received_ms)
        self._delay_base_ms = min(risen_ms, self._fresh_delay_ms, delay_ms)
        self._last_received_ms = received_ms
        stale = delay_ms - self._delay_base_ms > self._stale_ms
        if not stale:
            self._fresh_delay_ms = delay_ms
        return {
            side: None if state is None else self._since_base(side, state, stale) for side, state in buttons.items()
        }

    def _since_base(self, side: str, buttons: ControllerButtons, stale: bool) -> ControllerButtons:
        """A stale message, counts that go down, or a new number of counts set the base again."""
        base = self._base.setdefault(side, buttons.presses)
        if (
            stale
            or len(base) != len(buttons.presses)
            or any(now < then for now, then in zip(buttons.presses, base, strict=True))
        ):
            base = self._base[side] = buttons.presses
        return replace(buttons, presses=tuple(now - then for now, then in zip(buttons.presses, base, strict=True)))


class WebXR(pimm.ControlSystem):
    """WebXR server for a headset or a phone AR controller.

    Serves a single frontend at a time and streams controller/pose data back
    to the application over WebSocket. The headset page gets the frames of ``frame`` as a WebRTC video track.

    Important for iPhone
    - Use a WebXR-capable browser (XR Browser recommended).
    - For development, use HTTP (`use_https=False`). XR Browser on iOS blocks self-signed
      HTTPS and may not allow proceeding to an untrusted site, which also
      blocks the `wss://` WebSocket.
    - If HTTPS is required, provide a cert trusted by the device.

    Endpoints
    - `/`                -> selected frontend; it redirects a URL with a query to `/`
    - `/state`           -> the last `page_state` value as JSON
    - `/input`           -> POST a JSON object from the page, such as a choice of the teleoperator; ``page_input``
                            emits it
    - `/ws`              -> JSON controller stream from client to server. A message with a ``webrtc`` field carries a
                            WebRTC setup message of the video (``positronic.drivers.webrtc``), in both directions
    - Shared assets: `/three.min.js`, `/webxr-button.js`, `/core.js`, `/session-mode.js`
    - Headset-only: `/video-player.js`, `/webrtc-video.js`, `/quest-trigger.js`, `/press-count.js`

    Parameters
    - port: TCP port to bind.
    - ssl_keyfile, ssl_certfile: TLS files if `use_https=True`.
    - frontend: the page to serve, a ``Frontend``.
    - session_mode: the WebXR session that the page requests, a ``SessionMode``.
    - use_https: enable HTTPS; set False for iPhone dev.
    - stale_limit: how many seconds later than the delay base of its connection a message on ``/ws`` may come and
      still count its presses. A later message, as in a backlog after a stall of the network, presses nothing.
    - video_fps: the frame rate of the frames that ``frame`` receives.
    """

    def __init__(
        self,
        port: int,
        ssl_keyfile: str = 'key.pem',
        ssl_certfile: str = 'cert.pem',
        frontend: Frontend = Frontend.HEADSET,
        session_mode: SessionMode = SessionMode.AR,
        use_https: bool = True,
        sensitivity: float = 1.0,
        stale_limit: float = 0.5,
        video_fps: int = 30,
    ):
        utils.check_positive_int(video_fps, 'video_fps')
        utils.check_finite_positive(stale_limit, 'stale_limit')
        if session_mode is SessionMode.VR and frontend is not Frontend.HEADSET:
            raise ValueError(f'session_mode VR needs the headset frontend, got {frontend!r}')
        self.port = port
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        self.frontend = frontend
        self.session_mode = session_mode
        self.use_https = use_https
        self.server_thread = None
        self.frame = pimm.ControlSystemReceiver(self)
        # A JSON object that the page polls at ``/state``, such as the state of a trial. It is empty while nothing
        # sends one.
        self.page_state = pimm.ControlSystemReceiver(self)
        # Each JSON object that the page posts to ``/input``.
        self.page_input = pimm.ControlSystemEmitter(self)
        self.controller_positions = pimm.ControlSystemEmitter(self)
        self.buttons = pimm.ControlSystemEmitter(self)
        self.sensitivity = sensitivity
        self.stale_limit = stale_limit
        self.video_fps = video_fps

    def _next_frame(self, last_sent_time: pimm.Time | None) -> pimm.Message | None:
        """The frame the video track sends next. ``None`` while no new frame arrived or the camera is lost."""
        msg = self.frame.read()
        if msg is None or isinstance(msg.data, pimm.SignalError) or msg.time == last_sent_time:
            return None
        return msg

    @staticmethod
    def _file_route(path):
        """A route handler that serves the file at ``path``."""

        async def serve():
            return FileResponse(path)

        return serve

    def _add_page_routes(self, app: FastAPI) -> None:
        """The page and its scripts, from the package assets."""

        @app.get('/')
        async def root(request: Request):
            if request.url.query:
                return RedirectResponse('/')
            if self.frontend is Frontend.HEADSET:
                return FileResponse(utils.package_assets_path('assets/webxr/index.html'))
            return FileResponse(utils.package_assets_path('assets/webxr_phone/index.html'))

        @app.get(STATE_PATH)
        async def page_state():
            message = self.page_state.read()
            return JSONResponse({} if message is None else message.data)

        @app.post(INPUT_PATH)
        async def page_input(request: Request):
            # A browser sends `Origin` with each cross-site POST: a page on another site cannot choose for the
            # teleoperator. A client that sends none passes, as the controller stream does.
            origin = request.headers.get('origin')
            parsed = None if origin is None else urlparse(origin)
            if parsed is not None and (parsed.scheme, parsed.netloc) != (request.url.scheme, request.url.netloc):
                raise HTTPException(403, 'cross-origin request refused')
            try:
                body = await request.json()
            except ValueError:
                raise HTTPException(422, 'the input must be JSON') from None
            if not isinstance(body, dict):
                raise HTTPException(422, 'the input must be a JSON object')
            self.page_input.emit(body)
            return JSONResponse({})

        scripts = ['three.min.js', 'webxr-button.js', 'core.js']
        if self.frontend is Frontend.HEADSET:
            scripts += ['video-player.js', 'webrtc-video.js', 'quest-trigger.js', 'press-count.js']
        for script in scripts:
            app.get(f'/{script}')(self._file_route(utils.package_assets_path(f'assets/webxr/{script}')))

        @app.get('/session-mode.js')
        async def session_mode():
            return Response(
                f'export const SESSION_MODE = {json.dumps(self.session_mode.value)};\n', media_type='text/javascript'
            )

    def _add_controller_route(self, app: FastAPI, should_stop: pimm.SignalReceiver) -> None:
        """``/ws``: the controller stream from the page, and the setup messages of its video. The video peer of a socket
        closes with the socket."""

        # TODO: Require a pairing key before `/ws` accepts a client. Any client that can reach the port sends input.
        @app.websocket(CONTROLLER_PATH)
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            logger.info('WebSocket connection accepted')
            presses = _ConnectionPresses(self.stale_limit)
            peer: VideoPeer | None = None

            async def send_signal(message: dict[str, Any]) -> None:
                await websocket.send_json({_SIGNAL: message})

            try:
                fps = pimm.utils.RateCounter('Websocket')
                while not should_stop.value:
                    try:
                        # Use asyncio.wait_for with timeout to avoid blocking indefinitely
                        data = await asyncio.wait_for(websocket.receive_json(), timeout=1)
                        if _SIGNAL in data:
                            peer = peer or VideoPeer(self._next_frame, send_signal)
                            peer.signal(data[_SIGNAL])
                            continue
                        received_ms = time.monotonic() * 1000
                        controller_positions, buttons = parse_controller_message(data)
                        buttons = presses.counted(buttons, float(data[_TIMESTAMP]), received_ms)

                        # apply scaling
                        for transform in controller_positions.values():
                            if transform is not None:
                                transform.translation *= self.sensitivity

                        # A sample with no controller in it tells a reader that the controllers are not tracked
                        self.controller_positions.emit(controller_positions)
                        self.buttons.emit(buttons)
                        fps.tick()
                    except TimeoutError:
                        # Timeout is normal, just continue to check should_stop
                        continue
                    except Exception as e:
                        logger.error(f'Error processing WebSocket message: {e}')
                        break
            except Exception as e:
                logger.error(f'WebSocket error: {e}')
            finally:
                if peer is not None:
                    await peer.close()

    def _server(self, app: FastAPI) -> uvicorn.Server:
        keyfile = certfile = None
        if self.use_https:
            keyfile, certfile = _get_or_create_ssl_files(
                port=self.port, keyfile=self.ssl_keyfile, certfile=self.ssl_certfile
            )
        config = uvicorn.Config(
            app,
            host='0.0.0.0',
            port=self.port,
            log_level='warning',
            access_log=False,
            ssl_keyfile=keyfile,
            ssl_certfile=certfile,
        )
        return uvicorn.Server(config)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        app = FastAPI()
        self._add_page_routes(app)
        self._add_controller_route(app, should_stop)
        server = self._server(app)
        self.server_thread = threading.Thread(target=server.run, daemon=True)
        self.server_thread.start()

        primary_host = utils.resolve_host_ip()
        url = _build_access_url(primary_host, self.port, self.use_https)
        banner = '=' * 80
        print(banner)
        print(f' >>> WEBXR interface available at: {url} <<<')
        addresses = ', '.join(f'{name} {address}' for name, address in utils.interface_ipv4_addresses())
        print(f'     The addresses of this machine: {addresses or "none"}. Use the one that the phone can reach.')
        print(banner)

        while not should_stop.value:
            yield pimm.Sleep(0.1)
            if not self.server_thread.is_alive():
                raise RuntimeError('WebXR server thread died')

        server.should_exit = True
        self.server_thread.join()
