import asyncio
import json
import logging
import os
import subprocess
import tempfile
import threading
import traceback
from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

import pimm
from positronic import geom, utils
from positronic.gui.video_stream import AnnexBEncoder

logger = logging.getLogger(__name__)

# The names of the controllers in the positions and the buttons that ``WebXR`` emits.
LEFT, RIGHT = 'left', 'right'

# The routes of the page state and of the controller stream.
STATE_PATH, CONTROLLER_PATH, INPUT_PATH = '/state', '/ws', '/input'
# The frame rate that the video encoder spreads its bit rate over. A key frame comes once in this many frames.
VIDEO_FPS = 30

# The fields of a message of the page on ``/ws``. The orientation is a wxyz quaternion.
_CONTROLLERS, _POSITION, _ORIENTATION, _BUTTONS = 'controllers', 'position', 'orientation', 'buttons'


def parse_controller_message(
    data: Mapping[str, Any],
) -> tuple[dict[str, geom.Transform3D | None], dict[str, np.ndarray | None]]:
    """The poses and the buttons of the controllers in a message of the page on ``/ws``, the inverse of
    ``controller_message``. A controller that the page does not track has ``None`` for both."""
    controller_positions: dict[str, geom.Transform3D | None] = {LEFT: None, RIGHT: None}
    buttons_dict: dict[str, np.ndarray | None] = {LEFT: None, RIGHT: None}
    for side in [RIGHT, LEFT]:
        if data[_CONTROLLERS][side] is not None:
            translation = np.array(data[_CONTROLLERS][side][_POSITION], dtype=np.float64)
            rotation = np.array(data[_CONTROLLERS][side][_ORIENTATION], dtype=np.float64)
            buttons = np.array(data[_CONTROLLERS][side][_BUTTONS], dtype=np.float64)
            controller_positions[side] = geom.Transform3D(translation, geom.Rotation.from_quat(rotation))
            buttons_dict[side] = buttons

    return controller_positions, buttons_dict


def controller_message(
    positions: Mapping[str, geom.Transform3D], buttons: Mapping[str, Sequence[float]]
) -> dict[str, Any]:
    """The message of the page on ``/ws`` for the controllers that ``positions`` and ``buttons`` name. The page
    does not track the others."""
    if positions.keys() != buttons.keys():
        raise ValueError(
            f'positions and buttons must name the same controllers, got {list(positions)}, {list(buttons)}'
        )
    tracked = {
        side: {
            _POSITION: pose.translation.tolist(),
            _ORIENTATION: pose.rotation.as_quat.tolist(),
            _BUTTONS: [float(value) for value in buttons[side]],
        }
        for side, pose in positions.items()
    }
    return {_CONTROLLERS: {side: tracked.get(side) for side in (LEFT, RIGHT)}}


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


# The query that shows the phone page with the labels of ``positronic teleop``.
PHONE_TELEOP_QUERY = 'mode=teleop'


class Frontend(StrEnum):
    """The page that ``WebXR`` serves."""

    HEADSET = 'headset'
    IPHONE = 'iphone'
    # The phone page with the labels of ``positronic teleop``.
    IPHONE_TELEOP = 'iphone_teleop'

    @property
    def url_query(self) -> str:
        """The query of the page URL, empty when the page has none."""
        return PHONE_TELEOP_QUERY if self is Frontend.IPHONE_TELEOP else ''


class SessionMode(StrEnum):
    """The WebXR session that the page requests."""

    AR = 'immersive-ar'
    VR = 'immersive-vr'


def _build_access_url(host: str, port: int, use_https: bool) -> str:
    scheme = 'https' if use_https else 'http'
    return f'{scheme}://{host}:{port}/'


class WebXR(pimm.ControlSystem):
    """WebXR server for a headset or an iPhone AR controller.

    Serves a single frontend at a time and streams controller/pose data back
    to the application over WebSocket, with an optional H.264 video stream.

    Important for iPhone
    - Use a WebXR-capable browser (XR Browser recommended).
    - For development, use HTTP (`use_https=False`). XR Browser on iOS blocks self-signed
      HTTPS and may not allow proceeding to an untrusted site, which also
      blocks the `wss://` WebSocket.
    - If HTTPS is required, provide a cert trusted by the device.

    Endpoints
    - `/`                -> selected frontend; it redirects to `/?mode=teleop` for phone teleop, and to `/` for
                            any other query
    - `/state`           -> the last `page_state` value as JSON
    - `/input`           -> POST a JSON object from the page, such as a choice of the teleoperator; ``page_input``
                            emits it
    - `/ws`              -> JSON controller stream from client to server
    - `/video`           -> optional H.264 video (server to client): a JSON ``DecoderConfig`` text message before
                            each key frame, and each frame as a binary Annex B access unit
    - Shared assets: `/three.min.js`, `/webxr-button.js`, `/core.js`, `/session-mode.js`
    - Headset-only: `/video-player.js`

    Parameters
    - port: TCP port to bind.
    - ssl_keyfile, ssl_certfile: TLS files if `use_https=True`.
    - frontend: the page to serve, a ``Frontend``.
    - session_mode: the WebXR session that the page requests, a ``SessionMode``.
    - use_https: enable HTTPS; set False for iPhone dev.
    - video_bitrate: the target bit rate of the video stream, in bits per second.
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
        video_bitrate: int = 4_000_000,
    ):
        utils.check_positive_int(video_bitrate, 'video_bitrate')
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
        self.video_bitrate = video_bitrate

    def _next_frame(self, last_sent_time: pimm.Time | None) -> pimm.Message | None:
        """The frame the video stream sends next. ``None`` while no new frame arrived or the camera is lost."""
        msg = self.frame.read()
        if msg is None or isinstance(msg.data, pimm.SignalError) or msg.time == last_sent_time:
            return None
        return msg

    def _add_page_routes(self, app: FastAPI) -> None:
        """The page and its scripts, from the package assets."""

        @app.get('/')
        async def root(request: Request):
            query = self.frontend.url_query
            if request.url.query != query:
                return RedirectResponse(f'/?{query}' if query else '/')
            if self.frontend is Frontend.HEADSET:
                return FileResponse(utils.package_assets_path('assets/webxr/index.html'))
            return FileResponse(utils.package_assets_path('assets/webxr_iphone/index.html'))

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

        @app.get('/three.min.js')
        async def three_min():
            return FileResponse(utils.package_assets_path('assets/webxr/three.min.js'))

        @app.get('/webxr-button.js')
        async def webxr_button():
            return FileResponse(utils.package_assets_path('assets/webxr/webxr-button.js'))

        @app.get('/core.js')
        async def webxr_core():
            return FileResponse(utils.package_assets_path('assets/webxr/core.js'))

        @app.get('/session-mode.js')
        async def session_mode():
            return Response(
                f'export const SESSION_MODE = {json.dumps(self.session_mode.value)};\n', media_type='text/javascript'
            )

        if self.frontend is Frontend.HEADSET:

            @app.get('/video-player.js')
            async def video_player():
                return FileResponse(utils.package_assets_path('assets/webxr/video-player.js'))

    def _add_video_route(self, app: FastAPI, should_stop: pimm.SignalReceiver) -> None:
        """``/video``: each new frame as H.264. Each socket has its own encoder, so its stream starts at a key frame."""

        @app.websocket('/video')
        async def video_stream(websocket: WebSocket):
            await websocket.accept()
            logger.info('Video WebSocket connection accepted')
            # The page sends nothing on `/video`, so the first message it receives is the disconnect. Without a
            # receive, a closed socket stays unseen until the next frame goes out.
            closed = asyncio.ensure_future(websocket.receive())
            try:
                encoder = AnnexBEncoder(VIDEO_FPS, VIDEO_FPS, self.video_bitrate)
                fps = pimm.utils.RateCounter('Video Stream')
                last_sent_time = None
                while not should_stop.value and not closed.done():
                    await asyncio.sleep(1 / 60)

                    msg = self._next_frame(last_sent_time)
                    if msg is None:
                        continue
                    last_sent_time = msg.time
                    for unit in encoder.encode(msg.data):
                        if unit.config is not None:
                            await websocket.send_text(unit.config.model_dump_json())
                        await websocket.send_bytes(unit.data)
                    fps.tick()
            except Exception as e:
                logger.error(f'Video WebSocket error: {e}')
                logger.error(traceback.format_exc())
            finally:
                closed.cancel()
                logger.info('Video WebSocket connection closed')

    def _add_controller_route(self, app: FastAPI, should_stop: pimm.SignalReceiver) -> None:
        """``/ws``: the controller stream from the page."""

        # TODO: Require a pairing key before `/ws` accepts a client. Any client that can reach the port sends input.
        @app.websocket(CONTROLLER_PATH)
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            logger.info('WebSocket connection accepted')
            try:
                fps = pimm.utils.RateCounter('Websocket')
                while not should_stop.value:
                    try:
                        # Use asyncio.wait_for with timeout to avoid blocking indefinitely
                        data = await asyncio.wait_for(websocket.receive_json(), timeout=1)
                        controller_positions, buttons = parse_controller_message(data)

                        # apply scaling
                        for transform in controller_positions.values():
                            if transform is not None:
                                transform.translation *= self.sensitivity

                        if controller_positions[LEFT] is not None or controller_positions[RIGHT] is not None:
                            self.controller_positions.emit(controller_positions)
                        if buttons[LEFT] is not None or buttons[RIGHT] is not None:
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
        self._add_video_route(app, should_stop)
        self._add_controller_route(app, should_stop)
        server = self._server(app)
        self.server_thread = threading.Thread(target=server.run, daemon=True)
        self.server_thread.start()

        primary_host = utils.resolve_host_ip()
        url = _build_access_url(primary_host, self.port, self.use_https)
        if self.frontend.url_query:
            url += f'?{self.frontend.url_query}'
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
