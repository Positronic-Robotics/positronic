import asyncio
import base64
import logging
import os
import subprocess
import tempfile
import threading
import traceback
from collections.abc import Iterator
from enum import StrEnum

import numpy as np
import turbojpeg
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

import pimm
from positronic import geom, utils

logger = logging.getLogger(__name__)

_LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {'class': 'logging.FileHandler', 'formatter': 'default', 'filename': '/tmp/webxr.log', 'mode': 'w'}
    },
    'loggers': {'': {'handlers': ['file'], 'level': 'INFO'}},
    'formatters': {'default': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}},
}


# The names of the controllers in the positions and the buttons that ``WebXR`` emits.
LEFT, RIGHT = 'left', 'right'


def _parse_controller_data(data: dict):
    controller_positions = {LEFT: None, RIGHT: None}
    buttons_dict = {LEFT: None, RIGHT: None}
    for side in [RIGHT, LEFT]:
        if data['controllers'][side] is not None:
            translation = np.array(data['controllers'][side]['position'], dtype=np.float64)
            rotation = np.array(data['controllers'][side]['orientation'], dtype=np.float64)
            buttons = np.array(data['controllers'][side]['buttons'], dtype=np.float64)
            controller_positions[side] = geom.Transform3D(translation, geom.Rotation.from_quat(rotation))
            buttons_dict[side] = buttons

    return controller_positions, buttons_dict


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


def encode_frame(encoder: turbojpeg.TurboJPEG, image: np.ndarray, quality: int) -> str:
    """An RGB ``image`` as base64 JPEG, as the video stream sends it."""
    buffer = encoder.encode(image, quality=quality, pixel_format=turbojpeg.TJPF_RGB)
    assert isinstance(buffer, bytes), 'TurboJPEG returns the JPEG bytes when it is given no destination'
    return base64.b64encode(buffer).decode('utf-8')


class Frontend(StrEnum):
    """The page that ``WebXR`` serves."""

    OCULUS = 'oculus'
    IPHONE = 'iphone'
    # The phone page with the labels of ``positronic teleop``.
    IPHONE_TELEOP = 'iphone_teleop'


# The query that shows the phone page with the labels of ``positronic teleop``.
PHONE_TELEOP_QUERY = 'mode=teleop'


def _build_access_url(host: str, port: int, use_https: bool) -> str:
    scheme = 'https' if use_https else 'http'
    return f'{scheme}://{host}:{port}/'


class WebXR(pimm.ControlSystem):
    """WebXR server for Oculus headset or iPhone AR controller.

    Serves a single frontend at a time and streams controller/pose data back
    to the application over WebSocket, with an optional JPEG video stream.

    Important for iPhone
    - Use a WebXR-capable browser (XR Browser recommended).
    - For development, use HTTP (`use_https=False`). XR Browser on iOS blocks self-signed
      HTTPS and may not allow proceeding to an untrusted site, which also
      blocks the `wss://` WebSocket.
    - If HTTPS is required, provide a cert trusted by the device.

    Endpoints
    - `/`                -> selected frontend; the phone teleop frontend redirects to `/?mode=teleop`
    - `/state`           -> the last `page_state` value as JSON
    - `/ws`              -> JSON controller stream from client to server
    - `/video`           -> optional base64 JPEG frames (server to client)
    - Shared assets: `/three.min.js`, `/webxr-button.js`, `/core.js`
    - Oculus-only: `/video-player.js`

    Parameters
    - port: TCP port to bind.
    - ssl_keyfile, ssl_certfile: TLS files if `use_https=True`.
    - frontend: the page to serve, a ``Frontend``.
    - use_https: enable HTTPS; set False for iPhone dev.
    - jpeg_quality: the JPEG quality of the video stream.
    """

    def __init__(
        self,
        port: int,
        ssl_keyfile: str = 'key.pem',
        ssl_certfile: str = 'cert.pem',
        frontend: Frontend = Frontend.OCULUS,
        use_https: bool = True,
        sensitivity: float = 1.0,
        jpeg_quality: int = 50,
    ):
        if type(jpeg_quality) is not int or not 0 <= jpeg_quality <= 100:
            raise ValueError(f'jpeg_quality must be an integer between 0 and 100, got {jpeg_quality!r}')
        self.port = port
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        self.frontend = frontend
        self.use_https = use_https
        self.server_thread = None
        self.frame = pimm.ControlSystemReceiver(self)
        # A JSON object that the page polls at ``/state``, such as the state of a trial. It is empty while nothing
        # sends one.
        self.page_state = pimm.ControlSystemReceiver(self)
        self.controller_positions = pimm.ControlSystemEmitter(self)
        self.buttons = pimm.ControlSystemEmitter(self)
        self.sensitivity = sensitivity
        self.jpeg_quality = jpeg_quality

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
            if self.frontend is Frontend.OCULUS:
                return FileResponse(utils.package_assets_path('assets/webxr/index.html'))
            if self.frontend is Frontend.IPHONE_TELEOP and request.url.query != PHONE_TELEOP_QUERY:
                return RedirectResponse(f'/?{PHONE_TELEOP_QUERY}')
            return FileResponse(utils.package_assets_path('assets/webxr_iphone/index.html'))

        @app.get('/state')
        async def page_state():
            message = self.page_state.read()
            return JSONResponse({} if message is None else message.data)

        @app.get('/three.min.js')
        async def three_min():
            return FileResponse(utils.package_assets_path('assets/webxr/three.min.js'))

        @app.get('/webxr-button.js')
        async def webxr_button():
            return FileResponse(utils.package_assets_path('assets/webxr/webxr-button.js'))

        @app.get('/core.js')
        async def webxr_core():
            return FileResponse(utils.package_assets_path('assets/webxr/core.js'))

        if self.frontend is Frontend.OCULUS:

            @app.get('/video-player.js')
            async def video_player():
                return FileResponse(utils.package_assets_path('assets/webxr/video-player.js'))

    def _add_video_route(self, app: FastAPI, should_stop: pimm.SignalReceiver) -> None:
        """``/video``: each new frame as base64 JPEG."""

        @app.websocket('/video')
        async def video_stream(websocket: WebSocket):
            await websocket.accept()
            logger.info('Video WebSocket connection accepted')
            try:
                # The video needs libturbojpeg, and the page does not. So the encoder loads only here.
                jpeg_encoder = turbojpeg.TurboJPEG()
                fps = pimm.utils.RateCounter('Video Stream')
                last_sent_time = None
                while not should_stop.value:
                    await asyncio.sleep(1 / 60)

                    msg = self._next_frame(last_sent_time)
                    if msg is None:
                        continue
                    last_sent_time = msg.time
                    base64_frame = encode_frame(jpeg_encoder, msg.data, self.jpeg_quality)
                    await websocket.send_text(base64_frame)
                    fps.tick()
            except Exception as e:
                logger.error(f'Video WebSocket error: {e}')
                logger.error(traceback.format_exc())
            finally:
                logger.info('Video WebSocket connection closed')

    def _add_controller_route(self, app: FastAPI, should_stop: pimm.SignalReceiver) -> None:
        """``/ws``: the controller stream from the page."""

        # TODO: Require a pairing key before `/ws` accepts a client. Any client that can reach the port sends input.
        @app.websocket('/ws')
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            logger.info('WebSocket connection accepted')
            try:
                fps = pimm.utils.RateCounter('Websocket')
                while not should_stop.value:
                    try:
                        # Use asyncio.wait_for with timeout to avoid blocking indefinitely
                        data = await asyncio.wait_for(websocket.receive_json(), timeout=1)
                        controller_positions, buttons = _parse_controller_data(data)

                        # apply scaling
                        for side, transform in controller_positions.items():
                            if transform is not None:
                                controller_positions[side].translation *= self.sensitivity

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
        ssl_kwargs = {}
        if self.use_https:
            keyfile, certfile = _get_or_create_ssl_files(
                port=self.port, keyfile=self.ssl_keyfile, certfile=self.ssl_certfile
            )
            ssl_kwargs = {'ssl_keyfile': keyfile, 'ssl_certfile': certfile}
        config = uvicorn.Config(app, host='0.0.0.0', port=self.port, log_config=_LOG_CONFIG, **ssl_kwargs)
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
        if self.frontend is Frontend.IPHONE_TELEOP:
            url += f'?{PHONE_TELEOP_QUERY}'
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
