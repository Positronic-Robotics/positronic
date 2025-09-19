import asyncio
import base64
import os
import subprocess
import tempfile
import threading
import traceback
from typing import Iterator

import numpy as np
import turbojpeg
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse


import pimm
from positronic import geom

_LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'default',
            'filename': '/tmp/webxr.log',
            'mode': 'w',
        }
    },
    'loggers': {
        '': {
            'handlers': ['file'],
            'level': 'INFO',
        }
    },
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        }
    }
}


def _parse_controller_data(data: dict):
    controller_positions = {'left': None, 'right': None}
    buttons_dict = {'left': None, 'right': None}
    for side in ['right', 'left']:
        if data['controllers'][side] is not None:
            translation = np.array(data['controllers'][side]['position'], dtype=np.float64)
            rotation = np.array(data['controllers'][side]['orientation'], dtype=np.float64)
            buttons = np.array(data['controllers'][side]['buttons'], dtype=np.float64)
            controller_positions[side] = geom.Transform3D(translation, rotation)
            buttons_dict[side] = buttons

    return controller_positions, buttons_dict


def _get_or_create_ssl_files(port: int, keyfile: str, certfile: str) -> tuple[str, str]:
    """Return paths to SSL key/cert, creating basic self-signed ones if missing.

    Note: For iPhone/XR Browser development, prefer HTTP (use_https=False).
    """
    if os.path.exists(keyfile) and os.path.exists(certfile):
        return keyfile, certfile

    tmp_dir = tempfile.gettempdir()
    tmp_key = os.path.join(tmp_dir, f"webxr_key_{port}.pem")
    tmp_cert = os.path.join(tmp_dir, f"webxr_cert_{port}.pem")

    if os.path.exists(tmp_key) and os.path.exists(tmp_cert):
        return tmp_key, tmp_cert

    try:
        cl = [
            "openssl", "req", "-x509", "-nodes", "-days", "365", "-newkey", "rsa:2048", "-keyout", tmp_key, "-out",
            tmp_cert, "-subj", "/CN=localhost"
        ],
        subprocess.run(*cl, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Generated self-signed SSL certs in {tmp_dir}")
    except Exception as e:
        print("Failed to generate SSL certificates via openssl. Provide cert/key files or use HTTP.")
        raise e

    return tmp_key, tmp_cert


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
    - `/`                -> selected frontend (no redirect)
    - `/ws`              -> JSON controller stream from client to server
    - `/video`           -> optional base64 JPEG frames (server to client)
    - Shared assets: `/three.min.js`, `/webxr-button.js`, `/core.js`
    - Oculus-only: `/video-player.js`

    Parameters
    - port: TCP port to bind.
    - ssl_keyfile, ssl_certfile: TLS files if `use_https=True`.
    - frontend: "oculus" or "iphone" (single-frontend mode).
    - use_https: enable HTTPS; set False for iPhone dev.
    """

    def __init__(self,
                 port: int,
                 ssl_keyfile: str = "key.pem",
                 ssl_certfile: str = "cert.pem",
                 frontend: str = "oculus",
                 use_https: bool = True):
        self.port = port
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        assert frontend in ("oculus", "iphone"), f"Unknown frontend: {frontend}"
        self.frontend = frontend
        self.use_https = use_https
        self.server_thread = None
        self.frame = pimm.ControlSystemReceiver(self)
        self.controller_positions = pimm.ControlSystemEmitter(self)
        self.buttons = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        app = FastAPI()
        jpeg_encoder = turbojpeg.TurboJPEG()

        def encode_frame(image):
            buffer = jpeg_encoder.encode(image, quality=50)
            return base64.b64encode(buffer).decode('utf-8')

        @app.get("/")
        async def root():
            if self.frontend == "oculus":
                return FileResponse("positronic/assets/webxr/index.html")
            else:
                return FileResponse("positronic/assets/webxr_iphone/index.html")

        @app.get("/three.min.js")
        async def three_min():
            return FileResponse("positronic/assets/webxr/three.min.js")

        @app.get("/webxr-button.js")
        async def webxr_button():
            return FileResponse("positronic/assets/webxr/webxr-button.js")

        @app.get("/core.js")
        async def webxr_core():
            return FileResponse("positronic/assets/webxr/core.js")

        if self.frontend == "oculus":

            @app.get("/video-player.js")
            async def video_player():
                return FileResponse("positronic/assets/webxr/video-player.js")

        @app.websocket("/video")
        async def video_stream(websocket: WebSocket):
            await websocket.accept()
            print("Video WebSocket connection accepted")
            try:
                fps = pimm.utils.RateCounter("Video Stream")
                last_sent_ts = None
                while not should_stop.value:
                    await asyncio.sleep(1 / 60)

                    msg = self.frame.read()

                    if msg is None:
                        continue

                    if last_sent_ts is not None and last_sent_ts == msg.ts:
                        continue
                    last_sent_ts = msg.ts
                    base64_frame = encode_frame(msg.data)
                    await websocket.send_text(base64_frame)
                    fps.tick()
            except Exception as e:
                print(f"Video WebSocket error: {e}")
                print(traceback.format_exc())
            finally:
                print("Video WebSocket connection closed")

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            print("WebSocket connection accepted")
            try:
                fps = pimm.utils.RateCounter("Websocket")
                while not should_stop.value:
                    try:
                        # Use asyncio.wait_for with timeout to avoid blocking indefinitely
                        data = await asyncio.wait_for(websocket.receive_json(), timeout=1)
                        controller_positions, buttons = _parse_controller_data(data)
                        ts = clock.now_ns()
                        if controller_positions['left'] is not None or controller_positions['right'] is not None:
                            self.controller_positions.emit(controller_positions, ts)
                        if buttons['left'] is not None or buttons['right'] is not None:
                            self.buttons.emit(buttons, ts)
                        fps.tick()
                    except asyncio.TimeoutError:
                        # Timeout is normal, just continue to check should_stop
                        continue
                    except Exception as e:
                        print(f"Error processing WebSocket message: {e}")
                        break
            except Exception as e:
                print(f"WebSocket error: {e}")

        ssl_kwargs = {}
        if self.use_https:
            keyfile, certfile = _get_or_create_ssl_files(port=self.port,
                                                         keyfile=self.ssl_keyfile,
                                                         certfile=self.ssl_certfile)
            ssl_kwargs = dict(ssl_keyfile=keyfile, ssl_certfile=certfile)
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_config=_LOG_CONFIG, **ssl_kwargs)
        server = uvicorn.Server(config)
        self.server_thread = threading.Thread(target=server.run, daemon=True)
        self.server_thread.start()

        while not should_stop.value:
            yield pimm.Sleep(0.1)
            if not self.server_thread.is_alive():
                raise RuntimeError("WebXR server thread died")

        server.should_exit = True
        self.server_thread.join()
