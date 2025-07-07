import asyncio
import base64
import threading
import traceback

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import uvicorn
import turbojpeg

import ironic2 as ir
import geom
from ironic.utils import FPSCounter


def _parse_controller_data(data: dict):
    controller_positions = {'left': None, 'right': None}
    buttons_dict = {'left': None, 'right': None}
    for side in ['right', 'left']:
        if data['controllers'][side] is not None:
            translation = np.array(data['controllers'][side]['position'])
            rotation = np.array(data['controllers'][side]['orientation'])
            buttons = np.array(data['controllers'][side]['buttons'])
            controller_positions[side] = geom.Transform3D(translation, rotation)
            buttons_dict[side] = buttons

    return controller_positions, buttons_dict


class WebXR:

    frame: ir.SignalReader = ir.NoOpReader()
    controller_positions: ir.SignalEmitter = ir.NoOpEmitter()
    buttons: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(self,
                 port: int,
                 ssl_keyfile: str = "key.pem",
                 ssl_certfile: str = "cert.pem"):
        self.port = port
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        self.server_thread = None
        self.jpeg_encoder = turbojpeg.TurboJPEG()

    def run(self, should_stop: ir.SignalReader, clock: ir.Clock):  # noqa: C901  Function is too complex
        app = FastAPI()

        def encode_frame(image):
            buffer = self.jpeg_encoder.encode(image, quality=50)
            return base64.b64encode(buffer).decode('utf-8')

        @app.get("/")
        async def root():
            return FileResponse("positronic/assets/webxr/index.html")

        @app.get("/three.min.js")
        async def three_min():
            return FileResponse("positronic/assets/webxr/three.min.js")

        @app.get("/webxr-button.js")
        async def webxr_button():
            return FileResponse("positronic/assets/webxr/webxr-button.js")

        @app.get("/video-player.js")
        async def video_player():
            return FileResponse("positronic/assets/webxr/video-player.js")

        @app.websocket("/video")
        async def video_stream(websocket: WebSocket):
            await websocket.accept()
            print("Video WebSocket connection accepted")
            try:
                fps = FPSCounter("Video Stream")
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
                fps = FPSCounter("Websocket")
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

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.port,
            ssl_keyfile=self.ssl_keyfile,
            ssl_certfile=self.ssl_certfile,
            log_config={
                'version': 1,
                'disable_existing_loggers': False,
                'handlers': {
                    'file': {
                        'class': 'logging.FileHandler',
                        'formatter': 'default',
                        'filename': 'webxr.log',
                        'mode': 'w',
                    },
                    'console': {
                        'class': 'logging.FileHandler',
                        'formatter': 'default',
                        'filename': 'webxr.log',
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
            },
        )
        server = uvicorn.Server(config)
        self.server_thread = threading.Thread(target=server.run, daemon=True)
        self.server_thread.start()

        while not should_stop.value:
            yield 0.1

        server.should_exit = True
        self.server_thread.join()
