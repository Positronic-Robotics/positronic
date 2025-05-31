import base64

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import uvicorn

import ironic2 as ir
from geom import Transform3D
from ironic.utils import FPSCounter


def _parse_controller_data(self, data: dict):
    translation = np.array(data['position'])
    rotation = np.array(data['orientation'])
    buttons = np.array(data['buttons'])

    controller_position = Transform3D(translation, rotation)
    return controller_position, buttons


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

    def run(self, _should_stop: ir.SignalReader):  # noqa: C901  Function is too complex
        app = FastAPI()

        async def get_latest_frame_b64():
            msg = self.frame.value()
            if msg is ir.NoValue:
                return None

            buffer = self.jpeg_encoder.encode(msg.data, quality=50)
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
                while True:
                    base64_frame = await get_latest_frame_b64()
                    await websocket.send_text(base64_frame)
                    fps.tick()

            except Exception as e:
                print(f"Video WebSocket error: {e}")
            finally:
                print("Video WebSocket connection closed")

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            print("WebSocket connection accepted")
            try:
                fps = FPSCounter("Websocket")
                while True:
                    data = await websocket.receive_json()
                    controller_positions, buttons = self._parse_controller_data(data)
                    ts = ir.system_clock()
                    self.controller_positions.emit(ir.Message(controller_positions, ts))
                    self.buttons.emit(ir.Message(buttons, ts))
                    fps.tick()
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                print("WebSocket connection closed")

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
        # TODO: This is a blocking call. In order to terminate the server, someone need to set
        #       server.should_exit = True. One way to do this is to spawn a thread to check for should_stop.
        server.run()
