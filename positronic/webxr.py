# System that starts a webserver that collects tracking data from the WebXR page
# and outputs it further.

import asyncio
import multiprocessing

from fastapi import FastAPI, WebSocket
from starlette.responses import FileResponse
import uvicorn
import numpy as np
import turbojpeg
import base64

import ironic as ir

from ironic.utils import FPSCounter
from geom import Transform3D


def run_server(data_queue, frame_queue, port, ssl_keyfile, ssl_certfile):  # noqa: C901  Function is too complex
    app = FastAPI()

    async def get_latest_frame():
        frame = None
        while frame is None:
            try:
                frame = frame_queue.get(block=False)
            except multiprocessing.queues.Empty:
                await asyncio.sleep(1 / 30)

        return frame

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
                base64_frame = await get_latest_frame()
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
                data_queue.put((data, ir.system_clock()))
                fps.tick()
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            print("WebSocket connection closed")

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
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
    server.run()


@ir.ironic_system(input_ports=["frame"], output_ports=["transforms", "buttons"])
class WebXR(ir.ControlSystem):
    def __init__(self, port: int, ssl_keyfile: str = "key.pem", ssl_certfile: str = "cert.pem"):
        super().__init__()
        self.port = port
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile

        self.data_queue = multiprocessing.Queue(maxsize=10)
        self.frame_queue = multiprocessing.Queue(maxsize=1)
        self.server_process = None
        self.fps = ir.utils.FPSCounter("WebXR")
        self.jpeg_encoder = turbojpeg.TurboJPEG()

    async def setup(self):
        """Start the WebXR server process"""
        self.server_process = multiprocessing.Process(
            target=run_server,
            args=(self.data_queue, self.frame_queue, self.port, self.ssl_keyfile, self.ssl_certfile)
        )
        self.server_process.start()

    @ir.on_message("frame")
    async def update_frame(self, message: ir.Message):
        """Update the latest frame to be streamed"""
        if not self.frame_queue.full():
            buffer = self.jpeg_encoder.encode(message.data, quality=50)
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            self.frame_queue.put(base64_frame)

    async def cleanup(self):
        """Clean up the server process"""
        if self.server_process:
            print("Cancelling WebXR")
            self.server_process.terminate()
            self.server_process.join(timeout=5)
            if self.server_process.is_alive():
                print("WebXR did not terminate in time, terminating forcefully")
                self.server_process.kill()
            print("WebXR cancelled")

    async def step(self):
        data = None
        while not self.data_queue.empty():
            data, timestamp = self.data_queue.get()

        if data is None:
            return ir.State.ALIVE

        transforms_data = {'left': None, 'right': None}
        buttons_data = {'left': None, 'right': None}

        if data['controllers']['right'] is not None:
            pos = np.array(data['controllers']['right']['position'])
            quat = np.array(data['controllers']['right']['orientation'])
            buttons = np.array(data['controllers']['right']['buttons'])
            transform = Transform3D(pos, quat)
            transforms_data['right'] = transform
            buttons_data['right'] = buttons

        if data['controllers']['left'] is not None:
            pos = np.array(data['controllers']['left']['position'])
            quat = np.array(data['controllers']['left']['orientation'])
            buttons = np.array(data['controllers']['left']['buttons'])
            transform = Transform3D(pos, quat)
            transforms_data['left'] = transform
            buttons_data['left'] = buttons

        if transforms_data['left'] is not None or transforms_data['right'] is not None:
            await asyncio.gather(
                self.outs.transforms.write(ir.Message(transforms_data, timestamp)),
                self.outs.buttons.write(ir.Message(buttons_data, timestamp))
            )

        self.fps.tick()
        return ir.State.ALIVE
