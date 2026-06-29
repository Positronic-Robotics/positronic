import asyncio
import threading
from collections.abc import Iterator

import turbojpeg
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

import pimm
from positronic.policy.harness import Directive

_LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {'class': 'logging.FileHandler', 'formatter': 'default', 'filename': '/tmp/web_eval.log', 'mode': 'w'}
    },
    'loggers': {'': {'handlers': ['file'], 'level': 'INFO'}},
    'formatters': {'default': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}},
}


class WebEvalUI(pimm.ControlSystem):
    """Headless web operator surface for attended evals.

    Streams the live eval cameras as MJPEG to a browser and turns Start/Finish/Abort presses into
    harness directives. A drop-in directive source replacing the dearpygui/keyboard drivers, served
    over an SSH tunnel.
    """

    def __init__(self, task=None, port=8080, fps=30, jpeg_quality=50):
        self.task = task
        self.port = port
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.cameras = pimm.ReceiverDict(self, default=None)
        self.directive = pimm.ControlSystemEmitter(self)
        self._frames: dict[str, bytes] = {}

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        app = FastAPI()
        jpeg_encoder = turbojpeg.TurboJPEG()

        @app.get('/')
        async def index():
            feeds = ''.join(f'<img src="/stream/{name}" alt="{name}">' for name in self.cameras)
            return HTMLResponse(
                '<!doctype html><html><head><meta charset="utf-8"><title>Eval console</title></head><body>'
                f'<div class="feeds">{feeds}</div>'
                '<div class="controls">'
                '<button data-action="start">Start</button>'
                '<button data-action="finish">Finish</button>'
                '<button data-action="abort">Abort</button>'
                '</div>'
                '<script>'
                "for (const b of document.querySelectorAll('button')) "
                "b.onclick = () => fetch('/directive/' + b.dataset.action, {method: 'POST'});"
                '</script>'
                '</body></html>'
            )

        @app.get('/stream/{name}')
        async def stream(name: str):
            if name not in self.cameras:
                raise HTTPException(status_code=404)

            async def frames():
                last = None
                while not should_stop.value:
                    await asyncio.sleep(1 / self.fps)
                    jpeg = self._frames.get(name)
                    if jpeg is None or jpeg is last:
                        continue
                    last = jpeg
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'

            return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame')

        @app.post('/directive/{action}')
        async def directive(action: str):
            match action:
                case 'start':
                    self.directive.emit(Directive.RUN(task=self.task), clock.now_ns())
                case 'finish':
                    self.directive.emit(Directive.FINISH(), clock.now_ns())
                case 'abort':
                    self.directive.emit(Directive.ABORT(), clock.now_ns())
                case _:
                    raise HTTPException(status_code=404)

        config = uvicorn.Config(app, host='127.0.0.1', port=self.port, log_config=_LOG_CONFIG)
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        banner = '=' * 80
        print(banner)
        print(
            f' >>> WEB eval console: http://127.0.0.1:{self.port}/  '
            f'(tunnel with: ssh -L {self.port}:localhost:{self.port} <robot-host>) <<<'
        )
        print(banner)

        while not should_stop.value:
            for cam_name, camera in self.cameras.items():
                cam_msg = camera.read()
                if cam_msg.data is not None and cam_msg.updated:
                    self._frames[cam_name] = jpeg_encoder.encode(cam_msg.data.array, quality=self.jpeg_quality)
            if not server_thread.is_alive():
                raise RuntimeError('Web eval server thread died')
            yield pimm.Sleep(1 / self.fps)

        server.should_exit = True
        server_thread.join()
