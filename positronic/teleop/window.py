"""The window of the teleop service on the station screen: a page that a browser on the station shows.

It tells the station that a trial is human, takes Ready and Cancel, and shows what the headset shows.
"""

import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .service import NotWaiting, Phase, TeleopService
from .session import Reply

STATIC_DIR = Path(__file__).resolve().parent / 'static'
_HOST = '127.0.0.1'


class WindowStatus(BaseModel):
    phase: Phase
    task: str
    control: bool
    attempt_started: bool
    cameras: list[str]


def window_app(service: TeleopService) -> FastAPI:
    app = FastAPI()
    # The window binds the loopback address. A page under another name, which a DNS rebind can point at it, gets no
    # answer, so the origin check below compares against a host that it can trust.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[_HOST, 'localhost'])

    @app.middleware('http')
    async def refuse_foreign_origins(request: Request, call_next):
        """A page on another site cannot ready or cancel a trial. A browser sends ``Origin`` with each cross-site
        POST, and a client that sends none passes."""
        origin = request.headers.get('origin')
        if request.method != 'GET' and origin is not None and urlparse(origin).netloc != request.url.netloc:
            return await http_exception_handler(request, HTTPException(403, 'cross-origin request refused'))
        return await call_next(request)

    @app.exception_handler(NotWaiting)
    async def conflict(request: Request, exc: NotWaiting):
        return await http_exception_handler(request, HTTPException(409, str(exc)))

    def status() -> WindowStatus:
        state = service.state()
        return WindowStatus(
            phase=state.phase,
            task=state.task,
            control=state.control,
            attempt_started=state.attempt_started,
            cameras=list(state.cameras),
        )

    @app.get('/')
    async def page():
        return FileResponse(STATIC_DIR / 'window.html')

    @app.get('/status')
    async def get_status() -> WindowStatus:
        return status()

    @app.post('/ready')
    async def ready() -> WindowStatus:
        service.answer_ready(Reply.READY)
        return status()

    @app.post('/cancel')
    async def cancel() -> WindowStatus:
        service.answer_ready(Reply.CANCELLED)
        return status()

    @app.get('/camera/{name}.jpg')
    async def camera(name: str) -> Response:
        try:
            image = service.camera_image(name)
        except KeyError:
            raise HTTPException(404, f'the run sends no camera {name}') from None
        encoded, jpeg = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        assert encoded, 'OpenCV encodes an RGB image as JPEG'
        return Response(jpeg.tobytes(), media_type='image/jpeg', headers={'cache-control': 'no-store'})

    return app


class Window:
    """Serve ``app`` on ``127.0.0.1:port`` while the context is open. A station browser opens ``url``."""

    START_TIMEOUT_SEC = 10.0

    def __init__(self, app: FastAPI, port: int):
        self.url = f'http://{_HOST}:{port}/'
        self._server = uvicorn.Server(uvicorn.Config(app, host=_HOST, port=port, log_level='warning'))
        self._thread = threading.Thread(target=self._server.run, name='teleop-window', daemon=True)

    def __enter__(self) -> 'Window':
        self._thread.start()
        deadline = time.monotonic() + self.START_TIMEOUT_SEC
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError(f'the teleop window could not serve on {self.url}: the log above names the cause')
            if time.monotonic() > deadline:
                raise TimeoutError(f'the teleop window did not start on {self.url}')
            time.sleep(0.01)
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
