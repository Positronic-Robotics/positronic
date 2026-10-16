"""The window of the teleop service on the station screen: a page that a browser on the station shows.

It tells the station that a trial is human, takes Ready and Cancel, and shows what the headset shows.
"""

import threading
from pathlib import Path
from urllib.parse import urlparse

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .service import NotWaiting, TeleopService
from .session import Reply

STATIC_DIR = Path(__file__).resolve().parent / 'static'


class WindowStatus(BaseModel):
    phase: str
    task: str
    control: bool
    attempt_started: bool


def window_app(service: TeleopService) -> FastAPI:
    app = FastAPI()

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
            phase=state.phase.name, task=state.task, control=state.control, attempt_started=state.attempt_started
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

    @app.get('/view.jpg')
    async def view() -> Response:
        encoded, jpeg = cv2.imencode('.jpg', cv2.cvtColor(service.view(), cv2.COLOR_RGB2BGR))
        assert encoded, 'OpenCV encodes an RGB image as JPEG'
        return Response(jpeg.tobytes(), media_type='image/jpeg', headers={'cache-control': 'no-store'})

    return app


class Window:
    """Serve ``app`` on ``127.0.0.1:port`` while the context is open. A station browser opens ``url``."""

    def __init__(self, app: FastAPI, port: int):
        self.url = f'http://127.0.0.1:{port}/'
        self._server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
        self._thread = threading.Thread(target=self._server.run, name='teleop-window', daemon=True)

    def __enter__(self) -> 'Window':
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
