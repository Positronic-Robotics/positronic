"""The station console: a browser page with a live tile per camera, the controls that start and end episodes, and
the control that ends the run.

``positronic-inference web`` composes the world around it. The page reads the console through ``GET /status``.
"""

import asyncio
import logging
import queue
import threading
import traceback
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exception_handlers import http_exception_handler
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.requests import HTTPConnection

import pimm
from positronic import keys
from positronic.eval import Task
from positronic.gui.station import Outcome, Refused, RunView, Station, Verdict, outcome_of
from positronic.gui.video_stream import VideoStream, codec_string

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / 'static'

# The console reads each camera at this rate, so a tile shows at most this many frames per second.
TILE_FPS = 15
TILE_WIDTH = 640
# A fragment ends at each keyframe, so a tile runs this many frames behind its camera.
TILE_KEYFRAME_INTERVAL = 8
TILE_BITRATE = 1_000_000
CAMERA_STALE_AFTER_S = 1.0


@dataclass(frozen=True)
class EndTrial:
    """Ends the open trial with ``payload`` as its ``done``."""

    payload: dict[str, Any]


Action = Task | EndTrial


class CameraView(BaseModel):
    # The observation key, which also names the tile's stream: ``/video/{name}``.
    name: str
    label: str
    live: bool
    fps: float
    # The size of the last frame. ``None`` until the first frame.
    width: int | None
    height: int | None


class Status(BaseModel):
    run: RunView
    cameras: list[CameraView]
    policy: str
    host: str


class InstructionBody(BaseModel):
    override: str | None


class EndBody(BaseModel):
    verdict: Verdict


class CameraFeed:
    """One camera's tile: the stream the page plays, and the arrival times that give its frame rate."""

    def __init__(self):
        self.stream = VideoStream(TILE_FPS, TILE_WIDTH, TILE_KEYFRAME_INTERVAL, TILE_BITRATE)
        self._arrivals: deque[float] = deque(maxlen=2 * TILE_FPS)
        self._size: tuple[int, int] | None = None
        self._lock = threading.Lock()

    def push(self, rgb: np.ndarray, now: float) -> None:
        self.stream.push(rgb)
        with self._lock:
            self._arrivals.append(now)
            self._size = (rgb.shape[1], rgb.shape[0])

    def view(self, name: str, now: float) -> CameraView:
        with self._lock:
            arrivals, size = list(self._arrivals), self._size
        live = bool(arrivals) and now - arrivals[-1] < CAMERA_STALE_AFTER_S
        span = arrivals[-1] - arrivals[0] if live else 0.0
        return CameraView(
            name=name,
            label=name.removeprefix(keys.IMAGE_PREFIX).replace('_', ' '),
            live=live,
            fps=(len(arrivals) - 1) / span if span > 0 else 0.0,
            width=size[0] if size else None,
            height=size[1] if size else None,
        )


class StationConsole(pimm.ControlSystem):
    """Serves the station page and turns its presses into episodes. End run on the page returns from its loop, and
    that stops the World.

    Connect each camera to ``cameras`` and ``trials`` to a ``TrialForwarder``, which runs each trial and its verdict.
    Schedule it as a background control system, so the harness never waits for the encoders or the web server.
    ``next_task`` makes the trials, and the page names the policy with ``policy``.
    """

    def __init__(self, next_task: Callable[[], Task], *, policy: str, host: str, port: int):
        self._next_task = next_task
        self._policy = policy
        self._host = host
        self._port = port
        self.cameras = pimm.ReceiverDict(self)
        self.trials = pimm.calls.ControlSystemCaller[Action, dict[str, Any]](self)
        # The open episode's answer.
        self._episode: pimm.calls.Answer[dict[str, Any]] | None = None

    @staticmethod
    def _raise_if_stopped(server_thread: threading.Thread) -> None:
        if not server_thread.is_alive():
            raise RuntimeError('The station console web server stopped')

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        station = Station(self._next_task)
        feeds = {name: CameraFeed() for name in self.cameras}
        actions: queue.SimpleQueue[Action] = queue.SimpleQueue()
        app = self.build_app(station, feeds, actions.put, clock, should_stop)
        # The legacy asyncio `websockets` backend drains the transport from its reader and keepalive coroutines
        # while the video loop sends, and an assertion then kills the feed. The sans-io backend serializes writes.
        config = uvicorn.Config(
            app, host=self._host, port=self._port, ws='websockets-sansio', log_level='warning', access_log=False
        )
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        try:
            while not server.started:
                self._raise_if_stopped(server_thread)
                yield pimm.Sleep(0.05)
            logger.info(f'Station console: http://{self._host}:{self._port}/')
            limiter = pimm.RateLimiter(clock, hz=TILE_FPS)
            while not should_stop.value:
                self._raise_if_stopped(server_thread)
                for name, feed in feeds.items():
                    if (frame := pimm.value_updated(self.cameras[name])) is not None:
                        feed.push(frame.array, clock.now())
                self._drive_episode(station, actions, clock)
                if station.run_ended:
                    logger.info('The operator ended the run')
                    return
                yield limiter.wait()
        finally:
            for feed in feeds.values():
                feed.stream.close()
            server.should_exit = True
            server_thread.join()

    def _drive_episode(self, station: Station, actions: queue.SimpleQueue[Action], clock: pimm.Clock) -> None:
        """Send what the page asked for, in order, and record the episode once the harness answers it."""
        while True:
            try:
                action = actions.get_nowait()
            except queue.Empty:
                break
            answer = self.trials(action)
            if isinstance(action, Task):
                self._episode = answer
        if self._episode is None or not self._episode.done():
            return
        episode, self._episode = self._episode, None
        try:  # rules-allow: swallowed-error — the page shows the error, and the operator decides whether to retry
            result = episode.result()
        except Exception as exc:
            logger.exception('Episode failed')
            station.close(Outcome.ERROR, clock.now(), error=''.join(traceback.format_exception_only(exc)).strip())
        else:
            station.close(outcome_of(result), clock.now())

    def build_app(
        self,
        station: Station,
        feeds: dict[str, CameraFeed],
        submit: Callable[[Action], None],
        clock: pimm.Clock,
        should_stop: pimm.SignalReceiver,
    ) -> FastAPI:
        """The console's HTTP surface. ``submit`` hands a trial or its end to the control loop."""
        app = FastAPI()

        def sent_from_another_site(connection: HTTPConnection) -> bool:
            """A browser sends ``Origin`` with each cross-site POST and each WebSocket handshake. A client that sends
            none passes."""
            origin = connection.headers.get('origin')
            return origin is not None and urlparse(origin).netloc != connection.url.netloc

        @app.middleware('http')
        async def refuse_foreign_origins(request: Request, call_next):
            """A page on another site cannot start or end an episode."""
            if request.method != 'GET' and sent_from_another_site(request):
                return await http_exception_handler(request, HTTPException(403, 'cross-origin request refused'))
            return await call_next(request)

        @app.exception_handler(Refused)
        async def conflict(request: Request, exc: Refused):
            return await http_exception_handler(request, HTTPException(409, str(exc)))

        def status() -> Status:
            now = clock.now()
            cameras = [feed.view(name, now) for name, feed in feeds.items()]
            return Status(run=station.view(now), cameras=cameras, policy=self._policy, host=self._host)

        @app.get('/status')
        async def get_status() -> Status:
            return status()

        @app.post('/instruction')
        async def instruction(body: InstructionBody) -> Status:
            station.set_override(body.override)
            return status()

        @app.post('/episode/start')
        async def start() -> Status:
            submit(station.start(clock.now()))
            return status()

        @app.post('/episode/end')
        async def end(body: EndBody) -> Status:
            submit(EndTrial(station.end(body.verdict)))
            return status()

        @app.post('/run/end')
        async def end_run() -> Status:
            station.end_run()
            return status()

        @app.websocket('/video/{name}')
        async def video(websocket: WebSocket, name: str):
            """The HTTP middleware does not see a WebSocket, so this refuses a page on another site itself."""
            feed = feeds.get(name)
            if feed is None or sent_from_another_site(websocket):
                await websocket.close()
                return
            await websocket.accept()
            await _stream(websocket, feed.stream, should_stop)

        # A mount at the root takes every path that reaches it, so it comes after the routes.
        app.mount('/', StaticFiles(directory=STATIC_DIR, html=True))
        return app


def _next_fragment(subscriber: queue.Queue[bytes]) -> bytes | None:
    try:
        return subscriber.get(timeout=1.0)
    except queue.Empty:
        return None


async def _stream(websocket: WebSocket, stream: VideoStream, should_stop: pimm.SignalReceiver) -> None:
    """Send the stream until the client leaves, the server closes the socket, or the run stops."""
    subscriber = stream.subscribe()
    sending = asyncio.create_task(_send_stream(websocket, stream, subscriber, should_stop))
    leaving = asyncio.create_task(_until_disconnect(websocket))
    try:
        done, _ = await asyncio.wait({sending, leaving}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        sending.cancel()
        leaving.cancel()
        stream.unsubscribe(subscriber)


async def _send_stream(
    websocket: WebSocket, stream: VideoStream, subscriber: queue.Queue[bytes], should_stop: pimm.SignalReceiver
) -> None:
    """Send the codec string, the init segment, and then each fragment until the run stops."""
    loop = asyncio.get_running_loop()
    try:
        while not stream.init_segment and not should_stop.value:
            await asyncio.sleep(0.05)
        init = stream.init_segment
        if not init:
            return
        await websocket.send_text(codec_string(init))
        await websocket.send_bytes(init)
        while not should_stop.value:
            fragment = await loop.run_in_executor(None, _next_fragment, subscriber)
            if fragment is not None:
                await websocket.send_bytes(fragment)
    except WebSocketDisconnect:
        pass


async def _until_disconnect(websocket: WebSocket) -> None:
    """Return once the client leaves or the server closes the socket. The page sends nothing on it."""
    async for _ in websocket.iter_bytes():
        pass
