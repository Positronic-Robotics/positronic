import asyncio
import dataclasses
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import pimm
from positronic import geom, keys, utils
from positronic.drivers.roboarm import command
from positronic.policy.harness import Directive, HarnessStatus, Phase


class _Axis(StrEnum):
    """A jog axis, in the frame the console's key map is expressed in."""

    X = 'x'
    Y = 'y'
    Z = 'z'
    RX = 'rx'
    RY = 'ry'
    RZ = 'rz'


class _Step(StrEnum):
    """Which of the two configured step sizes a jog moves by."""

    FINE = 'fine'
    COARSE = 'coarse'


class _JogBody(BaseModel):
    axis: _Axis
    sign: Literal[-1, 1]
    scale: _Step


class _GripBody(BaseModel):
    value: float = Field(ge=0.0, le=1.0)


_TRANSLATION_AXES = {_Axis.X: 0, _Axis.Y: 1, _Axis.Z: 2}
_ROTATION_AXES = {_Axis.RX: 0, _Axis.RY: 1, _Axis.RZ: 2}


def _pkg_path(*parts: str) -> str:
    return str(Path(__file__).resolve().parent.joinpath(*parts))


def _shared_static() -> str:
    return str(Path(__file__).resolve().parent.parent / 'server' / 'static')


def _next_fragment(subscriber: queue.Queue) -> bytes | None:
    try:
        return subscriber.get(timeout=1.0)
    except queue.Empty:
        return None


def _codec_string(init: bytes) -> str:
    """Build the MSE codec string (``avc1.PPCCLL``) from the avcC box of a fragmented-MP4 init segment."""
    record = init[init.find(b'avcC') + 4 :]
    return f'avc1.{record[1]:02X}{record[2]:02X}{record[3]:02X}'


def _even(value: int) -> int:
    return max(2, value - value % 2)


def _resize_to_width(rgb: np.ndarray, width: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    height = _even(round(h * width / w))
    if (h, w) == (height, width):
        return rgb
    return (
        av.VideoFrame.from_ndarray(rgb, format='rgb24').reformat(width=width, height=height).to_ndarray(format='rgb24')
    )


def _tile(frames: list[np.ndarray], width: int) -> np.ndarray:
    """Stack frames vertically at a common (even) width, so the column encodes as one H.264 stream."""
    width = _even(width)
    return np.concatenate([_resize_to_width(frame, width) for frame in frames], axis=0)


class _ChunkBuffer:
    """Write-only sink that hands the muxer's output back to the producer one drain at a time."""

    def __init__(self):
        self._chunks: list[bytes] = []
        self._pos = 0

    def write(self, data) -> int:
        self._chunks.append(bytes(data))
        self._pos += len(data)
        return len(data)

    def drain(self) -> bytes:
        data = b''.join(self._chunks)
        self._chunks.clear()
        return data

    def tell(self) -> int:
        return self._pos

    def flush(self) -> None:
        pass


class _CameraStream:
    """Encodes RGB frames to a fragmented-MP4 H.264 byte stream and fans the fragments out to subscribers.

    The encoder runs in the producer thread; each completed fragment starts at a keyframe (``frag_keyframe``),
    so it is independently decodable after the init segment and a late subscriber can join at any fragment.
    """

    def __init__(self, fps: int, keyframe_interval: int, bitrate: int):
        self._fps = fps
        self._keyframe_interval = keyframe_interval
        self._bitrate = bitrate
        self._buffer = _ChunkBuffer()
        self._container = None
        self._stream = None
        self._init = b''
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue] = set()

    def _open(self, height: int, width: int) -> None:
        self._container = av.open(
            self._buffer, mode='w', format='mp4', options={'movflags': 'frag_keyframe+empty_moov+default_base_moof'}
        )
        stream = self._container.add_stream(
            'libx264', rate=self._fps, options={'preset': 'ultrafast', 'tune': 'zerolatency', 'profile': 'baseline'}
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        stream.gop_size = self._keyframe_interval
        stream.bit_rate = self._bitrate
        self._stream = stream

    def push(self, rgb: np.ndarray) -> None:
        if self._container is None:
            self._open(rgb.shape[0], rgb.shape[1])
        frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._dispatch(self._buffer.drain())

    def _dispatch(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            if not self._init:
                marker = data.find(b'moof')
                if marker < 4:
                    self._init += data
                    return
                self._init = data[: marker - 4]
                data = data[marker - 4 :]
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            if subscriber.full():
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
            subscriber.put(data)

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=self._fps)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    @property
    def init_segment(self) -> bytes:
        with self._lock:
            return self._init

    def close(self) -> None:
        if self._container is None:
            return
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()


class _WrapUpState(StrEnum):
    IDLE = 'idle'
    FINALIZING = 'finalizing'
    FAILED = 'failed'


@dataclass(frozen=True)
class _WrapUpStatus:
    """How the run-level wrap-up is going, published on GET /wrap_up for the console's overlay."""

    state: _WrapUpState
    detail: str = ''


class WebEvalUI(pimm.ControlSystem):
    """Headless web operator surface for attended evals.

    Tiles the live eval cameras into a single H.264 stream served to a browser and turns Start/Finish/Abort
    presses into harness directives. A drop-in directive source replacing the dearpygui/keyboard drivers,
    reachable over an SSH tunnel or directly on the host IP.
    """

    def __init__(
        self,
        task: str | None = None,
        port=8080,
        fps=20,
        width=640,
        keyframe_interval=15,
        bitrate=2_000_000,
        translation_fine=0.01,
        translation_coarse=0.05,
        rotation_fine=2.0,
        rotation_coarse=10.0,
    ):
        self.task = task
        self.port = port
        self.fps = fps
        self.width = width
        self.keyframe_interval = keyframe_interval
        self.bitrate = bitrate
        self.translation_fine = translation_fine
        self.translation_coarse = translation_coarse
        self.rotation_fine = rotation_fine
        self.rotation_coarse = rotation_coarse
        self.cameras = pimm.ReceiverDict(self, default=None)
        self.directive = pimm.ControlSystemEmitter(self)
        self.manual_command = pimm.ControlSystemEmitter(self)
        # Live harness/policy status for the badge (fed by Harness.status; may be unconnected).
        self.status = pimm.ControlSystemReceiver(self, default=None)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        templates = Jinja2Templates(directory=_pkg_path('templates'))
        names = list(self.cameras)
        stream = _CameraStream(self.fps, self.keyframe_interval, self.bitrate)
        latest: dict[str, np.ndarray] = {}
        # Read by the endpoints on the server thread and rebound by the run loop below: one reference
        # swap, which the GIL makes atomic for this poll-and-render use.
        status: HarnessStatus | None = None
        wrap_up_status = _WrapUpStatus(_WrapUpState.IDLE)
        # The last home a human confirmed the arm reached. Idle means the harness ISSUED the home, not
        # that the arm is there, and nothing reports the arrival — so teleop waits on this, and it lives
        # here rather than in a page, which a reload would clear while the arm is still travelling.
        confirmed_homes = 0

        app = FastAPI()
        app.mount('/static', StaticFiles(directory=_shared_static()), name='static')
        app.mount('/assets', StaticFiles(directory=_pkg_path('static')), name='assets')

        @app.get('/', response_class=HTMLResponse)
        async def index(request: Request):
            return templates.TemplateResponse(request, 'eval_console.html')

        @app.websocket('/video')
        async def video(websocket: WebSocket):
            await websocket.accept()
            subscriber = stream.subscribe()
            loop = asyncio.get_running_loop()
            try:
                while not stream.init_segment and not should_stop.value:
                    await asyncio.sleep(0.05)
                init = stream.init_segment
                if not init:
                    return
                await websocket.send_text(_codec_string(init))
                await websocket.send_bytes(init)
                while not should_stop.value:
                    fragment = await loop.run_in_executor(None, _next_fragment, subscriber)
                    if fragment is not None:
                        await websocket.send_bytes(fragment)
            except WebSocketDisconnect:
                pass
            finally:
                stream.unsubscribe(subscriber)

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

        @app.post('/jog')
        async def jog(body: _JogBody):
            if body.axis in _TRANSLATION_AXES:
                step = self.translation_fine if body.scale is _Step.FINE else self.translation_coarse
                translation = np.zeros(3)
                translation[_TRANSLATION_AXES[body.axis]] = body.sign * step
                delta = geom.Transform3D(translation=translation)
            elif body.axis in _ROTATION_AXES:
                angle = self.rotation_fine if body.scale is _Step.FINE else self.rotation_coarse
                rotvec = np.zeros(3)
                rotvec[_ROTATION_AXES[body.axis]] = np.deg2rad(body.sign * angle)
                delta = geom.Transform3D(rotation=geom.Rotation.from_rotvec(rotvec))
            self.manual_command.emit({keys.ROBOT_COMMAND: command.CartesianDelta(delta)}, clock.now_ns())

        @app.post('/grip')
        async def grip(body: _GripBody):
            self.manual_command.emit({keys.TARGET_GRIP: body.value}, clock.now_ns())

        @app.get('/status')
        async def get_status():
            # A harness that has published nothing is not an idle harness: answering idle here would
            # enable teleop over a rollout whose status never reached this process.
            if status is None:
                raise HTTPException(status_code=503, detail='the harness has published no status')
            # Serialized together because the console reads them together; the wire is where a payload
            # may hold what two owners know.
            return {**dataclasses.asdict(status), 'awaiting_park': status.homes_commanded > confirmed_homes}

        @app.get('/wrap_up')
        async def get_wrap_up():
            return dataclasses.asdict(wrap_up_status)

        @app.post('/parked')
        async def parked():
            nonlocal confirmed_homes
            confirmed_homes = status.homes_commanded if status is not None else 0

        @app.get('/ping')
        async def ping():  # tiny + no work, so its round-trip measures the browser<->robot-host link
            return {}

        # How long the harness gets to report the FINISH handled, and how long the home it then commands
        # gets to reach the arm. Tune the settle at the rig if a far pose needs longer to home.
        finalize_timeout_s = 30.0
        home_settle_s = 5.0

        @app.post('/finish_run')
        async def finish_run():
            # The World may only stop once the harness has acknowledged finalizing the live episode and
            # homing: stopping it directly aborts an open recording and leaves the arm where it stands.
            # Returns as soon as the wrap-up is scheduled; the console follows it on GET /wrap_up.
            async def _wrap_up():
                nonlocal wrap_up_status
                handled = status.directives_handled if status is not None else 0
                wrap_up_status = _WrapUpStatus(_WrapUpState.FINALIZING)
                self.directive.emit(Directive.FINISH(), clock.now_ns())
                for _ in range(int(finalize_timeout_s / 0.1)):
                    current = status
                    # The harness's own directive count is what marks this FINISH handled. An idle phase
                    # does not: the status is a periodic sample, so it can predate the emit above.
                    if current is not None and current.directives_handled > handled and current.phase == Phase.IDLE:
                        break
                    await asyncio.sleep(0.1)
                else:
                    # A stop here would abort the still-open recording and skip homing, the failures this
                    # path exists to prevent, so a wedged harness is left up for a human.
                    _fail(
                        f'The episode did not finalize within {finalize_timeout_s:.0f}s, so the run '
                        'was left up. Retry, or investigate the harness on the robot host.'
                    )
                    return
                await asyncio.sleep(home_settle_s)
                # Setting the event is the only stop that survives a nohup launch, where SIGINT is SIG_IGN.
                if isinstance(should_stop, pimm.world.EventReceiver):
                    should_stop._event.set()
                else:
                    _fail('The stop signal is not event-backed, so the run was left up. Stop it on the host.')

            def _fail(detail: str) -> None:
                nonlocal wrap_up_status
                wrap_up_status = _WrapUpStatus(_WrapUpState.FAILED, detail)
                print(f'finish_run: {detail}')

            asyncio.create_task(_wrap_up())
            return {'wrapping_up': True}

        # The legacy asyncio `websockets` backend drains the transport from its reader and keepalive
        # coroutines concurrently with our send loop, tripping an assertion that kills the feed. The
        # sans-io backend serializes every write through the event-loop transport instead.
        config = uvicorn.Config(app, host='0.0.0.0', port=self.port, ws='websockets-sansio')
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        host = utils.resolve_host_ip()
        banner = '=' * 80
        print(banner)
        print(f' >>> WEB eval console available at: http://{host}:{self.port}/ <<<')
        print(banner)

        try:
            while not should_stop.value:
                changed = False
                for name in names:
                    cam_msg = self.cameras[name].read()
                    if cam_msg.data is not None and cam_msg.updated:
                        latest[name] = cam_msg.data.array
                        changed = True
                status_msg = self.status.read()
                if status_msg is not None and status_msg.updated and status_msg.data is not None:
                    status = status_msg.data
                if changed and len(latest) == len(names):
                    stream.push(_tile([latest[name] for name in names], self.width))
                if not server_thread.is_alive():
                    raise RuntimeError('Web eval server thread died')
                yield pimm.Sleep(1 / self.fps)
        finally:
            stream.close()
            server.should_exit = True
            server_thread.join()
