import asyncio
import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pimm
from positronic import utils
from positronic.policy.harness import Directive


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


def _resize_to_height(rgb: np.ndarray, height: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    width = _even(round(w * height / h))
    if (h, w) == (height, width):
        return rgb
    return (
        av.VideoFrame.from_ndarray(rgb, format='rgb24').reformat(width=width, height=height).to_ndarray(format='rgb24')
    )


def _tile(frames: list[np.ndarray]) -> np.ndarray:
    """Lay frames out side by side at a common (even) height, so the row encodes as one H.264 stream."""
    height = _even(min(frame.shape[0] for frame in frames))
    return np.concatenate([_resize_to_height(frame, height) for frame in frames], axis=1)


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


class WebEvalUI(pimm.ControlSystem):
    """Headless web operator surface for attended evals.

    Tiles the live eval cameras into a single H.264 stream served to a browser and turns Start/Finish/Abort
    presses into harness directives. A drop-in directive source replacing the dearpygui/keyboard drivers,
    reachable over an SSH tunnel or directly on the host IP.
    """

    def __init__(self, port=8080, fps=30, keyframe_interval=15, bitrate=8_000_000):
        self.port = port
        self.fps = fps
        self.keyframe_interval = keyframe_interval
        self.bitrate = bitrate
        self.cameras = pimm.ReceiverDict(self, default=None)
        self.directive = pimm.ControlSystemEmitter(self)
        self._stream = _CameraStream(fps, keyframe_interval, bitrate)
        self._latest: dict[str, np.ndarray] = {}

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        templates = Jinja2Templates(directory=_pkg_path('templates'))
        names = list(self.cameras)

        app = FastAPI()
        app.mount('/static', StaticFiles(directory=_shared_static()), name='static')
        app.mount('/assets', StaticFiles(directory=_pkg_path('static')), name='assets')

        @app.get('/', response_class=HTMLResponse)
        async def index(request: Request):
            return templates.TemplateResponse(request, 'eval_console.html', {'cameras': names})

        @app.websocket('/video')
        async def video(websocket: WebSocket):
            await websocket.accept()
            subscriber = self._stream.subscribe()
            loop = asyncio.get_running_loop()
            try:
                while not self._stream.init_segment and not should_stop.value:
                    await asyncio.sleep(0.05)
                init = self._stream.init_segment
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
                self._stream.unsubscribe(subscriber)

        @app.post('/directive/{action}')
        async def directive(action: str):
            match action:
                case 'start':
                    self.directive.emit(Directive.RUN(), clock.now_ns())
                case 'finish':
                    self.directive.emit(Directive.FINISH(), clock.now_ns())
                case 'abort':
                    self.directive.emit(Directive.ABORT(), clock.now_ns())
                case _:
                    raise HTTPException(status_code=404)

        config = uvicorn.Config(app, host='0.0.0.0', port=self.port)
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
                        self._latest[name] = cam_msg.data.array
                        changed = True
                if changed and len(self._latest) == len(names):
                    self._stream.push(_tile([self._latest[name] for name in names]))
                if not server_thread.is_alive():
                    raise RuntimeError('Web eval server thread died')
                yield pimm.Sleep(1 / self.fps)
        finally:
            self._stream.close()
            server.should_exit = True
            server_thread.join()
