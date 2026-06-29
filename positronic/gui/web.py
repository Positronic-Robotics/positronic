import asyncio
import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import av
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pimm
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

    def push(self, rgb) -> None:
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

    Streams the live eval cameras as H.264 video to a browser and turns Start/Finish/Abort presses into
    harness directives. A drop-in directive source replacing the dearpygui/keyboard drivers, served over
    an SSH tunnel.
    """

    def __init__(self, task=None, port=8080, fps=30, keyframe_interval=15, bitrate=4_000_000):
        self.task = task
        self.port = port
        self.fps = fps
        self.keyframe_interval = keyframe_interval
        self.bitrate = bitrate
        self.cameras = pimm.ReceiverDict(self, default=None)
        self.directive = pimm.ControlSystemEmitter(self)
        self._streams: dict[str, _CameraStream] = {}

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        templates = Jinja2Templates(directory=_pkg_path('templates'))
        self._streams = {name: _CameraStream(self.fps, self.keyframe_interval, self.bitrate) for name in self.cameras}

        app = FastAPI()
        app.mount('/static', StaticFiles(directory=_shared_static()), name='static')
        app.mount('/assets', StaticFiles(directory=_pkg_path('static')), name='assets')

        @app.get('/', response_class=HTMLResponse)
        async def index(request: Request):
            return templates.TemplateResponse(request, 'eval_console.html', {'cameras': list(self._streams)})

        @app.websocket('/video/{name}')
        async def video(websocket: WebSocket, name: str):
            stream = self._streams.get(name)
            if stream is None:
                await websocket.close(code=1003)
                return
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

        config = uvicorn.Config(app, host='127.0.0.1', port=self.port)
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

        try:
            while not should_stop.value:
                for name, camera in self.cameras.items():
                    cam_msg = camera.read()
                    if cam_msg.data is not None and cam_msg.updated:
                        self._streams[name].push(cam_msg.data.array)
                if not server_thread.is_alive():
                    raise RuntimeError('Web eval server thread died')
                yield pimm.Sleep(1 / self.fps)
        finally:
            for stream in self._streams.values():
                stream.close()
            server.should_exit = True
            server_thread.join()
