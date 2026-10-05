"""Live H.264 for a browser: fragmented MP4 that Media Source Extensions can play from any fragment."""

import queue
import threading

import av
import numpy as np
from av.container import OutputContainer


def codec_string(init: bytes) -> str:
    """The MSE codec string (``avc1.PPCCLL``) that the avcC box of an init segment declares."""
    record = init[init.find(b'avcC') + 4 :]
    return f'avc1.{record[1]:02X}{record[2]:02X}{record[3]:02X}'


class _ChunkBuffer:
    """A write-only file for the muxer, emptied by the producer after each frame."""

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


class VideoStream:
    """Encodes RGB frames, scaled to ``width``, into one H.264 stream and gives each fragment to every subscriber.

    Each fragment starts at a keyframe, so a subscriber that joins late plays from the init segment and its first
    fragment. ``push`` and ``close`` run on one thread; the other methods are safe from any thread. A subscriber
    that falls behind loses its oldest fragment.

    FOOTGUN: call ``close``. The interpreter crashes at exit when it collects an open encoder.
    """

    def __init__(self, fps: int, width: int, keyframe_interval: int, bitrate: int):
        self._fps = fps
        self._width = width
        self._keyframe_interval = keyframe_interval
        self._bitrate = bitrate
        self._buffer = _ChunkBuffer()
        # Opened at the first frame, which gives the encoder its size.
        self._encoder: tuple[OutputContainer, av.VideoStream] | None = None
        # The muxer's output before the first fragment, which becomes the init segment when that fragment starts.
        self._header = b''
        self._init = b''
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue[bytes]] = set()

    def _open(self, height: int, width: int) -> tuple[OutputContainer, av.VideoStream]:
        container = av.open(
            self._buffer, mode='w', format='mp4', options={'movflags': 'frag_keyframe+empty_moov+default_base_moof'}
        )
        stream = container.add_stream(
            'libx264', rate=self._fps, options={'preset': 'ultrafast', 'tune': 'zerolatency', 'profile': 'baseline'}
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        stream.gop_size = self._keyframe_interval
        stream.bit_rate = self._bitrate
        return container, stream

    @staticmethod
    def _even(value: int) -> int:
        return max(2, value - value % 2)

    @staticmethod
    def _scaled_to_width(rgb: np.ndarray, width: int) -> np.ndarray:
        h, w = rgb.shape[:2]
        width = VideoStream._even(width)
        height = VideoStream._even(round(h * width / w))
        if (h, w) == (height, width):
            return rgb
        frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        return frame.reformat(width=width, height=height).to_ndarray(format='rgb24')

    def push(self, rgb: np.ndarray) -> None:
        scaled = self._scaled_to_width(rgb, self._width)
        if self._encoder is None:
            self._encoder = self._open(*scaled.shape[:2])
        container, stream = self._encoder
        for packet in stream.encode(av.VideoFrame.from_ndarray(scaled, format='rgb24')):
            container.mux(packet)
        self._dispatch(self._buffer.drain())

    def _dispatch(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            if not self._init:
                marker = data.find(b'moof')
                if marker < 4:
                    self._header += data
                    return
                self._init = self._header + data[: marker - 4]
                data = data[marker - 4 :]
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            if subscriber.full():
                try:
                    subscriber.get_nowait()
                except queue.Empty:  # the subscriber drained it first
                    pass
            subscriber.put(data)

    def subscribe(self) -> queue.Queue[bytes]:
        subscriber: queue.Queue[bytes] = queue.Queue(maxsize=self._fps)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[bytes]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    @property
    def init_segment(self) -> bytes:
        """Empty until the muxer writes the first fragment."""
        with self._lock:
            return self._init

    def close(self) -> None:
        if self._encoder is None:
            return
        container, stream = self._encoder
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()
        self._dispatch(self._buffer.drain())
