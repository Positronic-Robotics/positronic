"""Frame stream: send the frames of the next request while the robot executes the current chunk.

A protocol v1 or v2 server that declares ``frame_stream`` in its handshake takes the history rows of the
next request as separate messages on the session's connection (on v2 without the session envelope). It
decodes each row when the row arrives, and it can start its model work on the row while the robot moves.
The request then refers to the streamed rows and carries the other frames. The messages::

    handshake meta:  "frame_stream": {"version": 1, "codecs": ["jpeg", "h264"], ...}
    row:             {"__stream_row__": {"call": c, "back": b, "t_ns": int, "images": {key: <image>}}}
    request image:   {b"__stream_rows__": True, b"call": c, b"rows": [b, ...], b"tail": <image(s)>}
    H.264 image:     {b"__h264_frame__": True, b"data": bytes, b"shape": [h, w, 3]}
    answer:          {"frame_miss": [[key, back], ...]}

``call`` counts the requests of the session from 0. ``back`` counts the frames from a row to the
current frame of its request: 1 is the frame just before it. ``rows`` lists the referred rows, oldest
first, and ``tail`` carries the frames after them. An ``<image>`` is a JPEG frame (``encode_jpeg``) or
one access unit of a per-camera H.264 stream. A server that does not have a row that a request refers
to answers ``frame_miss``, and the client sends that request again in full. The H.264 units of one
camera are one stream for the whole session, in send order. Thus the server decodes each unit, also
the tail of a request that it answers with ``frame_miss``.

``PolicyServer`` serves its side with ``StreamedRows``. A server may implement its side without importing
positronic, so this description is the protocol: a change to it changes every server.

The streamer finds the rows of the next request before the request exists. ``TemporalStack`` samples,
at the request time ``T``, the latest control tick at or before ``T + offset`` for each offset.
``ChunkedSchedule`` starts a chunk when it reads the answer, or when the previous chunk ends if that is
later, and sends the next request at the first tick at or after the chunk's end less its inference lead:
the start plus ``len(chunk) / fps``, less the lead estimate, which follows the delays from each submit to
the read of its answer. The streamer predicts ``T`` from the same times and keeps its own copy of the
ticks. It plans the rows when the schedule reads the answer. A timer sends each row when its target time
has passed, so the last row reaches the server before the request; a row whose target has passed at the
plan goes at once. An answer whose next request is due at once, such as an empty chunk, plans no rows.
The request compares each row of its stack with the streamed row, and it refers only to equal rows. A
wrong prediction costs bytes, but it never changes an observation.
"""

import collections.abc as cabc
import logging
import queue
import threading
import time
from collections import deque
from enum import Enum
from fractions import Fraction
from typing import Any

import av
import numpy as np
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, encode_jpeg

logger = logging.getLogger(__name__)

ROW = '__stream_row__'
REF = b'__stream_rows__'
H264_FRAME = b'__h264_frame__'
# The handshake key, and what a server that serves ``StreamedRows`` declares under it.
FRAME_STREAM = 'frame_stream'
DECLARATION = {'version': 1, 'codecs': ['jpeg', 'h264']}


class FrameCodec(Enum):
    """The encoding of a streamed row."""

    JPEG = 'jpeg'
    H264 = 'h264'


def declared_codecs(meta: cabc.Mapping[str, Any]) -> frozenset[FrameCodec]:
    """The codecs of the version 1 frame stream that a handshake's ``meta`` declares. Empty when it declares none."""
    declaration = meta.get(FRAME_STREAM)
    if not isinstance(declaration, cabc.Mapping) or declaration.get('version') != 1:
        return frozenset()
    return frozenset(codec for codec in FrameCodec if codec.value in declaration.get('codecs', ()))


def streams_stack(keys: cabc.Sequence[str], offsets_sec: cabc.Sequence[float]) -> bool:
    """Whether a frame stream carries the rows of a ``TemporalStack`` of ``keys`` at ``offsets_sec``: its last offset
    is the current frame, and its rows and keys fit the bounds of ``StreamedRows``."""
    return (
        bool(offsets_sec)
        and offsets_sec[-1] == 0.0
        and len(offsets_sec) - 1 <= StreamedRows.MAX_BACK
        and len(keys) <= StreamedRows.MAX_IMAGES
    )


class _VideoEncoder:
    """One H.264 stream for one camera: one access unit for each frame, with no frame delay."""

    def __init__(self, width: int, height: int, crf: int, fps: float) -> None:
        self._ctx = av.CodecContext.create('libx264', 'w')
        self._ctx.width, self._ctx.height = width + width % 2, height + height % 2
        self._ctx.pix_fmt = 'yuv420p'
        self._ctx.time_base = Fraction(1, round(fps))
        self._ctx.framerate = Fraction(round(fps), 1)
        self._ctx.options = {
            'preset': 'veryfast',
            'tune': 'zerolatency',
            'crf': str(crf),
            'x264-params': 'keyint=100000:bframes=0:repeat-headers=1',
        }
        self._shape = (height, width)
        self._frames = 0

    def encode(self, rgb: np.ndarray) -> dict[bytes, Any]:
        h, w = rgb.shape[:2]
        if (h, w) != self._shape:
            raise ValueError(f'A camera stream changed its size from {self._shape} to {(h, w)}')
        pad = np.pad(rgb, ((0, self._ctx.height - h), (0, self._ctx.width - w), (0, 0)), mode='edge')
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pad), format='rgb24')
        frame.pts = self._frames
        self._frames += 1
        data = b''.join(bytes(packet) for packet in self._ctx.encode(frame))
        if not data:
            raise RuntimeError('The H.264 encoder held a frame back')
        return {H264_FRAME: True, b'data': data, b'shape': [h, w, 3]}


class _VideoDecoder:
    """One camera's H.264 stream on the server: each access unit decodes to its frame, with no frame delay."""

    def __init__(self) -> None:
        self._ctx = av.CodecContext.create('h264', 'r')

    def decode(self, unit: cabc.Mapping[bytes, Any]) -> np.ndarray:
        h, w, *_ = (int(n) for n in unit[b'shape'])
        if not (1 <= h <= StreamedRows.MAX_SIDE and 1 <= w <= StreamedRows.MAX_SIDE):
            raise ValueError(f'An H.264 frame of {h} x {w} is outside {StreamedRows.MAX_SIDE} px')
        frames = self._ctx.decode(av.Packet(bytes(unit[b'data'])))
        if len(frames) != 1:
            raise ValueError(f'An H.264 unit decoded to {len(frames)} frames, not one')
        rgb = frames[0].to_ndarray(format='rgb24')
        # The encoder pads an odd side by one pixel; a crop is a copy, so it does not keep the padded frame.
        if rgb.shape[0] - h not in (0, 1) or rgb.shape[1] - w not in (0, 1):
            raise ValueError(f'An H.264 unit decoded to {rgb.shape[:2]}, not the declared {(h, w)}')
        return rgb if rgb.shape[:2] == (h, w) else rgb[:h, :w].copy()


class RowsMissing(Exception):
    """A request refers to streamed rows that the server does not have: ``[[key, back], ...]``."""

    def __init__(self, missing: list[list[Any]]) -> None:
        super().__init__(f'The server does not have the streamed rows {missing}')
        self.missing = missing


class StreamedRows:
    """The server side of one session's frame stream: the rows that the client streamed, and the requests that
    refer to them.

    ``add`` takes one row message and ``resolve`` gives a request with each stack that refers to rows in full, as
    the client would have sent it without a stream. Each H.264 unit decodes in arrival order, also the tail of a
    request that refers to a row the server does not have. The client streams the rows of a call only after it read
    the answer before, so a row of a newer call drops the rows of the older ones. Bounds on the rows, images and
    frame sides refuse a malformed stream, and past ``MAX_HELD_BYTES`` a row is dropped: its request then misses
    it and goes again in full.
    """

    MAX_BACK = 64
    MAX_IMAGES = 16
    MAX_SIDE = 4096
    MAX_HELD_BYTES = 512 * 1024 * 1024

    def __init__(self) -> None:
        self._call = -1
        self._rows: dict[int, dict[str, np.ndarray]] = {}
        self._held = 0
        self._video: dict[str, _VideoDecoder] = {}

    def add(self, row: cabc.Mapping[str, Any]) -> None:
        """Decode and keep one row (the value under ``ROW``)."""
        call, back, images = int(row['call']), int(row['back']), row['images']
        if not 1 <= back <= self.MAX_BACK:
            raise ValueError(f'A streamed row goes {back} frames back, outside 1..{self.MAX_BACK}')
        if len(images) > self.MAX_IMAGES:
            raise ValueError(f'A streamed row of {len(images)} images, more than {self.MAX_IMAGES}')
        decoded = {str(key): self._frame(str(key), value) for key, value in images.items()}
        if call < self._call:
            return  # a row of a call that the server answered
        if call > self._call:
            self._call, self._rows, self._held = call, {}, 0
        held = self._held + sum(frame.nbytes for frame in decoded.values())
        held -= sum(frame.nbytes for frame in self._rows.get(back, {}).values())
        if held > self.MAX_HELD_BYTES:
            self._rows.pop(back, None)
            return
        self._rows[back], self._held = decoded, held

    def resolve(self, obs: cabc.Mapping[str, Any]) -> dict[str, Any]:
        """``obs`` with each stack that refers to rows built in full. Raises ``RowsMissing`` for absent rows."""
        out = dict(obs)
        missing: list[list[Any]] = []
        for key, value in obs.items():
            if not (isinstance(value, cabc.Mapping) and value.get(REF)):
                continue
            backs = [int(back) for back in value[b'rows']]
            if len(backs) > self.MAX_BACK or len(set(backs)) != len(backs):
                raise ValueError(f'{key}: a stack refers to each of at most {self.MAX_BACK} rows once, not {backs}')
            tail = self._tail(str(key), value[b'tail'])
            rows = self._rows if int(value[b'call']) == self._call else {}
            found = [rows.get(back, {}).get(str(key)) for back in backs]
            frames = [frame for frame in found if frame is not None]
            missing += [[key, back] for back, frame in zip(backs, found, strict=True) if frame is None]
            if not missing:
                out[key] = np.stack([*frames, *tail])
        if missing:
            raise RowsMissing(missing)
        return out

    def _frame(self, key: str, value: Any) -> np.ndarray:
        if isinstance(value, cabc.Mapping) and value.get(H264_FRAME):
            decoder = self._video.get(key)
            if decoder is None:
                decoder = self._video[key] = _VideoDecoder()
            return decoder.decode(value)
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
            return value  # a JPEG frame, decoded with the message
        raise ValueError(f'{key}: a streamed frame must be a JPEG frame or an H.264 unit')

    def _tail(self, key: str, tail: Any) -> list[np.ndarray]:
        if isinstance(tail, np.ndarray) and tail.ndim == 4:
            return list(tail)
        return [self._frame(key, tail)]


class FrameStreamer:
    """Streams the history rows of each next request of one session.

    ``keys``, ``offsets_sec`` and ``pad_start`` are those of the declared ``TemporalStack``. ``resize``
    sizes the frames as the declared stack does. ``fps`` and ``horizon_sec`` are those of the schedule, and
    ``lead`` gives its inference lead from the delay of an answer (``DurationEstimateRun.update``; None: no lead).
    ``send`` puts one message on the session's connection, under the lock of the connection. ``clock`` is
    the episode clock in nanoseconds.
    """

    def __init__(
        self,
        send: cabc.Callable[[dict[str, Any]], None],
        clock: cabc.Callable[[], int],
        *,
        codec: FrameCodec,
        keys: tuple[str, ...],
        offsets_sec: tuple[float, ...],
        pad_start: bool,
        fps: float,
        horizon_sec: float | None = None,
        lead: cabc.Callable[[float], float] | None = None,
        resize: cabc.Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        crf: int = 18,
    ) -> None:
        if not offsets_sec or offsets_sec[-1] != 0.0:
            raise ValueError('A frame stream needs the current frame as the last stack offset')
        self._send = send
        self._clock = clock
        self._codec = codec
        self._keys = keys
        self._offsets = offsets_sec
        self._pad_start = pad_start
        self._fps = fps
        self._horizon_sec = horizon_sec
        self._lead = lead
        self._submitted_at: float | None = None
        self._chunk_end = 0.0
        self._resize = resize
        self._quality = jpeg_quality
        self._crf = crf
        self._ticks: deque[tuple[float, dict[str, np.ndarray]]] = deque()
        self._image_keys: tuple[str, ...] | None = None
        self.call = 0
        self._targets: list[tuple[int, float]] = []  # (back, target time), oldest first
        self._streamed: dict[int, dict[str, np.ndarray]] = {}
        self._video: dict[str, _VideoEncoder] = {}
        self._queue: queue.Queue[tuple[int, int, dict[str, np.ndarray]] | None] = queue.Queue()
        self._failed: BaseException | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._rows_sent = self._rows_referred = self._frames_inline = 0
        threading.Thread(target=self._send_rows, name='FrameStreamer', daemon=True).start()
        threading.Thread(target=self._time_rows, name='FrameStreamerTimer', daemon=True).start()

    def on_tick(self, obs: cabc.Mapping[str, Any]) -> None:
        """Record the frames of one raw observation of the run."""
        now = self._clock() / 1e9
        if self._image_keys is None:
            self._image_keys = tuple(
                key
                for key in self._keys
                if isinstance(obs.get(key), np.ndarray) and obs[key].ndim == 3 and obs[key].shape[-1] == 3
            )
        with self._lock:
            self._ticks.append((now, {key: obs[key] for key in self._image_keys if key in obs}))
            keep = now + self._offsets[0] - 1.0  # one second before the oldest offset
            while len(self._ticks) > 2 and self._ticks[1][0] <= keep:
                self._ticks.popleft()
        self._dispatch_due(now)

    def on_submit(self) -> None:
        """Record the time the schedule submits a call."""
        self._submitted_at = self._clock() / 1e9

    def on_answer(self, chunk: cabc.Sized) -> None:
        """Plan the rows of the next request when the schedule reads ``chunk``."""
        self.flush()
        now = self._clock() / 1e9
        delay = 0.0 if self._submitted_at is None else now - self._submitted_at
        lead = 0.0 if self._lead is None else self._lead(delay)
        duration = len(chunk) / self._fps
        if self._horizon_sec is not None:
            duration = min(duration, self._horizon_sec)
        start = max(now, self._chunk_end)  # an early answer waits behind the commands still due
        self._chunk_end = start + duration
        t_next = self._chunk_end - min(lead, duration)
        n = len(self._offsets)
        with self._lock:
            self.call += 1
            self._streamed = {}
            planned = t_next > now
            self._targets = [(n - 1 - i, t_next + o) for i, o in enumerate(self._offsets[:-1])] if planned else []
        self._wake.set()
        if planned:
            self._dispatch_due(now)  # rows whose target has passed go at once

    def request(self, obs: cabc.Mapping[str, Any]) -> dict[str, Any]:
        """The observation of a request, with each image stack referring to the streamed rows it holds."""
        self._dispatch_due(self._clock() / 1e9)
        with self._lock:
            self._targets = []  # a row that is not due yet is not in this request's stack
        self.flush()
        out = dict(obs)
        for key in self._image_keys or ():
            stack = obs.get(key)
            if isinstance(stack, np.ndarray) and stack.ndim == 4:
                out[key] = self._refer(key, stack)
        return out

    def flush(self) -> None:
        """Wait until every queued row is on the wire."""
        self._queue.join()
        if self._failed is not None:
            raise RuntimeError('The frame streamer failed') from self._failed

    def close(self) -> None:
        """Stop the streamer. It drops the rows that are not on the wire yet."""
        self._closed = True
        self._wake.set()
        self._queue.put(None)
        logger.info(
            'Frame stream closed: %d rows sent, %d rows referred to, %d frames inline',
            self._rows_sent,
            self._rows_referred,
            self._frames_inline,
        )

    def _dispatch_due(self, now: float) -> None:
        """Queue every row whose target time has passed, with the frame that ``TemporalStack`` samples for it:
        the latest tick at or before the target."""
        with self._lock:
            while self._targets and now >= self._targets[0][1]:
                back, target = self._targets.pop(0)
                if not self._ticks or (not self._pad_start and target < self._ticks[0][0]):
                    continue  # TemporalStack drops an offset before its first entry
                times = [t for t, _ in self._ticks]
                index = max(int(np.searchsorted(times, target, side='right')) - 1, 0)
                self._queue.put((self.call, back, self._ticks[index][1]))

    def _time_rows(self) -> None:
        """Dispatch each row at its target time, between control ticks."""
        while not self._closed:
            with self._lock:
                due = self._targets[0][1] if self._targets else None
            if due is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            wait = due - self._clock() / 1e9 + 0.001  # a tick stamped at the target is in
            if wait > 0:
                self._wake.wait(min(wait, 0.5))
                self._wake.clear()
                continue
            self._dispatch_due(self._clock() / 1e9)

    def _refer(self, key: str, stack: np.ndarray) -> dict[bytes, Any]:
        """The request value of one stack: its oldest rows that equal the streamed rows, then the other frames."""
        n = len(stack)
        rows: list[int] = []
        for j in range(n - 1):
            back = n - 1 - j
            row = self._streamed.get(back, {}).get(key)
            if row is None or row.shape != stack[j].shape or not np.array_equal(row, stack[j]):
                break
            rows.append(back)
        if self._codec is FrameCodec.H264 and len(rows) == n - 1:
            tail = self._encode(key, stack[-1])
        elif self._codec is FrameCodec.H264:
            rows = []  # an H.264 stream cannot carry an inline frame out of order
            tail = encode_jpeg(stack, self._quality)
        else:
            tail = encode_jpeg(stack[len(rows) :], self._quality)
        self._rows_referred += len(rows)
        self._frames_inline += n - len(rows)
        return {REF: True, b'call': self.call, b'rows': rows, b'tail': tail}

    def _encode(self, key: str, image: np.ndarray) -> dict[bytes, Any]:
        if self._codec is FrameCodec.JPEG:
            return encode_jpeg(image, self._quality)
        encoder = self._video.get(key)
        if encoder is None:
            encoder = self._video[key] = _VideoEncoder(image.shape[1], image.shape[0], self._crf, self._fps)
        return encoder.encode(image)

    def _send_rows(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._failed is None and not self._closed:
                    self._send_row(*item)
            except BaseException as failure:  # the next flush raises it
                logger.exception('The frame streamer failed')
                self._failed = failure
            finally:
                self._queue.task_done()

    def _send_row(self, call: int, back: int, images: dict[str, np.ndarray]) -> None:
        sized = images if self._resize is None else self._resize(images)
        encoded = {key: self._encode(key, image) for key, image in sized.items()}
        self._send({ROW: {'call': call, 'back': back, 't_ns': time.time_ns(), 'images': encoded}})
        self._rows_sent += 1
        with self._lock:
            if call == self.call:
                self._streamed[back] = sized
