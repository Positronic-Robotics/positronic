import queue
import struct
import threading
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Protocol, overload

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from av.container import OutputContainer
from av.video.stream import VideoStream

from .signal import IndicesLike, Kind, Signal, SignalMeta, SignalWriter, Time
from .vector import ParquetTimeIndex, timestamp_table


class VideoEncoderSession(Protocol):
    """One video file that an encoder writes, frame by frame."""

    def write(self, frame: np.ndarray, index: int) -> None:
        """Encode one ``(H, W, 3)`` uint8 RGB frame. ``index`` counts from 0 and increases by 1 each call."""
        ...

    def finish(self) -> None:
        """Flush the encoder and close the file. Raise if the file is not complete."""
        ...

    def abort(self) -> None:
        """Stop at once and release the encoder. The file can stay incomplete."""
        ...


class VideoEncoder(Protocol):
    """A picklable choice of video encoder. Each ``open`` starts one file."""

    def ensure_available(self) -> None:
        """Raise if this host cannot run the encoder."""
        ...

    def open(self, path: Path, width: int, height: int, fps: int, gop: int) -> VideoEncoderSession:
        """Start a file. Frame ``index`` plays at ``index / fps`` seconds, with a keyframe every ``gop`` frames."""
        ...


class _LibavSession:
    def __init__(self, container: OutputContainer, stream: VideoStream):
        self._container = container
        self._stream = stream

    def write(self, frame: np.ndarray, index: int) -> None:
        video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        video_frame.pts = index
        for packet in self._stream.encode(video_frame):  # Every frame may produce 0, 1, or more packets
            self._container.mux(packet)

    def finish(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()

    def abort(self) -> None:
        self._container.close()


@dataclass(frozen=True)
class LibavEncoder:
    """Encodes in the calling process with a PyAV (libav) codec."""

    codec: str = 'h264'
    # Encoder options as (name, value) pairs, e.g. x264 ``preset``/``tune``; empty keeps the codec defaults.
    options: tuple[tuple[str, str], ...] = ()

    def ensure_available(self) -> None:
        if av.Codec(self.codec, 'w').type != 'video':
            raise ValueError(f"'{self.codec}' is not a video codec")

    def open(self, path: Path, width: int, height: int, fps: int, gop: int) -> VideoEncoderSession:
        container = av.open(str(path), mode='w')
        stream = container.add_stream(self.codec, rate=fps, options=dict(self.options))
        if not isinstance(stream, VideoStream):
            container.close()
            raise ValueError(f"'{self.codec}' is not a video codec")
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        stream.gop_size = gop
        return _LibavSession(container, stream)


# libx264 at its default preset
DEFAULT_VIDEO_ENCODER = LibavEncoder()


class VideoSignalWriter(SignalWriter[np.ndarray]):
    """Writer for video signals.

    Stores video frames in a video file (e.g., MP4/MKV) with a Parquet index
    containing frame timestamps for fast random access.
    """

    def __init__(
        self,
        video_path: Path,
        frames_index_path: Path,
        encoder: VideoEncoder = DEFAULT_VIDEO_ENCODER,
        gop_size: int = 30,
        fps: int = 100,
    ):
        """Initialize VideoSignalWriter.

        Args:
            video_path: Path to the video file to write
            frames_index_path: Path to frames.parquet index file
            encoder: The encoder that writes the video file
            gop_size: Group of Pictures size - distance between keyframes (default: 30)
            fps: Frame rate for encoding (default: 100)
        """
        super().__init__()
        self.video_path = video_path
        self.frames_index_path = frames_index_path
        self.encoder = encoder
        self.gop_size = gop_size
        self.fps = fps

        self._finished = False
        self._aborted = False
        self._frame_count = 0

        self._session: VideoEncoderSession | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._frame_timestamps: dict[str, list[int]] = {}

        # One encoder thread per writer; the bound makes ``append`` wait when the encoder falls behind.
        self._frames: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(maxsize=8)
        self._encoder_thread: threading.Thread | None = None
        self._encoder_error: Exception | None = None

    def _open_encoder(self, first_frame: np.ndarray) -> None:
        """Open the encoder session based on first frame dimensions."""
        if first_frame.ndim != 3 or first_frame.shape[2] != 3:
            raise ValueError(f'Expected frame shape (H, W, 3), got {first_frame.shape}')

        if first_frame.dtype != np.uint8:
            raise ValueError(f'Expected uint8 dtype, got {first_frame.dtype}')

        height, width = first_frame.shape[:2]
        self._height, self._width = height, width
        self._session = self.encoder.open(self.video_path, width, height, self.fps, self.gop_size)

    def append(self, data: np.ndarray, timestamps: Time) -> None:
        """Append a video frame with timestamp.

        Args:
            data: Image frame as uint8 numpy array with shape (H, W, 3)
            timestamps: Named coordinates, non-decreasing on every timeline and increasing on at least one.

        Raises:
            RuntimeError: If writer has been finished
            ValueError: If timestamp is not increasing or data shape/dtype doesn't match
        """
        if self._finished:
            raise RuntimeError('Cannot append to a finished writer')
        if self._aborted:
            raise RuntimeError('Cannot append to an aborted writer')

        self._validate_timestamps(timestamps)
        if self._session is None:
            self._open_encoder(data)
        else:
            if data.shape[:2] != (self._height, self._width):
                raise ValueError(f"Frame shape {data.shape[:2]} doesn't match expected ({self._height}, {self._width})")
            if data.dtype != np.uint8:
                raise ValueError(f'Expected uint8 dtype, got {data.dtype}')

        if self._encoder_error is not None:
            raise RuntimeError('Video encoding failed') from self._encoder_error

        if not self._frame_timestamps:
            self._frame_timestamps = {name: [] for name in timestamps}
        for name, coordinate in timestamps.items():
            self._frame_timestamps[name].append(coordinate)

        if self._encoder_thread is None:
            self._encoder_thread = threading.Thread(
                target=self._encode_loop, name=f'encode:{self.video_path.name}', daemon=True
            )
            self._encoder_thread.start()
        # Copy before enqueueing: callers routinely pass views into shared memory that the producer
        # overwrites with the next frame.
        self._frames.put((data.copy(), self._frame_count))

        self._frame_count += 1
        self._last_time = timestamps

    def _encode_loop(self) -> None:
        session = self._session
        assert session is not None  # the thread starts only after the encoder opens
        try:
            while True:
                item = self._frames.get()
                if item is None:
                    return
                data, index = item
                session.write(data, index)
        except Exception as e:
            self._encoder_error = e
            # Keep draining so a blocked ``append`` unblocks; the error surfaces on the caller thread.
            while self._frames.get() is not None:
                pass

    def _stop_encoder(self) -> None:
        if self._encoder_thread is not None:
            self._frames.put(None)
            self._encoder_thread.join()
            self._encoder_thread = None

    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalize the writing on context exit (even on exceptions)."""
        if self._finished or self._aborted:
            return
        self._finished = True

        self._stop_encoder()
        if self._session is not None:
            if self._encoder_error is not None:
                self._session.abort()
                raise RuntimeError('Video encoding failed') from self._encoder_error
            try:
                self._session.finish()
            except Exception as e:
                self._session.abort()
                raise RuntimeError('Video encoding failed') from e

        pq.write_table(timestamp_table(self._frame_timestamps), self.frames_index_path)

    def abort(self) -> None:
        """Abort writing and remove any partial outputs."""
        if self._aborted:
            return
        if self._finished:
            raise RuntimeError('Cannot abort a finished writer')

        self._stop_encoder()
        if self._session is not None:
            self._session.abort()
        self._session = None

        for p in [self.video_path, self.frames_index_path]:
            if p.exists():
                p.unlink()

        self._aborted = True


class _VideoNavigator:
    """Efficiently navigates video frames using buffering and smart seeking."""

    def __init__(self, video_path: Path, seek_threshold: int):
        self._container = av.open(str(video_path))
        self._stream = self._container.streams.video[0]

        rate = self._stream.average_rate or self._stream.rate
        ticks_per_frame = round(1.0 / (float(rate) * float(self._stream.time_base)))
        self._ticks_per_frame = max(1, int(ticks_per_frame))
        self._seek_threshold = seek_threshold

        self._frame_buffer: deque[tuple[int, av.VideoFrame]] = deque()

        self._demux_iter = iter(self._container.demux(self._stream))
        self._last_idx = -1

    @property
    def last_decoded_frame_index(self) -> int:
        """Returns the index of the last decoded frame."""
        return self._last_idx

    def seek_if_needed(self, target_frame_index: int):
        """Seeks to target frame if distance exceeds threshold."""
        target_pts = target_frame_index * self._ticks_per_frame

        if self._last_idx != -1 and 0 < target_frame_index - self._last_idx <= self._seek_threshold:
            return

        self._container.seek(target_pts, stream=self._stream)
        self._demux_iter = iter(self._container.demux(self._stream))
        self._frame_buffer.clear()
        self._last_idx = -1

    def __iter__(self) -> Iterator[tuple[int, av.VideoFrame]]:
        return self

    def __next__(self) -> tuple[int, av.VideoFrame]:
        """Returns next frame from buffer or decodes new packets."""
        while self._frame_buffer:
            return self._frame_buffer.popleft()

        while not self._frame_buffer:
            packet = next(self._demux_iter)
            for frame in packet.decode():
                assert frame.pts is not None
                self._last_idx = int(frame.pts // self._ticks_per_frame)
                self._frame_buffer.append((self._last_idx, frame))

        return self._frame_buffer.popleft()


class VideoSignal(Signal[np.ndarray]):
    """Reader for video signals.

    Reads video frames from a video file (e.g., MP4/MKV) with a Parquet index
    containing frame timestamps for fast random access.
    """

    def __init__(self, video_path: Path, frames_index_path: Path, seek_threshold: int | None = None):
        """Initialize VideoSignal reader.

        Args:
            video_path: Path to the video file to read
            frames_index_path: Path to frames.parquet index file
            seek_threshold: Max forward distance (in frames) to decode sequentially
                before performing an indexed seek (defaults to GOP size if available)
        """
        self.video_path = video_path
        self.frames_index_path = frames_index_path
        # TODO: Read GOP size from video by analysing distance between keyframes
        # TODO: Profile it to find the best default threshold
        self._seek_threshold = seek_threshold or 30

        self._time_index = ParquetTimeIndex(frames_index_path, 'ts_ns')
        self._navigator: _VideoNavigator | None = None

    @property
    def timelines(self) -> tuple[str, ...]:
        return self._time_index.timelines

    def bounds(self, timelines: tuple[str, ...]) -> tuple[Time, Time]:
        if not len(self):
            raise ValueError('Signal is empty')
        self._validate_selection(timelines)
        return self._time_index.bounds(timelines)

    @property
    def _nav(self) -> _VideoNavigator:
        if self._navigator is None:
            self._navigator = _VideoNavigator(self.video_path, self._seek_threshold)
        return self._navigator

    def __len__(self) -> int:
        """Returns the number of frames in the signal."""
        return len(self._time_index)

    @lru_cache(maxsize=1)  # Access to the same index might be frequent
    def _get_frame_at_index(self, index: int) -> np.ndarray:
        """Internal method to get a frame at a specific index."""
        index = int(index)
        self._nav.seek_if_needed(index)
        for frame_index, frame in self._nav:
            if frame_index == index:
                return frame.to_ndarray(format='rgb24')
            elif frame_index > index:
                break

        raise IndexError(f'Could not decode frame {index}')

    def _ts_at(self, indices: IndicesLike, timelines: tuple[str, ...]) -> Sequence[Time]:
        return self._time_index.read(indices, timelines)

    class _LazyFrames(Sequence[np.ndarray]):
        """Lazy, indexable sequence of decoded frames for selected indices.

        Decodes frames on demand using the parent `VideoSignal` navigator.
        Supports random access by position and slicing without materializing
        all frames up front.
        """

        def __init__(self, parent: 'VideoSignal', indices: IndicesLike):
            self._parent = parent
            # Store as numpy int64 array for efficient indexing/slicing
            self._indices = np.asarray(indices, dtype=np.int64)

        def __len__(self) -> int:
            return int(self._indices.shape[0])

        @overload
        def __getitem__(self, pos: int) -> np.ndarray: ...

        @overload
        def __getitem__(self, pos: slice) -> Sequence[np.ndarray]: ...

        def __getitem__(self, pos: int | slice) -> np.ndarray | Sequence[np.ndarray]:
            if isinstance(pos, slice):
                return VideoSignal._LazyFrames(self._parent, self._indices[pos])
            idx = int(self._indices[int(pos)])
            return self._parent._get_frame_at_index(idx)

    def _values_at(self, indices: IndicesLike) -> Sequence[np.ndarray]:
        if isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            idxs = np.arange(start, stop, step, dtype=np.int64)
        else:
            idxs = np.asarray(indices, dtype=np.int64)
        return VideoSignal._LazyFrames(self, idxs)

    def _search_ts(self, queries: Sequence[Time]) -> Sequence[int] | np.ndarray:
        return self._time_index.search(queries)

    @cached_property
    def meta(self) -> SignalMeta:
        # Video frames are HWC (height, width, channel); classify as image
        if len(self) == 0:
            raise ValueError('Signal is empty')
        base = super().meta
        return SignalMeta(dtype=base.dtype, shape=base.shape, kind=Kind.IMAGE)

    # SupportsEncodedRepresentation protocol implementation

    @property
    def encoding_format(self) -> str:
        """Format identifier for video encoded representation."""
        return 'positronic.video.v1'

    def iter_encoded_chunks(self) -> Iterator[bytes]:
        """Stream video + timestamps as a simple container format.

        Format v1:
          - 8 bytes: video file size (uint64 little-endian)
          - N bytes: video file content (raw H.264/MP4)
          - 8 bytes: Arrow IPC size (uint64 little-endian)
          - M bytes: Arrow IPC stream (timestamps table)

        The Arrow table preserves the timestamp columns and their schema metadata.
        Using Arrow IPC format (not parquet) to decouple from storage format.
        """

        # Stream video file
        video_size = self.video_path.stat().st_size
        yield struct.pack('<Q', video_size)
        with open(self.video_path, 'rb') as f:
            while chunk := f.read(64 * 1024):
                yield chunk

        # Read timestamps from parquet, serialize as Arrow IPC
        frames_table = pq.read_table(self.frames_index_path)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, frames_table.schema) as writer:
            writer.write_table(frames_table)
        arrow_bytes = sink.getvalue().to_pybytes()

        yield struct.pack('<Q', len(arrow_bytes))
        yield arrow_bytes
