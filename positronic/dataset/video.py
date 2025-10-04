from collections import deque
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .signal import IndicesLike, Kind, RealNumericArrayLike, Signal, SignalMeta, SignalWriter, is_realnum_dtype


class VideoSignalWriter(SignalWriter[np.ndarray]):
    """Writer for video signals.

    Stores video frames in a video file (e.g., MP4/MKV) with a Parquet index
    containing frame timestamps for fast random access.
    """

    def __init__(
        self, video_path: Path, frames_index_path: Path, codec: str = 'h264', gop_size: int = 30, fps: int = 100
    ):
        """Initialize VideoSignalWriter.

        Args:
            video_path: Path to the video file to write
            frames_index_path: Path to frames.parquet index file
            codec: Video codec to use (default: 'h264')
            gop_size: Group of Pictures size - distance between keyframes (default: 30)
            fps: Frame rate for encoding (default: 30)
        """
        self.video_path = video_path
        self.frames_index_path = frames_index_path
        self.codec = codec
        self.gop_size = gop_size
        self.fps = fps

        self._finished = False
        self._aborted = False
        self._frame_count = 0
        self._last_ts = None

        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._frame_timestamps: list[int] = []

    def _init_video_encoder(self, first_frame: np.ndarray) -> None:
        """Initialize video encoder based on first frame dimensions."""
        if first_frame.ndim != 3 or first_frame.shape[2] != 3:
            raise ValueError(f'Expected frame shape (H, W, 3), got {first_frame.shape}')

        if first_frame.dtype != np.uint8:
            raise ValueError(f'Expected uint8 dtype, got {first_frame.dtype}')

        self._height, self._width = first_frame.shape[:2]

        self._container = av.open(str(self.video_path), mode='w')
        self._stream = self._container.add_stream(self.codec, rate=self.fps)
        self._stream.width = self._width
        self._stream.height = self._height
        self._stream.pix_fmt = 'yuv420p'
        self._stream.gop_size = self.gop_size

    def append(self, data: np.ndarray, ts_ns: int) -> None:
        """Append a video frame with timestamp.

        Args:
            data: Image frame as uint8 numpy array with shape (H, W, 3)
            ts_ns: Timestamp in nanoseconds (must be strictly increasing)

        Raises:
            RuntimeError: If writer has been finished
            ValueError: If timestamp is not increasing or data shape/dtype doesn't match
        """
        if self._finished:
            raise RuntimeError('Cannot append to a finished writer')
        if self._aborted:
            raise RuntimeError('Cannot append to an aborted writer')

        if self._last_ts is not None and ts_ns <= self._last_ts:
            raise ValueError(f'Timestamp {ts_ns} is not increasing (last was {self._last_ts})')

        if self._container is None:
            self._init_video_encoder(data)
        else:
            if data.shape[:2] != (self._height, self._width):
                raise ValueError(f"Frame shape {data.shape[:2]} doesn't match expected ({self._height}, {self._width})")
            if data.dtype != np.uint8:
                raise ValueError(f'Expected uint8 dtype, got {data.dtype}')

        self._frame_timestamps.append(ts_ns)

        frame = av.VideoFrame.from_ndarray(data, format='rgb24')
        frame.pts = self._frame_count

        for packet in self._stream.encode(frame):  # Every frame may produce 0, 1, or more packets
            self._container.mux(packet)

        self._frame_count += 1
        self._last_ts = ts_ns

    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalize the writing on context exit (even on exceptions)."""
        if self._finished or self._aborted:
            return
        self._finished = True

        if self._container is not None:
            for packet in self._stream.encode():  # Flush remaining frames
                self._container.mux(packet)
            self._container.close()

        # Write frame index (just timestamps, frame numbers are implicit indices)
        if self._frame_timestamps:
            frames_table = pa.table({'ts_ns': pa.array(self._frame_timestamps, type=pa.int64())})
        else:
            schema = pa.schema([('ts_ns', pa.int64())])
            frames_table = pa.table({'ts_ns': []}, schema=schema)

        pq.write_table(frames_table, self.frames_index_path)

    def abort(self) -> None:
        """Abort writing and remove any partial outputs."""
        if self._aborted:
            return
        if self._finished:
            raise RuntimeError('Cannot abort a finished writer')

        if self._container is not None:
            self._container.close()
        self._container = None

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

        self._timestamps = None
        self._navigator: _VideoNavigator | None = None

    def _load_timestamps(self):
        """Lazily load timestamps from the index file."""
        if self._timestamps is None:
            frames_table = pq.read_table(self.frames_index_path)
            self._timestamps = frames_table['ts_ns'].to_numpy()

    @property
    def _nav(self) -> _VideoNavigator:
        if self._navigator is None:
            self._navigator = _VideoNavigator(self.video_path, self._seek_threshold)
        return self._navigator

    def __len__(self) -> int:
        """Returns the number of frames in the signal."""
        self._load_timestamps()
        return len(self._timestamps)

    @lru_cache(maxsize=1)  # Access to the same index might be frequent
    def _get_frame_at_index(self, index: int) -> tuple[np.ndarray, int]:
        """Internal method to get a frame at a specific index."""
        index = int(index)
        self._nav.seek_if_needed(index)
        for frame_index, frame in self._nav:
            if frame_index == index:
                return frame.to_ndarray(format='rgb24'), self._timestamps[index]
            elif frame_index > index:
                break

        raise IndexError(f'Could not decode frame {index}')

    def _ts_at(self, index_or_indices: IndicesLike) -> Sequence[int] | np.ndarray:
        self._load_timestamps()
        return self._timestamps[index_or_indices]

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

        def __getitem__(self, pos: int | slice) -> np.ndarray | Sequence[np.ndarray]:
            if isinstance(pos, slice):
                return VideoSignal._LazyFrames(self._parent, self._indices[pos])
            idx = int(self._indices[int(pos)])
            frame, _ts = self._parent._get_frame_at_index(idx)
            return frame

    def _values_at(self, index_or_indices: IndicesLike) -> Sequence[np.ndarray]:
        self._load_timestamps()
        if isinstance(index_or_indices, slice):
            start, stop, step = index_or_indices.indices(len(self))
            idxs = np.arange(start, stop, step, dtype=np.int64)
        else:
            idxs = np.asarray(index_or_indices, dtype=np.int64)
        return VideoSignal._LazyFrames(self, idxs)

    def _search_ts(self, ts_or_array: RealNumericArrayLike) -> IndicesLike:
        self._load_timestamps()
        req = np.asarray(ts_or_array)
        if req.size == 0:
            return np.array([], dtype=np.int64)
        if not is_realnum_dtype(req.dtype):
            raise TypeError(f'Invalid timestamp array dtype: {req.dtype}')
        return np.searchsorted(self._timestamps, req, side='right') - 1

    @property
    @lru_cache(maxsize=1)
    def meta(self) -> SignalMeta:
        # Video frames are HWC (height, width, channel); classify as image
        if len(self) == 0:
            raise ValueError('Signal is empty')
        base = super().meta
        return SignalMeta(dtype=base.dtype, shape=base.shape, kind=Kind.IMAGE, names=['height', 'width', 'channel'])
