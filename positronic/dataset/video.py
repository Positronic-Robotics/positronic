from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .core import Signal, SignalWriter


class VideoSignalWriter(SignalWriter[np.ndarray]):
    """Writer for video signals.

    Stores video frames in a video file (e.g., MP4/MKV) with a Parquet index
    containing frame timestamps for fast random access.
    """

    def __init__(self,
                 video_path: Path,
                 frames_index_path: Path,
                 codec: str = 'h264',
                 gop_size: int = 30,
                 fps: int = 100):
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
        self._frame_count = 0
        self._last_ts = None

        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._frame_timestamps: List[int] = []

    def _init_video_encoder(self, first_frame: np.ndarray) -> None:
        """Initialize video encoder based on first frame dimensions."""
        if first_frame.ndim != 3 or first_frame.shape[2] != 3:
            raise ValueError(f"Expected frame shape (H, W, 3), got {first_frame.shape}")

        if first_frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 dtype, got {first_frame.dtype}")

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
            raise RuntimeError("Cannot append to a finished writer")

        if self._last_ts is not None and ts_ns <= self._last_ts:
            raise ValueError(f"Timestamp {ts_ns} is not increasing (last was {self._last_ts})")

        if self._container is None:
            self._init_video_encoder(data)
        else:
            if data.shape[:2] != (self._height, self._width):
                raise ValueError(f"Frame shape {data.shape[:2]} doesn't match expected ({self._height}, {self._width})")
            if data.dtype != np.uint8:
                raise ValueError(f"Expected uint8 dtype, got {data.dtype}")

        self._frame_timestamps.append(ts_ns)

        frame = av.VideoFrame.from_ndarray(data, format='rgb24')
        frame.pts = self._frame_count

        for packet in self._stream.encode(frame):  # Every frame may produce 0, 1, or more packets
            self._container.mux(packet)

        self._frame_count += 1
        self._last_ts = ts_ns

    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalize the writing on context exit (even on exceptions)."""
        if self._finished:
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


class _VideoSliceView(Signal[np.ndarray]):
    """A view of a VideoSignal using an array of indices.

    Optionally keeps its own timestamps (e.g., for time-sampled views) which may
    differ from the parent's frame timestamps and can include repeated values.
    """

    def __init__(self, parent: "VideoSignal", indices: np.ndarray, sample_timestamps: Optional[np.ndarray] = None):
        """Initialize a slice view.

        Args:
            parent: The parent VideoSignal
            indices: Array of indices into the parent signal
            sample_timestamps: Optional timestamps for this view (used for time-sampled views)
        """
        self.parent = parent
        self.indices = np.asarray(indices, dtype=np.int64)
        self._sample_timestamps = None if sample_timestamps is None else np.asarray(sample_timestamps, dtype=np.int64)

    @classmethod
    def empty(cls):
        return cls(None, np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    def _load_timestamps(self):
        """Ensure parent's timestamps are loaded."""
        self.parent._load_timestamps()

    def __len__(self) -> int:
        """Returns the number of frames in this view."""
        return len(self.indices)

    @property
    def start_ts(self) -> int:
        if len(self) == 0:
            raise ValueError("Signal is empty")
        return int(self._timestamps[0])

    @property
    def last_ts(self) -> int:
        if len(self) == 0:
            raise ValueError("Signal is empty")
        return int(self._timestamps[-1])

    @property
    def time(self):
        """Returns an indexer for accessing Signal data by timestamp."""
        return _VideoTimeIndexer(self)

    @property
    def _timestamps(self):
        """Access the timestamps for this view."""
        if self._sample_timestamps is not None:
            return self._sample_timestamps
        self._load_timestamps()
        return self.parent._timestamps[self.indices]

    def _get_frame_at_index(self, index: int) -> Tuple[np.ndarray, int]:
        """Get frame at the given index in this view."""
        frame, _orig_ts = self.parent._get_frame_at_index(self.indices[index])
        return frame, int(self._timestamps[index])

    def _p_idx_for_pos(self, pos: np.ndarray) -> Tuple["VideoSignal", np.ndarray]:
        """Map local positions to parent indices for array-based time lookups."""
        return self.parent, self.indices[pos]

    def _p_idx_for_range(self, start_idx: int, stop_idx: int) -> Tuple["VideoSignal", np.ndarray]:
        """Map local range [start_idx:stop_idx] to parent indices for slice-based time lookups."""
        return self.parent, self.indices[start_idx:stop_idx]

    def __getitem__(self, index: Union[int, slice, np.ndarray,
                                       list]) -> Union[Tuple[np.ndarray, int], "Signal[np.ndarray]"]:
        """Access a frame by index within this slice."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.indices))
            if step <= 0:
                raise IndexError(f"Slice step must be positive, got {step}")
            new_indices = self.indices[start:stop:step]
            new_ts = None if self._sample_timestamps is None else self._timestamps[start:stop:step]
            return _VideoSliceView(self.parent, new_indices, new_ts)

        if isinstance(index, (list, np.ndarray)):
            index = np.asarray(index)
            if index.dtype == bool:
                raise NotImplementedError("Boolean indexing is not supported")
            index = np.where(index < 0, index + len(self), index)
            if np.any((index < 0) | (index >= len(self))):
                raise IndexError("Index out of range")
            new_indices = self.indices[index]
            new_ts = None if self._sample_timestamps is None else self._timestamps[index]
            return _VideoSliceView(self.parent, new_indices, new_ts)

        if not isinstance(index, (int, np.integer)):
            raise NotImplementedError(f"Unsupported index type: {type(index)}")
        index = int(index)

        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range")
        return self._get_frame_at_index(index)


class _VideoTimeIndexer:
    """Helper class to implement the time property for VideoSignal."""

    def __init__(self, signal: Union["VideoSignal", "_VideoSliceView"]):
        self.signal = signal

    @property
    def _timestamps(self) -> np.ndarray:
        self.signal._load_timestamps()
        return self.signal._timestamps

    def __getitem__(self, key: Union[int, slice, List[int],
                                     np.ndarray]) -> Union[Tuple[np.ndarray, int], Signal[np.ndarray]]:
        """Access frame by timestamp or time range.

        Args:
            key: Timestamp in nanoseconds or slice for time range

        Returns:
            If int: Tuple of (frame, timestamp_ns) for the frame at or before the given timestamp
            If slice: VideoSignal view containing frames in the time range

        Raises:
            KeyError: If no frame exists at or before the given timestamp
            NotImplementedError: If key is not an integer/array/slice
        """
        if isinstance(key, slice):
            return self._get_by_slice(key)
        if isinstance(key, (list, np.ndarray)):
            return self._get_by_array(key)
        if isinstance(key, (int, np.integer)):
            return self._get_by_scalar(int(key))
        raise NotImplementedError(f"Unsupported key type for time indexing: {type(key)}")

    def _get_by_scalar(self, ts: int) -> Tuple[np.ndarray, int]:
        idx = int(np.searchsorted(self._timestamps, ts, side='right') - 1)
        if idx < 0:
            raise KeyError(f"No record at or before timestamp {ts}")
        return self.signal[idx]

    def _get_by_array(self, key: Union[List[int], np.ndarray]) -> Signal[np.ndarray]:
        req_ts = np.asarray(key)
        if req_ts.size == 0:
            raise TypeError("Empty timestamp arrays are not supported")
        if not np.issubdtype(req_ts.dtype, np.integer):
            raise TypeError(f"Invalid timestamp array dtype: {req_ts.dtype}")

        pos = np.searchsorted(self._timestamps, req_ts, side='right') - 1
        if not np.all(pos >= 0):
            raise KeyError("No record at or before some of the requested timestamps")

        parent, parent_indices = self.signal._p_idx_for_pos(pos)
        return _VideoSliceView(parent, parent_indices, sample_timestamps=req_ts.astype(np.int64, copy=False))

    def _get_by_slice(self, sl: slice) -> Signal[np.ndarray]:
        # Stepped time slice -> sample at requested timestamps
        if sl.step is not None:
            if sl.step <= 0:
                raise KeyError("Slice step must be positive")
            if sl.start is None:
                raise KeyError("Slice start is required when step is provided")
            stop = sl.stop if sl.stop is not None else int(self._timestamps[-1]) + 1
            return self._get_by_array(np.arange(sl.start, stop, sl.step, dtype=np.int64))

        # Non-stepped time slice -> contiguous view by frame indices
        if len(self._timestamps) == 0:
            return _VideoSliceView.empty()

        start = sl.start if sl.start is not None else int(self._timestamps[0])
        stop = sl.stop if sl.stop is not None else int(self._timestamps[-1]) + 1

        start_idx = int(np.searchsorted(self._timestamps, start, side='left'))
        stop_idx = int(np.searchsorted(self._timestamps, stop, side='left'))
        parent, indices = self.signal._p_idx_for_range(start_idx, stop_idx)
        return _VideoSliceView(parent, indices)


class _VideoNavigator:
    """Efficiently navigates video frames using buffering and smart seeking."""

    def __init__(self, video_path: Path, seek_threshold: int):
        self._container = av.open(str(video_path))
        self._stream = self._container.streams.video[0]

        rate = self._stream.average_rate or self._stream.rate
        ticks_per_frame = round(1.0 / (float(rate) * float(self._stream.time_base)))
        self._ticks_per_frame = max(1, int(ticks_per_frame))
        self._seek_threshold = seek_threshold

        self._frame_buffer: deque[Tuple[int, av.VideoFrame]] = deque()

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

    def __iter__(self) -> Iterator[Tuple[int, av.VideoFrame]]:
        return self

    def __next__(self) -> Tuple[int, av.VideoFrame]:
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

    def __init__(self, video_path: Path, frames_index_path: Path, seek_threshold: Optional[int] = None):
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

    @property
    def start_ts(self) -> int:
        """Returns the timestamp of the first frame in the signal."""
        self._load_timestamps()
        if len(self._timestamps) == 0:
            raise ValueError("Signal is empty")
        return int(self._timestamps[0])

    @property
    def last_ts(self) -> int:
        """Returns the timestamp of the last frame in the signal."""
        self._load_timestamps()
        if len(self._timestamps) == 0:
            raise ValueError("Signal is empty")
        return int(self._timestamps[-1])

    @property
    def time(self):
        """Returns an indexer for accessing Signal data by timestamp."""
        return _VideoTimeIndexer(self)

    @lru_cache(maxsize=1)  # Access to the same index might be frequent
    def _get_frame_at_index(self, index: int) -> Tuple[np.ndarray, int]:
        """Internal method to get a frame at a specific index."""
        index = int(index)
        self._nav.seek_if_needed(index)
        for frame_index, frame in self._nav:
            if frame_index == index:
                return frame.to_ndarray(format='rgb24'), self._timestamps[index]
            elif frame_index > index:
                break

        raise IndexError(f"Could not decode frame {index}")

    def __getitem__(self, index: Union[int, slice, np.ndarray,
                                       list]) -> Union[Tuple[np.ndarray, int], Signal[np.ndarray]]:
        """Access a frame by index, slice, or array of indices.

        Args:
            index: Integer index, slice, or array-like of indices

        Returns:
            If index: Tuple of (frame, timestamp_ns)
            If slice/array: VideoSignal view of the selected data
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step <= 0:
                raise IndexError(f"Slice step must be positive, got {step}")
            indices = np.arange(start, stop, step, dtype=np.int64)
            return _VideoSliceView(self, indices)

        if isinstance(index, (list, np.ndarray)):
            index = np.asarray(index)
            if index.dtype == bool:
                raise NotImplementedError("Boolean indexing is not supported")
            index = np.where(index < 0, index + len(self), index)
            if np.any((index < 0) | (index >= len(self))):
                raise IndexError("Index out of range")
            return _VideoSliceView(self, index)

        if not isinstance(index, (int, np.integer)):
            raise NotImplementedError(f"Unsupported index type: {type(index)}")
        index = int(index)
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range")

        return self._get_frame_at_index(index)

    def _p_idx_for_pos(self, pos: np.ndarray) -> Tuple["VideoSignal", np.ndarray]:
        """Map absolute positions to parent indices for array-based time lookups."""
        return self, pos.astype(np.int64, copy=False)

    def _p_idx_for_range(self, start_idx: int, stop_idx: int) -> Tuple["VideoSignal", np.ndarray]:
        """Map absolute range [start_idx:stop_idx] to parent indices for slice-based time lookups."""
        return self, np.arange(start_idx, stop_idx, dtype=np.int64)
