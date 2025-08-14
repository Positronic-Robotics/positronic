from pathlib import Path
from typing import List, Optional, Tuple, Union

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
                 fps: int = 30):
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

        # Video encoding components - initialized on first frame
        self._container: Optional[av.container.OutputContainer] = None
        self._stream: Optional[av.video.stream.VideoStream] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None

        # Frame timestamps buffer
        self._frame_timestamps: List[int] = []

    def _init_video_encoder(self, first_frame: np.ndarray) -> None:
        """Initialize video encoder based on first frame dimensions."""
        if first_frame.ndim != 3 or first_frame.shape[2] != 3:
            raise ValueError(f"Expected frame shape (H, W, 3), got {first_frame.shape}")

        if first_frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 dtype, got {first_frame.dtype}")

        self._height, self._width = first_frame.shape[:2]

        # Open container for writing
        self._container = av.open(str(self.video_path), mode='w')

        # Create video stream
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

    def finish(self) -> None:
        """Finalize the writing. All following append calls will fail."""
        if self._finished:
            return

        self._finished = True

        if self._container is not None:
            for packet in self._stream.encode():  # Flush remaining frames
                self._container.mux(packet)
            self._container.close()

        # Write frame index (just timestamps, frame numbers are implicit indices)
        if self._frame_timestamps:
            frames_table = pa.table({'ts_ns': pa.array(self._frame_timestamps, type=pa.timestamp('ns'))})
        else:
            schema = pa.schema([('ts_ns', pa.timestamp('ns'))])
            frames_table = pa.table({'ts_ns': []}, schema=schema)

        pq.write_table(frames_table, self.frames_index_path)


class _VideoSliceView(Signal[np.ndarray]):
    """A slice view of a VideoSignal."""

    def __init__(self, parent: "VideoSignal", start: int, stop: int):
        """Initialize a slice view.

        Args:
            parent: The parent VideoSignal
            start: Start index (inclusive)
            stop: Stop index (exclusive)
        """
        self.parent = parent
        self.start = start
        self.stop = stop

    def _load_timestamps(self):
        """Ensure parent's timestamps are loaded."""
        self.parent._load_timestamps()

    def __len__(self) -> int:
        """Returns the number of frames in this view."""
        self._load_timestamps()
        return self.stop - self.start

    @property
    def time(self):
        """Returns an indexer for accessing Signal data by timestamp."""
        return _VideoTimeIndexer(self)

    @property
    def _timestamps(self):
        """Access the slice of parent's timestamps."""
        self._load_timestamps()
        return self.parent._timestamps[self.start:self.stop]

    def _get_frame_at_index(self, index: int) -> Tuple[np.ndarray, int]:
        return self.parent._get_frame_at_index(self.start + index)

    def __getitem__(self, index: Union[int, slice]) -> Union[Tuple[np.ndarray, int], "Signal[np.ndarray]"]:
        """Access a frame by index within this slice."""
        if isinstance(index, slice):
            start, stop, step = index.indices(self.stop - self.start)
            if step != 1:
                raise NotImplementedError(f"Only step=1 is supported for slices, got step={step}")
            return _VideoSliceView(self.parent, self.start + start, self.start + stop)

        if not isinstance(index, (int, np.integer)):
            raise NotImplementedError(f"Only integer indexing is supported, got {type(index)}")
        index = int(index)

        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range")
        return self.parent._get_frame_at_index(self.start + index)


class _VideoTimeIndexer:
    """Helper class to implement the time property for VideoSignal."""

    def __init__(self, signal: Union["VideoSignal", "_VideoSliceView"]):
        self.signal = signal

    @property
    def _timestamps(self) -> np.ndarray:
        self.signal._load_timestamps()
        return self.signal._timestamps

    def __getitem__(self, key: Union[int, slice]) -> Union[Tuple[np.ndarray, int], Signal[np.ndarray]]:
        """Access frame by timestamp or time range.

        Args:
            key: Timestamp in nanoseconds or slice for time range

        Returns:
            If int: Tuple of (frame, timestamp_ns) for the frame at or before the given timestamp
            If slice: VideoSignal view containing frames in the time range

        Raises:
            KeyError: If no frame exists at or before the given timestamp
            NotImplementedError: If key is not an integer or slice, or if slice has step
        """
        if isinstance(key, slice):
            if key.step is not None:
                raise NotImplementedError(f"Time slicing with step is not supported, got step={key.step}")

            start_idx = 0 if key.start is None else np.searchsorted(self._timestamps, key.start)
            stop_idx = len(self.signal) if key.stop is None else np.searchsorted(self._timestamps, key.stop)

            if isinstance(self.signal, _VideoSliceView):
                new_start = self.signal.start + start_idx
                new_stop = self.signal.start + stop_idx
                return _VideoSliceView(self.signal.parent, new_start, new_stop)
            else:
                return _VideoSliceView(self.signal, start_idx, stop_idx)

        if not isinstance(key, int):
            raise NotImplementedError(f"Only single integer timestamp or slice is supported, got {type(key)}")

        idx = np.searchsorted(self._timestamps, key, side='right') - 1
        if idx < 0:
            raise KeyError(f"No record at or before timestamp {key}")
        return self.signal[idx]


class VideoSignal(Signal[np.ndarray]):
    """Reader for video signals.

    Reads video frames from a video file (e.g., MP4/MKV) with a Parquet index
    containing frame timestamps for fast random access.
    """

    def __init__(self, video_path: Path, frames_index_path: Path):
        """Initialize VideoSignal reader.

        Args:
            video_path: Path to the video file to read
            frames_index_path: Path to frames.parquet index file
        """
        self.video_path = video_path
        self.frames_index_path = frames_index_path

        self._timestamps = None  # Lazy-load timestamps

        # Lazy-load video container
        self._container: Optional[av.container.InputContainer] = None
        self._stream: Optional[av.video.stream.VideoStream] = None

    def _load_timestamps(self):
        """Lazily load timestamps from the index file."""
        if self._timestamps is None:
            frames_table = pq.read_table(self.frames_index_path)
            self._timestamps = frames_table['ts_ns'].to_numpy()

    def _open_video(self):
        """Open video container for reading."""
        if self._container is None:
            self._container = av.open(str(self.video_path))
            self._stream = self._container.streams.video[0]

    def __len__(self) -> int:
        """Returns the number of frames in the signal."""
        self._load_timestamps()
        return len(self._timestamps)

    @property
    def time(self):
        """Returns an indexer for accessing Signal data by timestamp."""
        return _VideoTimeIndexer(self)

    def _get_frame_at_index(self, index: int) -> Tuple[np.ndarray, int]:
        """Internal method to get a frame at a specific index.

        Args:
            index: Absolute index in the video (must be valid)

        Returns:
            Tuple of (frame, timestamp_ns)
        """
        self._load_timestamps()
        self._open_video()
        self._container.seek(0, stream=self._stream)

        frame_count = 0
        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                if frame_count == index:
                    arr = frame.to_ndarray(format='rgb24')
                    return (arr, self._timestamps[index])
                frame_count += 1

        raise IndexError(f"Could not decode frame {index}")

    def __getitem__(self, index: Union[int, slice]) -> Union[Tuple[np.ndarray, int], Signal[np.ndarray]]:
        """Access a frame by index or slice.

        Args:
            index: Integer index or slice

        Returns:
            If index: Tuple of (frame, timestamp_ns)
            If slice: VideoSignal view of the sliced data
        """
        if isinstance(index, slice):
            self._load_timestamps()
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise NotImplementedError(f"Only step=1 is supported for slices, got step={step}")
            return _VideoSliceView(self, start, stop)

        if not isinstance(index, (int, np.integer)):
            raise NotImplementedError(f"Only integer indexing is supported, got {type(index)}")
        index = int(index)
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range")

        return self._get_frame_at_index(index)
