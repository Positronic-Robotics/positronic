from pathlib import Path
from typing import List, Optional, Tuple

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

        # Lazy-load timestamp index
        self._timestamps = None

        # Lazy-load video container
        self._container: Optional[av.container.InputContainer] = None
        self._stream: Optional[av.video.stream.VideoStream] = None

    def _load_timestamps(self):
        """Lazily load timestamps from the index file."""
        if self._timestamps is None:
            frames_table = pq.read_table(self.frames_index_path)
            # Convert timestamps to raw nanoseconds (int64)
            ts_column = frames_table['ts_ns']
            self._timestamps = np.array([t.value for t in ts_column], dtype=np.int64)

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
        """Returns an indexer for accessing Signal data by timestamp.

        Not implemented yet - will be added later.
        """
        raise NotImplementedError("Time indexing not yet implemented for VideoSignal")

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """Access a frame by index.

        Args:
            index: Integer index

        Returns:
            Tuple of (frame, timestamp_ns)
        """
        if not isinstance(index, int):
            raise NotImplementedError(f"Only integer indexing is supported, got {type(index)}")

        self._load_timestamps()

        if index < 0:  # Handle negative indexing
            index += len(self._timestamps)

        if not 0 <= index < len(self._timestamps):
            raise IndexError(f"Index {index} out of range")

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
