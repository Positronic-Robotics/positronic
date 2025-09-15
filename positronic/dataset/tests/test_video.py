import numpy as np
import pyarrow.parquet as pq
import pytest

from positronic.dataset.video import VideoSignal, VideoSignalWriter
from positronic.dataset.signal import Kind


@pytest.fixture
def video_paths(tmp_path):
    """Create paths for video and index files."""
    return {'video': tmp_path / "test.mp4", 'frames': tmp_path / "frames.parquet"}


@pytest.fixture
def writer(video_paths):
    """Create a VideoSignalWriter instance."""
    return VideoSignalWriter(video_paths['video'], video_paths['frames'])


def create_frame(value=0, shape=(100, 100, 3)):
    """Create a test frame with given value and shape."""
    return np.full(shape, value, dtype=np.uint8)


def create_video_signal(video_paths, frames_with_timestamps):
    """Helper to create a video signal with given frames and timestamps."""
    with VideoSignalWriter(video_paths['video'], video_paths['frames']) as writer:
        for frame, ts in frames_with_timestamps:
            writer.append(frame, ts)
    return VideoSignal(video_paths['video'], video_paths['frames'])


def assert_frames_equal(frame1, frame2, tolerance=20):
    """Assert that two frames are approximately equal, accounting for video compression artifacts.

    Args:
        frame1: First frame to compare
        frame2: Second frame to compare
        tolerance: Maximum allowed difference in median pixel values (default: 20)
    """
    assert frame1.shape == frame2.shape, f"Shape mismatch: {frame1.shape} != {frame2.shape}"
    assert frame1.dtype == frame2.dtype, f"Dtype mismatch: {frame1.dtype} != {frame2.dtype}"

    # Compare median values to account for compression artifacts
    median1 = np.median(frame1)
    median2 = np.median(frame2)
    assert median1 == pytest.approx(median2, abs=tolerance), \
        f"Frame content mismatch: median {median1} != {median2} (tolerance={tolerance})"


class TestVideoSignalWriter:

    def test_empty_writer(self, writer, video_paths):
        """Test creating and closing an empty writer."""
        with writer:
            pass

        # Check that index file exists and has correct schema
        assert video_paths['frames'].exists()
        table = pq.read_table(video_paths['frames'])
        assert len(table) == 0
        assert 'ts_ns' in table.column_names

    def test_write_single_frame(self, writer, video_paths):
        """Test writing a single frame."""
        frame = create_frame(value=128)
        with writer as w:
            w.append(frame, 1000)

        # Check video file was created
        assert video_paths['video'].exists()
        assert video_paths['video'].stat().st_size > 0

        # Check index file has exactly one timestamp
        frames_table = pq.read_table(video_paths['frames'])
        assert len(frames_table) == 1
        # Timestamps are stored as int64
        assert frames_table['ts_ns'][0].as_py() == 1000

    def test_write_multiple_frames(self, writer, video_paths):
        """Test writing multiple frames with increasing timestamps."""
        with writer as w:
            # Write 10 frames
            timestamps = [1000 * (i + 1) for i in range(10)]
            for i, ts in enumerate(timestamps):
                w.append(create_frame(i * 25, (50, 50, 3)), ts)

        # Should have exactly 10 timestamps in the index
        frames_table = pq.read_table(video_paths['frames'])
        assert len(frames_table) == 10
        # Verify timestamps match what we wrote
        stored_ts = [t.as_py() for t in frames_table['ts_ns']]
        assert stored_ts == timestamps

    def test_invalid_frame_shape(self, video_paths):
        """Test that invalid frame shapes are rejected."""
        invalid_frames = [
            (np.zeros((100, 100), dtype=np.uint8), "Expected frame shape"),  # 2D
            (np.zeros((100, 100, 4), dtype=np.uint8), "Expected frame shape"),  # 4 channels
        ]

        for frame, match in invalid_frames:
            with VideoSignalWriter(video_paths['video'], video_paths['frames']) as writer:
                with pytest.raises(ValueError, match=match):
                    writer.append(frame, 1000)

    def test_invalid_dtype(self, writer):
        """Test that invalid dtypes are rejected."""
        frame = np.zeros((100, 100, 3), dtype=np.float32)
        with writer:
            with pytest.raises(ValueError, match="Expected uint8 dtype"):
                writer.append(frame, 1000)

    def test_non_increasing_timestamp(self, writer):
        """Test that non-increasing timestamps are rejected."""
        frame1 = create_frame(0)
        frame2 = create_frame(1)
        with writer as w:
            w.append(frame1, 2000)
            # Try same and earlier timestamps
            for ts in [2000, 1000]:
                with pytest.raises(ValueError, match="not increasing"):
                    w.append(frame2, ts)

    def test_inconsistent_dimensions(self, writer):
        """Test that frame dimensions must be consistent."""
        with writer as w:
            w.append(create_frame(0, (100, 100, 3)), 1000)
            # Different dimensions should fail
            with pytest.raises(ValueError, match="Frame shape"):
                w.append(create_frame(0, (50, 50, 3)), 2000)

    def test_append_after_context_exit(self, writer):
        """Test that appending after finish raises an error."""
        frame = create_frame()
        with writer as w:
            w.append(frame, 1000)
        with pytest.raises(RuntimeError, match="Cannot append to a finished writer"):
            w.append(frame, 2000)


class TestVideoSignalStartLastTs:

    def test_video_start_last_ts_basic(self, video_paths):
        from positronic.dataset.video import VideoSignal
        with VideoSignalWriter(video_paths['video'], video_paths['frames'], gop_size=5, fps=30) as writer:
            writer.append(create_frame(10), 1000)
            writer.append(create_frame(20), 2000)
            writer.append(create_frame(30), 4000)

        s = VideoSignal(video_paths['video'], video_paths['frames'])
        assert s.start_ts == 1000
        assert s.last_ts == 4000

    def test_video_start_last_ts_empty_raises(self, video_paths):
        from positronic.dataset.video import VideoSignal
        with VideoSignalWriter(video_paths['video'], video_paths['frames']):
            pass
        s = VideoSignal(video_paths['video'], video_paths['frames'])
        with pytest.raises(ValueError):
            _ = s.start_ts
        with pytest.raises(ValueError):
            _ = s.last_ts


class TestVideoInterface:

    def test_len_values_ts_at(self, video_paths):
        sig = create_video_signal(video_paths, [(create_frame(50), 1000), (create_frame(100), 2000)])
        assert len(sig) == 2
        frame0, ts0 = sig[0]
        assert ts0 == 1000
        assert_frames_equal(frame0, create_frame(50))
        assert sig._ts_at([1])[0] == 2000

    def test_video_kind_and_names(self, video_paths):
        sig = create_video_signal(video_paths, [(create_frame(10), 1000)])
        assert sig.kind == Kind.IMAGE
        assert sig.names == ["height", "width", "channel"]

    def test_video_kind_names_empty_raises(self, video_paths):
        # Create empty video index
        with VideoSignalWriter(video_paths['video'], video_paths['frames']):
            pass
        s = VideoSignal(video_paths['video'], video_paths['frames'])
        with pytest.raises(ValueError):
            _ = s.kind
        with pytest.raises(ValueError):
            _ = s.names

    def test_video_view_meta_inherits_and_empty_view_raises(self, video_paths):
        sig = create_video_signal(video_paths, [(create_frame(10), 1000), (create_frame(20), 2000)])
        view = sig[0:2]
        assert view.kind == Kind.IMAGE
        assert view.names == ["height", "width", "channel"]
        empty_view = sig[0:0]
        with pytest.raises(ValueError):
            _ = empty_view.kind
        with pytest.raises(ValueError):
            _ = empty_view.names

    def test_search_ts_empty_and_numeric(self, video_paths):
        sig = create_video_signal(video_paths, [(create_frame(50), 1000)])
        empty = sig._search_ts(np.array([], dtype=np.int64))
        assert isinstance(empty, np.ndarray)
        assert empty.size == 0
        # Accept float array; floor index for one element
        idx = sig._search_ts(np.array([999.9, 1000.0, 1000.1], dtype=np.float64))
        assert np.array_equal(idx, np.array([-1, 0, 0]))
        # Accept scalar float via list-like
        assert sig._search_ts([1000.0])[0] == 0
