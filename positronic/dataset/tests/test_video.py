import numpy as np
import pyarrow.parquet as pq
import pytest

from pimm.dataset.video import VideoSignal, VideoSignalWriter


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
    writer = VideoSignalWriter(video_paths['video'], video_paths['frames'])
    for frame, ts in frames_with_timestamps:
        writer.append(frame, ts)
    writer.finish()
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
        writer.finish()

        # Check that index file exists and has correct schema
        assert video_paths['frames'].exists()
        table = pq.read_table(video_paths['frames'])
        assert len(table) == 0
        assert 'ts_ns' in table.column_names

    def test_write_single_frame(self, writer, video_paths):
        """Test writing a single frame."""
        frame = create_frame(value=128)
        writer.append(frame, 1000)
        writer.finish()

        # Check video file was created
        assert video_paths['video'].exists()
        assert video_paths['video'].stat().st_size > 0

        # Check index file has exactly one timestamp
        frames_table = pq.read_table(video_paths['frames'])
        assert len(frames_table) == 1
        # Convert timestamp to nanoseconds
        assert frames_table['ts_ns'][0].value == 1000

    def test_write_multiple_frames(self, video_paths):
        """Test writing multiple frames with increasing timestamps."""
        writer = VideoSignalWriter(video_paths['video'], video_paths['frames'], gop_size=5)

        # Write 10 frames
        timestamps = [1000 * (i + 1) for i in range(10)]
        for i, ts in enumerate(timestamps):
            writer.append(create_frame(i * 25, (50, 50, 3)), ts)
        writer.finish()

        # Should have exactly 10 timestamps in the index
        frames_table = pq.read_table(video_paths['frames'])
        assert len(frames_table) == 10
        # Verify timestamps match what we wrote (convert from timestamp to ns)
        stored_ts = [t.value for t in frames_table['ts_ns']]
        assert stored_ts == timestamps

    def test_invalid_frame_shape(self, writer):
        """Test that invalid frame shapes are rejected."""
        invalid_frames = [
            (np.zeros((100, 100), dtype=np.uint8), "Expected frame shape"),  # 2D
            (np.zeros((100, 100, 4), dtype=np.uint8), "Expected frame shape"),  # 4 channels
        ]

        for frame, match in invalid_frames:
            with pytest.raises(ValueError, match=match):
                writer.append(frame, 1000)

    def test_invalid_dtype(self, writer):
        """Test that invalid dtypes are rejected."""
        frame = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected uint8 dtype"):
            writer.append(frame, 1000)

    def test_non_increasing_timestamp(self, writer):
        """Test that non-increasing timestamps are rejected."""
        frame1 = create_frame(0)
        frame2 = create_frame(1)

        writer.append(frame1, 2000)

        # Try same and earlier timestamps
        for ts in [2000, 1000]:
            with pytest.raises(ValueError, match="not increasing"):
                writer.append(frame2, ts)

    def test_inconsistent_dimensions(self, writer):
        """Test that frame dimensions must be consistent."""
        writer.append(create_frame(0, (100, 100, 3)), 1000)

        # Different dimensions should fail
        with pytest.raises(ValueError, match="Frame shape"):
            writer.append(create_frame(0, (50, 50, 3)), 2000)

    def test_append_after_finish(self, writer):
        """Test that appending after finish raises an error."""
        frame = create_frame()
        writer.append(frame, 1000)
        writer.finish()

        with pytest.raises(RuntimeError, match="Cannot append to a finished writer"):
            writer.append(frame, 2000)


class TestVideoSignalIndexAccess:
    """Test integer index access to VideoSignal."""

    def test_empty_signal(self, video_paths):
        """Test reading an empty video signal."""
        signal = create_video_signal(video_paths, [])
        assert len(signal) == 0

        with pytest.raises(IndexError):
            signal[0]

    def test_single_frame(self, video_paths):
        """Test reading a single frame."""
        expected_frame = create_frame(value=128)
        signal = create_video_signal(video_paths, [(expected_frame, 1000)])
        assert len(signal) == 1

        frame, ts = signal[0]
        assert ts == 1000
        assert_frames_equal(frame, expected_frame)

    def test_multiple_frames(self, video_paths):
        """Test reading multiple frames with different content."""
        # Create frames with distinct values
        expected_frames = [create_frame(value=50), create_frame(value=128), create_frame(value=200)]
        frames_with_ts = [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)]
        signal = create_video_signal(video_paths, frames_with_ts)
        assert len(signal) == 3

        # Check each frame
        for i in range(3):
            frame, ts = signal[i]
            assert ts == (i + 1) * 1000
            assert_frames_equal(frame, expected_frames[i])

    def test_negative_indexing(self, video_paths):
        """Test negative index access."""
        expected_frames = [create_frame(value=50), create_frame(value=128), create_frame(value=200)]
        frames_with_ts = [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Test negative indices return correct frames
        frame, ts = signal[-1]
        assert ts == 3000
        assert_frames_equal(frame, expected_frames[2])

        frame, ts = signal[-2]
        assert ts == 2000
        assert_frames_equal(frame, expected_frames[1])

        frame, ts = signal[-3]
        assert ts == 1000
        assert_frames_equal(frame, expected_frames[0])

    def test_index_out_of_range(self, video_paths):
        """Test that out of range indices raise IndexError."""
        signal = create_video_signal(video_paths, [(create_frame(value=100), 1000), (create_frame(value=150), 2000)])

        with pytest.raises(IndexError):
            signal[2]

        with pytest.raises(IndexError):
            signal[-3]

    def test_repeated_access(self, video_paths):
        """Test that we can access the same frame multiple times."""
        expected_frame = create_frame(value=150)
        signal = create_video_signal(video_paths, [(expected_frame, 1000)])

        # Access same frame multiple times
        frame1, ts1 = signal[0]
        frame2, ts2 = signal[0]

        assert ts1 == ts2 == 1000
        assert_frames_equal(frame1, expected_frame)
        assert_frames_equal(frame2, expected_frame)


class TestVideoSignalTimeAccess:
    """Test time-based access to VideoSignal."""

    def test_empty_signal(self, video_paths):
        """Test time access on empty signal."""
        signal = create_video_signal(video_paths, [])

        with pytest.raises(KeyError):
            signal.time[1000]

    def test_exact_timestamp(self, video_paths):
        """Test accessing frame at exact timestamp."""
        expected_frame = create_frame(value=128)
        signal = create_video_signal(video_paths, [(expected_frame, 1000)])

        frame, ts = signal.time[1000]
        assert ts == 1000
        assert_frames_equal(frame, expected_frame)

    def test_between_timestamps(self, video_paths):
        """Test accessing frame between timestamps."""
        expected_frames = [create_frame(value=50), create_frame(value=128), create_frame(value=200)]
        signal = create_video_signal(video_paths, [(expected_frames[0], 1000), (expected_frames[1], 2000),
                                                   (expected_frames[2], 3000)])

        # Access at 1500 should return frame at 1000
        frame, ts = signal.time[1500]
        assert ts == 1000
        assert_frames_equal(frame, expected_frames[0])

        # Access at 2500 should return frame at 2000
        frame, ts = signal.time[2500]
        assert ts == 2000
        assert_frames_equal(frame, expected_frames[1])

    def test_after_last_timestamp(self, video_paths):
        """Test accessing frame after last timestamp."""
        expected_frames = [create_frame(value=50), create_frame(value=128)]
        signal = create_video_signal(video_paths, [(expected_frames[0], 1000), (expected_frames[1], 2000)])

        # Access at 5000 should return last frame at 2000
        frame, ts = signal.time[5000]
        assert ts == 2000
        assert_frames_equal(frame, expected_frames[1])

    def test_before_first_timestamp(self, video_paths):
        """Test that accessing before first timestamp raises KeyError."""
        signal = create_video_signal(video_paths, [(create_frame(value=100), 1000)])

        with pytest.raises(KeyError):
            signal.time[999]

        with pytest.raises(KeyError):
            signal.time[500]
