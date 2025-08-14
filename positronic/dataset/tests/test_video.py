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
        # Timestamps are stored as int64
        assert frames_table['ts_ns'][0].as_py() == 1000

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
        # Verify timestamps match what we wrote
        stored_ts = [t.as_py() for t in frames_table['ts_ns']]
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


class TestVideoSignalSliceAccess:
    """Test slice-based access to VideoSignal."""

    def test_slice_basic(self, video_paths):
        """Test basic slicing."""
        expected_frames = [create_frame(value=50), create_frame(value=100), create_frame(value=150), create_frame(value=200)]
        frames_with_ts = [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Slice [1:3]
        sliced = signal[1:3]
        assert len(sliced) == 2

        frame, ts = sliced[0]
        assert ts == 2000
        assert_frames_equal(frame, expected_frames[1])

        frame, ts = sliced[1]
        assert ts == 3000
        assert_frames_equal(frame, expected_frames[2])

    def test_slice_with_negative_indices(self, video_paths):
        """Test slicing with negative indices."""
        expected_frames = [create_frame(value=50), create_frame(value=100), create_frame(value=150)]
        frames_with_ts = [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Slice [-2:]
        sliced = signal[-2:]
        assert len(sliced) == 2

        frame, ts = sliced[0]
        assert ts == 2000
        assert_frames_equal(frame, expected_frames[1])

        frame, ts = sliced[1]
        assert ts == 3000
        assert_frames_equal(frame, expected_frames[2])

    def test_slice_with_positive_step(self, video_paths):
        """Test that positive steps work correctly."""
        expected_frames = [create_frame(value=i*50) for i in range(5)]
        frames_with_ts = [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Step=2 should give us frames 0, 2, 4
        sliced = signal[::2]
        assert len(sliced) == 3

        frame, ts = sliced[0]
        assert ts == 1000
        assert_frames_equal(frame, expected_frames[0])

        frame, ts = sliced[1]
        assert ts == 3000
        assert_frames_equal(frame, expected_frames[2])

        frame, ts = sliced[2]
        assert ts == 5000
        assert_frames_equal(frame, expected_frames[4])

    def test_slice_negative_step_raises(self, video_paths):
        """Test that negative/zero steps raise IndexError."""
        signal = create_video_signal(video_paths, [(create_frame(value=100), 1000), (create_frame(value=150), 2000)])

        with pytest.raises(IndexError, match="Slice step must be positive"):
            signal[::-1]

        with pytest.raises(IndexError, match="Slice step must be positive"):
            signal[1::-1]

    def test_slice_of_slice(self, video_paths):
        """Test slicing a sliced signal."""
        expected_frames = [create_frame(value=i*50) for i in range(5)]
        frames_with_ts = [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # First slice [1:4] -> frames 1,2,3
        sliced1 = signal[1:4]
        assert len(sliced1) == 3

        # Second slice [1:] of first slice -> frames 2,3
        sliced2 = sliced1[1:]
        assert len(sliced2) == 2

        frame, ts = sliced2[0]
        assert ts == 3000
        assert_frames_equal(frame, expected_frames[2])

        frame, ts = sliced2[1]
        assert ts == 4000
        assert_frames_equal(frame, expected_frames[3])

    def test_slice_empty(self, video_paths):
        """Test empty slices."""
        signal = create_video_signal(video_paths, [(create_frame(value=100), 1000), (create_frame(value=150), 2000)])

        # Empty slice
        sliced = signal[2:2]
        assert len(sliced) == 0

        # Out of range slice
        sliced = signal[5:10]
        assert len(sliced) == 0

    def test_slice_negative_index_access(self, video_paths):
        """Test negative indexing within a slice."""
        expected_frames = [create_frame(value=i*50) for i in range(4)]
        frames_with_ts = [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)]
        signal = create_video_signal(video_paths, frames_with_ts)

        sliced = signal[1:3]  # frames 1,2

        # Access with negative index
        frame, ts = sliced[-1]
        assert ts == 3000
        assert_frames_equal(frame, expected_frames[2])

        frame, ts = sliced[-2]
        assert ts == 2000
        assert_frames_equal(frame, expected_frames[1])

    def test_slice_out_of_bounds(self, video_paths):
        """Test out of bounds access in slice."""
        signal = create_video_signal(video_paths, [(create_frame(value=100), 1000), (create_frame(value=150), 2000)])

        sliced = signal[0:1]
        assert len(sliced) == 1

        with pytest.raises(IndexError):
            sliced[1]

        with pytest.raises(IndexError):
            sliced[-2]


class TestVideoSignalArrayIndexAccess:
    """Test array-based index access to VideoSignal."""

    def test_array_index_basic(self, video_paths):
        """Test basic array indexing."""
        expected_frames = [create_frame(value=i*50) for i in range(5)]
        frames_with_ts = [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Access frames 0, 2, 4 via array
        view = signal[[0, 2, 4]]
        assert len(view) == 3

        frame, ts = view[0]
        assert ts == 1000
        assert_frames_equal(frame, expected_frames[0])

        frame, ts = view[1]
        assert ts == 3000
        assert_frames_equal(frame, expected_frames[2])

        frame, ts = view[2]
        assert ts == 5000
        assert_frames_equal(frame, expected_frames[4])

    def test_array_index_with_negatives(self, video_paths):
        """Test array indexing with negative indices."""
        expected_frames = [create_frame(value=i*50) for i in range(4)]
        frames_with_ts = [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Mix positive and negative indices
        view = signal[[0, -1, 1, -2]]
        assert len(view) == 4

        assert view[0][1] == 1000  # frame 0
        assert view[1][1] == 4000  # frame -1 (last)
        assert view[2][1] == 2000  # frame 1
        assert view[3][1] == 3000  # frame -2

    def test_array_index_numpy(self, video_paths):
        """Test array indexing with numpy array."""
        expected_frames = [create_frame(value=i*50) for i in range(5)]
        frames_with_ts = [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Use numpy array for indexing
        indices = np.array([1, 3, 2])
        view = signal[indices]
        assert len(view) == 3

        assert view[0][1] == 2000
        assert view[1][1] == 4000
        assert view[2][1] == 3000

    def test_array_index_out_of_bounds(self, video_paths):
        """Test that out of bounds array indices raise IndexError."""
        signal = create_video_signal(video_paths, [(create_frame(value=100), 1000), (create_frame(value=150), 2000)])

        with pytest.raises(IndexError):
            signal[[0, 5]]

        with pytest.raises(IndexError):
            signal[[-3, 0]]

    def test_array_index_on_slice(self, video_paths):
        """Test array indexing on a sliced signal."""
        expected_frames = [create_frame(value=i*50) for i in range(6)]
        frames_with_ts = [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # First take a slice [1:5] -> frames 1,2,3,4
        sliced = signal[1:5]

        # Then use array indexing [0, 2] -> frames 1, 3 from original
        view = sliced[[0, 2]]
        assert len(view) == 2

        frame, ts = view[0]
        assert ts == 2000
        assert_frames_equal(frame, expected_frames[1])

        frame, ts = view[1]
        assert ts == 4000
        assert_frames_equal(frame, expected_frames[3])


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


class TestVideoSignalTimeSliceAccess:
    """Test time-based slice access to VideoSignal."""

    def test_time_slice_basic(self, video_paths):
        """Test basic time slicing."""
        expected_frames = [create_frame(value=50), create_frame(value=100), create_frame(value=150), create_frame(value=200)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(4)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Time slice [1500:3500] should include frames at 2000 and 3000
        sliced = signal.time[1500:3500]
        assert len(sliced) == 2

        frame, ts = sliced[0]
        assert ts == 2000
        assert_frames_equal(frame, expected_frames[1])

        frame, ts = sliced[1]
        assert ts == 3000
        assert_frames_equal(frame, expected_frames[2])

    def test_time_slice_inclusive_exclusive(self, video_paths):
        """Test that time slicing is inclusive of start, exclusive of stop."""
        expected_frames = [create_frame(value=50), create_frame(value=100), create_frame(value=150)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(3)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Exact boundaries
        sliced = signal.time[1000:3000]
        assert len(sliced) == 2
        assert sliced[0][1] == 1000
        assert sliced[1][1] == 2000

    def test_time_slice_no_start(self, video_paths):
        """Test time slicing with no start time."""
        expected_frames = [create_frame(value=50), create_frame(value=100), create_frame(value=150)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(3)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # No start means from beginning
        sliced = signal.time[:2500]
        assert len(sliced) == 2
        assert sliced[0][1] == 1000
        assert sliced[1][1] == 2000

    def test_time_slice_no_stop(self, video_paths):
        """Test time slicing with no stop time."""
        expected_frames = [create_frame(value=50), create_frame(value=100), create_frame(value=150)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(3)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # No stop means to end
        sliced = signal.time[1500:]
        assert len(sliced) == 2
        assert sliced[0][1] == 2000
        assert sliced[1][1] == 3000

    def test_time_slice_full(self, video_paths):
        """Test full time slice."""
        expected_frames = [create_frame(value=50), create_frame(value=100)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(2)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Full slice
        sliced = signal.time[:]
        assert len(sliced) == 2
        assert sliced[0][1] == 1000
        assert sliced[1][1] == 2000

    def test_time_slice_empty(self, video_paths):
        """Test empty time slices."""
        signal = create_video_signal(video_paths, [(create_frame(value=100), 2000), (create_frame(value=150), 3000)])

        # Before all data
        sliced = signal.time[500:1000]
        assert len(sliced) == 0

        # After all data
        sliced = signal.time[4000:5000]
        assert len(sliced) == 0

        # Empty range
        sliced = signal.time[2500:2500]
        assert len(sliced) == 0

    def test_time_slice_with_step_raises(self, video_paths):
        """Test that time slicing with step raises NotImplementedError."""
        signal = create_video_signal(video_paths, [(create_frame(value=100), 1000)])

        with pytest.raises(NotImplementedError, match="step is not supported"):
            signal.time[1000:3000:500]

    def test_time_slice_of_slice(self, video_paths):
        """Test time slicing of a time-sliced signal."""
        expected_frames = [create_frame(value=i*50) for i in range(5)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(5)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # First time slice [1500:4500] -> frames at 2000, 3000, 4000
        sliced1 = signal.time[1500:4500]
        assert len(sliced1) == 3

        # Second time slice [2500:3500] of first -> frame at 3000
        sliced2 = sliced1.time[2500:3500]
        assert len(sliced2) == 1

        frame, ts = sliced2[0]
        assert ts == 3000
        assert_frames_equal(frame, expected_frames[2])

    def test_time_slice_then_index(self, video_paths):
        """Test integer indexing on a time-sliced signal."""
        expected_frames = [create_frame(value=i*50) for i in range(4)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(4)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Time slice then index
        sliced = signal.time[1500:3500]  # frames at 2000, 3000

        frame, ts = sliced[0]
        assert ts == 2000
        assert_frames_equal(frame, expected_frames[1])

        frame, ts = sliced[-1]
        assert ts == 3000
        assert_frames_equal(frame, expected_frames[2])

    def test_index_slice_then_time_slice(self, video_paths):
        """Test time slicing on an index-sliced signal."""
        expected_frames = [create_frame(value=i*50) for i in range(5)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(5)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Index slice [1:4] -> frames at 2000, 3000, 4000
        index_sliced = signal[1:4]
        assert len(index_sliced) == 3

        # Time slice [2500:3500] -> frame at 3000
        time_sliced = index_sliced.time[2500:3500]
        assert len(time_sliced) == 1

        frame, ts = time_sliced[0]
        assert ts == 3000
        assert_frames_equal(frame, expected_frames[2])
