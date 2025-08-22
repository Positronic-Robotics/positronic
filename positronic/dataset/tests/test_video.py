import numpy as np
import pyarrow.parquet as pq
import pytest

from positronic.dataset.video import VideoSignal, VideoSignalWriter


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


def assert_frames_equal(frame1, frame2, tolerance=3):
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

    def test_time_array_after_end_simple(self, video_paths):
        """Simple array-based time sampling beyond end returns last frame for each requested time."""
        frames = [create_frame(value=v) for v in (10, 20)]
        signal = create_video_signal(video_paths, [(frames[0], 1000), (frames[1], 2000)])

        view = signal.time[[2500]]
        assert len(view) == 1
        frame, ts = view[0]
        assert ts == 2500
        assert_frames_equal(frame, frames[-1])

    def test_long_video(self, video_paths):
        values = [(10 + i * 113) % 256 for i in range(1000)]
        frames = [create_frame(v) for v in values]
        signal = create_video_signal(video_paths, [(f, i * 1000) for i, f in enumerate(frames)])
        seek_indexes = [100, 101, 102, 300, 5, 7, 10, 14, 19, 25, 100, 200, 300, 400, 401, 402, 999]
        for index in seek_indexes:
            frame, ts = signal[index]
            assert ts == index * 1000
            assert_frames_equal(frame, frames[index])


class TestVideoSignalSliceAccess:
    """Test slice-based access to VideoSignal."""

    def test_slice_basic(self, video_paths):
        """Test basic slicing."""
        expected_frames = [
            create_frame(value=50),
            create_frame(value=100),
            create_frame(value=150),
            create_frame(value=200)
        ]
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
        expected_frames = [create_frame(value=i * 50) for i in range(5)]
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
        expected_frames = [create_frame(value=i * 50) for i in range(5)]
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
        expected_frames = [create_frame(value=i * 50) for i in range(4)]
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
        expected_frames = [create_frame(value=i * 50) for i in range(5)]
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
        expected_frames = [create_frame(value=i * 50) for i in range(4)]
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
        expected_frames = [create_frame(value=i * 50) for i in range(5)]
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
        expected_frames = [create_frame(value=i * 50) for i in range(6)]
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
        expected_frames = [
            create_frame(value=50),
            create_frame(value=100),
            create_frame(value=150),
            create_frame(value=200)
        ]
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

    def test_time_slice_with_step_basic(self, video_paths):
        """Test stepped time slicing returns requested timestamps with carry-back frames."""
        expected_frames = [create_frame(value=i * 50) for i in range(5)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(5)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Sample at 1000, 2000, 3000, 4000 (end exclusive)
        sampled = signal.time[1000:5000:1000]
        assert len(sampled) == 4

        for i in range(4):
            frame, ts = sampled[i]
            assert ts == (i + 1) * 1000
            assert_frames_equal(frame, expected_frames[i])

    def test_time_slice_with_step_offgrid(self, video_paths):
        """Test stepped time slicing at off-grid timestamps returns requested timestamps."""
        expected_frames = [create_frame(value=i * 50) for i in range(3)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(3)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Request 1500 and 2500 -> should get frames at 1000 and 2000
        sampled = signal.time[1500:3000:1000]
        assert len(sampled) == 2
        frame0, ts0 = sampled[0]
        frame1, ts1 = sampled[1]
        assert ts0 == 1500
        assert ts1 == 2500
        assert_frames_equal(frame0, expected_frames[0])
        assert_frames_equal(frame1, expected_frames[1])

    def test_time_slice_with_step_missing_start_raises(self, video_paths):
        signal = create_video_signal(video_paths, [(create_frame(value=10), 1000), (create_frame(value=20), 2000)])
        with pytest.raises(KeyError):
            _ = signal.time[:3000:1000]

    def test_time_slice_with_step_start_before_first_raises(self, video_paths):
        signal = create_video_signal(video_paths, [(create_frame(value=10), 2000), (create_frame(value=20), 3000)])
        with pytest.raises(KeyError):
            _ = signal.time[1000:4000:1000]

    def test_time_slice_with_step_missing_stop(self, video_paths):
        expected_frames = [create_frame(value=i * 10) for i in range(5)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(5)]
        signal = create_video_signal(video_paths, frames_with_ts)
        sampled = signal.time[1000::1000]
        assert len(sampled) == 5
        # First and last
        assert sampled[0][1] == 1000
        assert sampled[-1][1] == 5000

    def test_time_slice_of_slice(self, video_paths):
        """Test time slicing of a time-sliced signal."""
        expected_frames = [create_frame(value=i * 50) for i in range(5)]
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
        expected_frames = [create_frame(value=i * 50) for i in range(4)]
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
        expected_frames = [create_frame(value=i * 50) for i in range(5)]
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


class TestVideoSignalTimeArrayAccess:
    """Test time-based array access to VideoSignal."""

    def test_time_array_exact_and_offgrid(self, video_paths):
        expected_frames = [create_frame(value=i * 10) for i in range(3)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(3)]
        signal = create_video_signal(video_paths, frames_with_ts)

        req = [1000, 1500, 3000]
        view = signal.time[req]
        assert len(view) == 3
        f0, t0 = view[0]
        f1, t1 = view[1]
        f2, t2 = view[2]
        assert t0 == 1000
        assert t1 == 1500
        assert t2 == 3000
        assert_frames_equal(f0, expected_frames[0])
        assert_frames_equal(f1, expected_frames[0])
        assert_frames_equal(f2, expected_frames[2])

    def test_time_array_unsorted_and_repeated(self, video_paths):
        expected_frames = [create_frame(value=i * 10) for i in range(3)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(3)]
        signal = create_video_signal(video_paths, frames_with_ts)

        req = [3000, 1000, 3000]
        view = signal.time[req]
        assert len(view) == 3
        # Timestamps are preserved as requested
        assert view[0][1] == 3000
        assert view[1][1] == 1000
        assert view[2][1] == 3000
        # Frames correspond to carry-back values
        assert_frames_equal(view[0][0], expected_frames[2])
        assert_frames_equal(view[1][0], expected_frames[0])
        assert_frames_equal(view[2][0], expected_frames[2])

    def test_time_array_before_first_raises(self, video_paths):
        signal = create_video_signal(video_paths, [(create_frame(value=10), 1000), (create_frame(value=20), 2000)])
        with pytest.raises(KeyError):
            _ = signal.time[[500, 1000]]

    def test_time_array_before_first_float_timestamp_raises(self, video_paths):
        signal = create_video_signal(video_paths, [(create_frame(value=10), 1000), (create_frame(value=20), 2000)])
        with pytest.raises(KeyError):
            _ = signal.time[[999.9, 1500]]

    def test_time_array_empty_raises(self, video_paths):
        signal = create_video_signal(video_paths, [(create_frame(value=10), 1000), (create_frame(value=20), 2000)])
        with pytest.raises(TypeError):
            _ = signal.time[[]]
        with pytest.raises(TypeError):
            _ = signal.time[np.array([], dtype=np.int64)]

    def test_time_array_float_inputs_returns_nearest_past_frame(self, video_paths):
        signal = create_video_signal(video_paths, [(create_frame(value=10), 1000), (create_frame(value=20), 2000)])
        view = signal.time[np.array([1000.1, 2500.4], dtype=np.float64)]
        assert len(view) == 2
        assert view[0][1] == 1000
        assert view[1][1] == 2500
        assert_frames_equal(view[0][0], create_frame(value=10))
        assert_frames_equal(view[1][0], create_frame(value=20))

    def test_time_array_complex_timestamp_raises(self, video_paths):
        signal = create_video_signal(video_paths, [(create_frame(value=10), 1000), (create_frame(value=20), 2000)])
        with pytest.raises(TypeError):
            _ = signal.time[np.array([1000.1, 1500], dtype=np.complex128)]

    def test_time_array_multiple_offgrid_same_gap(self, video_paths):
        # Create three frames at 1000, 2000, 3000 ns with distinct contents
        expected_frames = [create_frame(value=i * 40) for i in range(3)]
        frames_with_ts = [(expected_frames[i], (i + 1) * 1000) for i in range(3)]
        signal = create_video_signal(video_paths, frames_with_ts)

        # Request multiple off-grid timestamps that all fall between 1000 and 2000
        req = np.array([1100, 1500, 1900], dtype=np.int64)
        view = signal.time[req]

        # Length matches requests, timestamps preserved, frames carry back to the one at 1000
        assert len(view) == len(req)
        for i, t in enumerate(req):
            frame, ts = view[i]
            assert ts == int(t)
            assert_frames_equal(frame, expected_frames[0])


class TestVideoSignalIteration:
    """Verify that VideoSignal and its views are iterable."""

    def test_iter_over_signal(self, video_paths):
        expected_frames = [create_frame(value=v) for v in (50, 100, 150)]
        signal = create_video_signal(video_paths, [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)])
        items = list(signal)
        assert len(items) == 3
        for i, (frame, ts) in enumerate(items):
            assert ts == (i + 1) * 1000
            assert_frames_equal(frame, expected_frames[i])

    def test_iter_over_index_slice(self, video_paths):
        expected_frames = [create_frame(value=i * 25) for i in range(4)]
        signal = create_video_signal(video_paths, [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)])
        view = signal[1:3]
        items = list(view)
        assert len(items) == 2
        assert items[0][1] == 2000
        assert_frames_equal(items[0][0], expected_frames[1])
        assert items[1][1] == 3000
        assert_frames_equal(items[1][0], expected_frames[2])

    def test_iter_over_time_slice(self, video_paths):
        expected_frames = [create_frame(value=i * 30) for i in range(4)]
        signal = create_video_signal(video_paths, [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)])
        view = signal.time[1500:3500]
        items = list(view)
        assert len(items) == 2
        assert items[0][1] == 2000
        assert_frames_equal(items[0][0], expected_frames[1])
        assert items[1][1] == 3000
        assert_frames_equal(items[1][0], expected_frames[2])

    def test_iter_over_time_stepped(self, video_paths):
        expected_frames = [create_frame(value=i * 40) for i in range(3)]
        signal = create_video_signal(video_paths, [(f, (i + 1) * 1000) for i, f in enumerate(expected_frames)])
        sampled = signal.time[1500:3000:1000]
        items = list(sampled)
        assert len(items) == 2
        # Requested timestamps 1500 and 2500
        assert items[0][1] == 1500
        assert_frames_equal(items[0][0], expected_frames[0])
        assert items[1][1] == 2500
        assert_frames_equal(items[1][0], expected_frames[1])


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
