import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from positronic.dataset.signal import Kind
from positronic.dataset.video import VideoSignal, VideoSignalWriter


@pytest.fixture
def video_paths(tmp_path):
    """Create paths for video and index files."""
    return {'video': tmp_path / 'test.mp4', 'frames': tmp_path / 'frames.parquet'}


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
    assert frame1.shape == frame2.shape, f'Shape mismatch: {frame1.shape} != {frame2.shape}'
    assert frame1.dtype == frame2.dtype, f'Dtype mismatch: {frame1.dtype} != {frame2.dtype}'

    # Compare median values to account for compression artifacts
    median1 = np.median(frame1)
    median2 = np.median(frame2)
    assert median1 == pytest.approx(median2, abs=tolerance), (
        f'Frame content mismatch: median {median1} != {median2} (tolerance={tolerance})'
    )


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
            (np.zeros((100, 100), dtype=np.uint8), 'Expected frame shape'),  # 2D
            (np.zeros((100, 100, 4), dtype=np.uint8), 'Expected frame shape'),  # 4 channels
        ]

        for frame, match in invalid_frames:
            with VideoSignalWriter(video_paths['video'], video_paths['frames']) as writer:
                with pytest.raises(ValueError, match=match):
                    writer.append(frame, 1000)

    def test_invalid_dtype(self, writer):
        """Test that invalid dtypes are rejected."""
        frame = np.zeros((100, 100, 3), dtype=np.float32)
        with writer:
            with pytest.raises(ValueError, match='Expected uint8 dtype'):
                writer.append(frame, 1000)

    def test_non_increasing_timestamp(self, writer):
        """Test that non-increasing timestamps are rejected."""
        frame1 = create_frame(0)
        frame2 = create_frame(1)
        with writer as w:
            w.append(frame1, 2000)
            # Try same and earlier timestamps
            for ts in [2000, 1000]:
                with pytest.raises(ValueError, match='not increasing'):
                    w.append(frame2, ts)

    def test_inconsistent_dimensions(self, writer):
        """Test that frame dimensions must be consistent."""
        with writer as w:
            w.append(create_frame(0, (100, 100, 3)), 1000)
            # Different dimensions should fail
            with pytest.raises(ValueError, match='Frame shape'):
                w.append(create_frame(0, (50, 50, 3)), 2000)

    def test_append_after_context_exit(self, writer):
        """Test that appending after finish raises an error."""
        frame = create_frame()
        with writer as w:
            w.append(frame, 1000)
        with pytest.raises(RuntimeError, match='Cannot append to a finished writer'):
            w.append(frame, 2000)


class TestVideoSignalStartLastTs:
    def test_video_start_last_ts_basic(self, video_paths):
        with VideoSignalWriter(video_paths['video'], video_paths['frames'], gop_size=5) as writer:
            writer.append(create_frame(10), 1000)
            writer.append(create_frame(20), 2000)
            writer.append(create_frame(30), 4000)

        s = VideoSignal(video_paths['video'], video_paths['frames'])
        assert s.start_ts == 1000
        assert s.last_ts == 4000

    def test_video_start_last_ts_empty_raises(self, video_paths):
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

    def test_video_kind(self, video_paths):
        sig = create_video_signal(video_paths, [(create_frame(10), 1000)])
        assert sig.kind == Kind.IMAGE

    def test_video_kind_empty_raises(self, video_paths):
        # Create empty video index
        with VideoSignalWriter(video_paths['video'], video_paths['frames']):
            pass
        s = VideoSignal(video_paths['video'], video_paths['frames'])
        with pytest.raises(ValueError):
            _ = s.kind

    def test_video_view_meta_inherits_and_empty_view_raises(self, video_paths):
        sig = create_video_signal(video_paths, [(create_frame(10), 1000), (create_frame(20), 2000)])
        view = sig[0:2]
        assert view.kind == Kind.IMAGE
        empty_view = sig[0:0]
        with pytest.raises(ValueError):
            _ = empty_view.kind

    def test_search_ts_empty_and_numeric(self, video_paths):
        sig = create_video_signal(video_paths, [(create_frame(50), 1000)])
        empty = sig._search_ts(np.array([], dtype=np.int64))
        assert isinstance(empty, np.ndarray)
        assert empty.size == 0
        idx = sig._search_ts(np.array([999, 1000, 1001], dtype=np.int64))
        assert np.array_equal(idx, np.array([-1, 0, 0]))
        assert sig._search_ts([1000])[0] == 0


class TestVideoExtraTimelines:
    def test_video_writer_with_extra_timelines(self, video_paths):
        """Test that VideoSignalWriter stores extra timelines in frames index."""
        with VideoSignalWriter(video_paths['video'], video_paths['frames']) as w:
            w.append(create_frame(50), 1000, extra_ts={'producer': 900, 'consumer': 1100})
            w.append(create_frame(100), 2000, extra_ts={'producer': 1900, 'consumer': 2100})
            w.append(create_frame(150), 3000, extra_ts={'producer': 2900, 'consumer': 3100})

        # Read the frames index directly
        table = pq.read_table(video_paths['frames'])
        assert {'ts_ns', 'pts', 'ts_ns.consumer', 'ts_ns.producer'} == set(table.column_names)

        # Verify the data
        assert table['ts_ns'].to_pylist() == [1000, 2000, 3000]
        assert table['ts_ns.producer'].to_pylist() == [900, 1900, 2900]
        assert table['ts_ns.consumer'].to_pylist() == [1100, 2100, 3100]

    def test_video_writer_empty_with_no_extra_timelines(self, video_paths):
        """Test empty video writer doesn't create extra timeline columns."""
        with VideoSignalWriter(video_paths['video'], video_paths['frames']):
            pass

        table = pq.read_table(video_paths['frames'])
        assert {'ts_ns', 'pts'} == set(table.column_names)
        assert len(table) == 0

    def test_video_inconsistent_extra_ts_keys_raises(self, video_paths):
        """Test that inconsistent extra_ts keys across appends raises ValueError."""
        with pytest.raises(ValueError, match='extra_ts keys must be consistent'):
            with VideoSignalWriter(video_paths['video'], video_paths['frames']) as w:
                w.append(create_frame(50), 1000, extra_ts={'producer': 900})
                w.append(create_frame(100), 2000, extra_ts={'producer': 1900, 'consumer': 2100})

    def test_video_missing_extra_ts_after_first_raises(self, video_paths):
        """Test that omitting extra_ts after providing it first raises ValueError."""
        with pytest.raises(ValueError, match='extra_ts keys must be consistent'):
            with VideoSignalWriter(video_paths['video'], video_paths['frames']) as w:
                w.append(create_frame(50), 1000, extra_ts={'producer': 900})
                w.append(create_frame(100), 2000)

    def test_video_adding_extra_ts_after_none_raises(self, video_paths):
        """Test that adding extra_ts after first append without it raises ValueError."""
        with pytest.raises(ValueError, match='extra_ts keys must be consistent'):
            with VideoSignalWriter(video_paths['video'], video_paths['frames']) as w:
                w.append(create_frame(50), 1000)
                w.append(create_frame(100), 2000, extra_ts={'producer': 1900})


# Frames at ~15 Hz with jitter and a long gap: spacings a single frame rate could not describe.
JITTERY_TS_NS = [
    1_000_000_000,
    1_066_700_000,
    1_133_000_000,
    1_201_400_000,
    1_268_000_000,
    1_600_000_000,
    1_667_100_000,
    1_733_000_000,
    1_800_900_000,
    1_866_000_000,
    1_934_000_000,
    2_000_000_000,
]


def decoded_pts(video_path):
    """Presentation timestamps and time base a plain container reader sees, in decode order."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        assert time_base is not None
        pts = []
        for frame in container.decode(stream):
            assert frame.pts is not None
            pts.append(frame.pts)
        return pts, time_base


class TestVideoContainerTiming:
    def test_container_timestamps_reproduce_recorded_span(self, video_paths):
        sig = create_video_signal(video_paths, [(create_frame(10 + i * 20), ts) for i, ts in enumerate(JITTERY_TS_NS)])
        assert len(sig) == len(JITTERY_TS_NS)

        pts, time_base = decoded_pts(video_paths['video'])
        assert len(pts) == len(JITTERY_TS_NS)
        # An external player derives frame times from pts * time_base; they must match the recorded
        # timestamps, to within the microsecond the container resolves.
        offsets_ns = [int(p * time_base * 1_000_000_000) for p in pts]
        expected_ns = [ts - JITTERY_TS_NS[0] for ts in JITTERY_TS_NS]
        assert offsets_ns == pytest.approx(expected_ns, abs=1000)

    def test_index_pts_are_what_the_container_carries(self, video_paths):
        create_video_signal(video_paths, [(create_frame(10 + i * 20), ts) for i, ts in enumerate(JITTERY_TS_NS)])

        indexed = pq.read_table(video_paths['frames'])['pts'].to_pylist()
        pts, _ = decoded_pts(video_paths['video'])
        assert indexed == pts

    def test_random_access_with_jittery_timestamps(self, video_paths):
        frames = [(create_frame(10 + i * 20), ts) for i, ts in enumerate(JITTERY_TS_NS)]
        with VideoSignalWriter(video_paths['video'], video_paths['frames'], gop_size=3) as writer:
            for frame, ts in frames:
                writer.append(frame, ts)
        sig = VideoSignal(video_paths['video'], video_paths['frames'], seek_threshold=2)

        # Reverse and then scattered order, so every read has to seek rather than decode onwards.
        for i in list(reversed(range(len(frames)))) + [0, 7, 3, 11, 5]:
            frame, ts = sig[i]
            assert ts == JITTERY_TS_NS[i]
            assert_frames_equal(frame, frames[i][0], tolerance=8)

    def test_search_by_time_lands_on_the_right_frame(self, video_paths):
        sig = create_video_signal(video_paths, [(create_frame(10 + i * 20), ts) for i, ts in enumerate(JITTERY_TS_NS)])
        # A time inside the long gap resolves to the frame that was current then, not to a frame
        # position derived from an assumed constant rate.
        frame, ts = sig.time[1_500_000_000]
        assert ts == JITTERY_TS_NS[4]
        assert_frames_equal(frame, create_frame(10 + 4 * 20), tolerance=8)


class TestSubMicrosecondFrames:
    def test_pts_stay_strictly_increasing_within_one_microsecond(self, video_paths):
        # The first three land in the same microsecond once rounded; the last is well clear.
        timestamps = [1_000_000_000, 1_000_000_100, 1_000_000_200, 1_000_500_000]
        frames = [(create_frame(10 + i * 40), ts) for i, ts in enumerate(timestamps)]
        sig = create_video_signal(video_paths, frames)

        indexed = pq.read_table(video_paths['frames'])['pts'].to_pylist()
        assert indexed == [0, 1, 2, 500]
        assert np.all(np.diff(indexed) > 0)

        pts, _ = decoded_pts(video_paths['video'])
        assert pts == indexed
        for i, (frame, ts) in enumerate(frames):
            got_frame, got_ts = sig[i]
            assert got_ts == ts
            assert_frames_equal(got_frame, frame, tolerance=8)


def write_legacy_video(video_paths, frames_with_timestamps, declared_fps=100, gop_size=30):
    """Write a video the way it was written before the frames index carried presentation timestamps.

    Presentation timestamps are the ordinal frame number against a fixed declared rate, and the index
    holds timestamps only. Written out explicitly so the fallback is tested against the real old format
    rather than against whatever the current writer produces.
    """
    container = av.open(str(video_paths['video']), mode='w')
    stream = container.add_stream('h264', rate=declared_fps)
    first_frame = frames_with_timestamps[0][0]
    stream.height, stream.width = first_frame.shape[:2]
    stream.pix_fmt = 'yuv420p'
    stream.gop_size = gop_size

    for i, (data, _ts) in enumerate(frames_with_timestamps):
        frame = av.VideoFrame.from_ndarray(data, format='rgb24')
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    timestamps = [ts for _data, ts in frames_with_timestamps]
    pq.write_table(pa.table({'ts_ns': timestamps}, schema=pa.schema([('ts_ns', pa.int64())])), video_paths['frames'])
    return VideoSignal(video_paths['video'], video_paths['frames'])


class TestIndexWithoutPtsColumn:
    def test_legacy_video_reads_through_declared_rate(self, video_paths):
        frames = [(create_frame(10 + i * 20), ts) for i, ts in enumerate(JITTERY_TS_NS)]
        sig = write_legacy_video(video_paths, frames, gop_size=3)

        assert 'pts' not in pq.read_table(video_paths['frames']).column_names
        assert len(sig) == len(frames)
        assert sig.start_ts == JITTERY_TS_NS[0]
        assert sig.last_ts == JITTERY_TS_NS[-1]

        for i in list(range(len(frames))) + list(reversed(range(len(frames)))) + [0, 7, 3]:
            frame, ts = sig[i]
            assert ts == JITTERY_TS_NS[i]
            assert_frames_equal(frame, frames[i][0], tolerance=8)

    def test_legacy_video_ordinal_pts_are_left_alone(self, video_paths):
        frames = [(create_frame(10 + i * 20), ts) for i, ts in enumerate(JITTERY_TS_NS)]
        write_legacy_video(video_paths, frames)

        # The fallback reads the file as written; it does not rewrite or reinterpret its timing.
        pts, time_base = decoded_pts(video_paths['video'])
        ticks_per_frame = round(1 / (100 * time_base))
        assert pts == [i * ticks_per_frame for i in range(len(frames))]
