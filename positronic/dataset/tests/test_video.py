from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import pytest
from av.codec.codec import UnknownCodecError

from positronic.dataset.signal import Kind
from positronic.dataset.video import LibavEncoder, VideoSignal, VideoSignalWriter


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


# ``FakeEncoder.fail_at`` value that fails the flush rather than a frame
FINISH = -1


@dataclass
class FakeSession:
    path: Path
    fail_at: int | None
    writes: list[tuple[int, int]] = field(default_factory=list)
    ended: str | None = None

    def write(self, frame: np.ndarray, index: int) -> None:
        if index == self.fail_at:
            raise OSError('encoder died')
        self.writes.append((index, int(frame[0, 0, 0])))

    def finish(self) -> None:
        if self.fail_at == FINISH:
            raise OSError('flush failed')
        self.ended = 'finish'

    def abort(self) -> None:
        self.ended = 'abort'


@dataclass
class FakeEncoder:
    fail_at: int | None = None
    opened: list[tuple[int, int, int, int]] = field(default_factory=list)
    sessions: list[FakeSession] = field(default_factory=list)

    def ensure_available(self) -> None:
        pass

    def open(self, path: Path, width: int, height: int, fps: int, gop: int) -> FakeSession:
        path.write_bytes(b'partial')
        self.opened.append((width, height, fps, gop))
        self.sessions.append(FakeSession(path, self.fail_at))
        return self.sessions[-1]


class TestVideoEncoderSeam:
    def test_encoder_gets_every_frame_in_order_then_finishes(self, video_paths):
        encoder = FakeEncoder()
        with VideoSignalWriter(video_paths['video'], video_paths['frames'], encoder, gop_size=12, fps=50) as w:
            for i in range(20):
                w.append(create_frame(i, (6, 8, 3)), 1000 * (i + 1))

        assert encoder.opened == [(8, 6, 50, 12)]
        (session,) = encoder.sessions
        assert session.writes == [(i, i) for i in range(20)]
        assert session.ended == 'finish'
        assert len(pq.read_table(video_paths['frames'])) == 20

    def test_empty_writer_never_opens_the_encoder(self, video_paths):
        encoder = FakeEncoder()
        with VideoSignalWriter(video_paths['video'], video_paths['frames'], encoder):
            pass
        assert encoder.sessions == []

    def test_abort_stops_the_encoder_and_deletes_the_files(self, video_paths):
        encoder = FakeEncoder()
        w = VideoSignalWriter(video_paths['video'], video_paths['frames'], encoder)
        w.append(create_frame(0), 1000)
        w.abort()

        assert encoder.sessions[0].ended == 'abort'
        assert not video_paths['video'].exists()
        assert not video_paths['frames'].exists()

    def test_an_encoder_error_surfaces_on_append(self, video_paths):
        w = VideoSignalWriter(video_paths['video'], video_paths['frames'], FakeEncoder(fail_at=0))
        # An append past the 8 queued frames waits for the failed write and sees its error.
        with pytest.raises(RuntimeError, match='Video encoding failed'):
            for i in range(20):
                w.append(create_frame(i), 1000 * (i + 1))
        w.abort()

    def test_an_encoder_error_surfaces_on_exit_and_aborts_the_session(self, video_paths):
        encoder = FakeEncoder(fail_at=1)
        w = VideoSignalWriter(video_paths['video'], video_paths['frames'], encoder)
        w.append(create_frame(0), 1000)
        w.append(create_frame(1), 2000)
        with pytest.raises(RuntimeError, match='Video encoding failed'):
            w.__exit__(None, None, None)
        assert encoder.sessions[0].ended == 'abort'

    def test_a_failed_finish_surfaces_on_exit(self, video_paths):
        w = VideoSignalWriter(video_paths['video'], video_paths['frames'], FakeEncoder(fail_at=FINISH))
        w.append(create_frame(0), 1000)
        with pytest.raises(RuntimeError, match='Video encoding failed'):
            w.__exit__(None, None, None)


class TestLibavEncoder:
    def test_options_reach_the_codec(self, video_paths):
        encoder = LibavEncoder(options={'preset': 'ultrafast', 'bframes': '0'})
        with VideoSignalWriter(video_paths['video'], video_paths['frames'], encoder) as w:
            for i in range(5):
                w.append(create_frame(i * 40), 1000 * (i + 1))

        with av.open(str(video_paths['video'])) as container:
            (stream,) = container.streams.video
            assert stream.codec_context.name == 'h264'
            assert not stream.codec_context.has_b_frames

    def test_an_absent_codec_is_refused(self):
        with pytest.raises(UnknownCodecError):
            LibavEncoder(codec='no-such-codec').ensure_available()
        LibavEncoder().ensure_available()


class TestVideoSignalStartLastTs:
    def test_video_start_last_ts_basic(self, video_paths):
        with VideoSignalWriter(video_paths['video'], video_paths['frames'], gop_size=5, fps=30) as writer:
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
        assert {'ts_ns', 'ts_ns.consumer', 'ts_ns.producer'} == set(table.column_names)

        # Verify the data
        assert table['ts_ns'].to_pylist() == [1000, 2000, 3000]
        assert table['ts_ns.producer'].to_pylist() == [900, 1900, 2900]
        assert table['ts_ns.consumer'].to_pylist() == [1100, 2100, 3100]

    def test_video_writer_empty_with_no_extra_timelines(self, video_paths):
        """Test empty video writer doesn't create extra timeline columns."""
        with VideoSignalWriter(video_paths['video'], video_paths['frames']):
            pass

        table = pq.read_table(video_paths['frames'])
        assert {'ts_ns'} == set(table.column_names)
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
