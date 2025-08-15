import numpy as np
import pytest
from pathlib import Path
import pyarrow.parquet as pq

from pimm.dataset.video import VideoSignalWriter


@pytest.fixture
def video_paths(tmp_path):
    """Create paths for video and index files."""
    return {
        'video': tmp_path / "test.mp4",
        'frames': tmp_path / "frames.parquet"
    }


@pytest.fixture
def writer(video_paths):
    """Create a VideoSignalWriter instance."""
    return VideoSignalWriter(
        video_paths['video'],
        video_paths['frames']
    )


def create_frame(value=0, shape=(100, 100, 3)):
    """Create a test frame with given value and shape."""
    return np.full(shape, value, dtype=np.uint8)


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
        writer = VideoSignalWriter(
            video_paths['video'],
            video_paths['frames'],
            gop_size=5
        )
        
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