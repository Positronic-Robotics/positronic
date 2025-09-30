import numpy as np

from positronic.dataset.signal import Kind
from positronic.dataset.transforms import image

from ...tests.utils import DummySignal


def test_image_resize_basic():
    # Create a simple image signal with uniform frames
    h, w = 4, 6
    frame1 = np.full((h, w, 3), 10, dtype=np.uint8)
    frame2 = np.full((h, w, 3), 200, dtype=np.uint8)
    ts = [1000, 2000]
    sig = DummySignal(ts, [frame1, frame2])

    # Resize to (width=3, height=2)
    resized = image.resize(3, 2, sig)
    assert len(resized) == 2

    v0, t0 = resized[0]
    v1, t1 = resized[1]
    assert t0 == 1000 and t1 == 2000
    assert v0.shape == (2, 3, 3)
    assert v1.shape == (2, 3, 3)
    assert v0.dtype == np.uint8 and v1.dtype == np.uint8
    # Uniform frames should remain uniform after resize
    assert np.unique(v0).tolist() == [10]
    assert np.unique(v1).tolist() == [200]
    assert resized.names == ['height', 'width', 'channel']
    assert resized.kind == Kind.IMAGE


def test_image_resize_with_pad_basic():
    # Frame narrower than target: expect horizontal padding with zeros
    h, w = 4, 2
    frame = np.full((h, w, 3), 255, dtype=np.uint8)  # white
    ts = [1000]
    sig = DummySignal(ts, [frame])

    resized = image.resize_with_pad(4, 4, sig)  # target H=W=4
    v, t = resized[0]
    assert t == 1000
    assert v.shape == (4, 4, 3)
    assert v.dtype == np.uint8
    # Left and right columns should be zeros (black padding), middle columns white
    left_col = v[:, 0, :]
    right_col = v[:, -1, :]
    mid = v[:, 1:-1, :]
    assert np.unique(left_col).tolist() == [0]
    assert np.unique(right_col).tolist() == [0]
    assert np.unique(mid).tolist() == [255]
    assert resized.names == ['height', 'width', 'channel']
    assert resized.kind == Kind.IMAGE
