import numpy as np
import pytest

from positronic.dataset.transforms import Elementwise
from .utils import DummySignal


@pytest.fixture
def sig_simple():
    ts = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
    vals = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    return DummySignal(ts, vals)


def _times10(x):
    # Supports scalar int, numpy arrays and python lists
    if isinstance(x, np.ndarray):
        return x * 10
    if isinstance(x, list):
        return [v * 10 for v in x]
    return x * 10


class TestElementwiseCoreBehavior:

    def test_length_and_value_transform(self, sig_simple):
        ew = Elementwise(sig_simple, _times10)
        # Length preserved; values transformed; timestamps unchanged
        assert len(ew) == len(sig_simple)
        assert ew[0] == (100, 1000)
        assert ew[2] == (300, 3000)
        # Underlying signal remains unchanged
        assert sig_simple[0] == (10, 1000)
        assert sig_simple[2] == (30, 3000)

    def test_batch_application_and_timestamps_preserved(self, sig_simple):
        ew = Elementwise(sig_simple, _times10)
        # Verify batch access applies transform elementwise and preserves timestamps
        view = ew[1:4]
        assert list(view) == [(200, 2000), (300, 3000), (400, 4000)]
