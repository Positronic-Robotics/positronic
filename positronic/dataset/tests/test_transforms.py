import numpy as np
import pytest

from positronic.dataset.transforms import Elementwise, Previous, Next, JoinDeltaTime
from .utils import DummySignal


@pytest.fixture
def sig_simple():
    ts = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
    vals = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    return DummySignal(ts, vals)


@pytest.fixture
def empty_signal():
    ts = np.array([], dtype=np.int64)
    vals = np.array([], dtype=np.int64)
    return DummySignal(ts, vals)


def _times10(x):
    # Supports scalar int, numpy arrays and python lists
    if isinstance(x, np.ndarray):
        return x * 10
    if isinstance(x, list):
        return [v * 10 for v in x]
    return x * 10


def test_elementwise(sig_simple):
    ew = Elementwise(sig_simple, _times10)
    # Length preserved; values transformed; timestamps unchanged
    assert list(ew) == [(100, 1000), (200, 2000), (300, 3000), (400, 4000), (500, 5000)]
    # Underlying signal remains unchanged
    assert list(sig_simple) == [(10, 1000), (20, 2000), (30, 3000), (40, 4000), (50, 5000)]


def test_previous(sig_simple):
    pv = Previous(sig_simple)
    assert list(pv) == [((20, 10, 1000), 2000), ((30, 20, 1000), 3000), ((40, 30, 1000), 4000), ((50, 40, 1000), 5000)]
    # Underlying signal remains unchanged
    assert list(sig_simple) == [(10, 1000), (20, 2000), (30, 3000), (40, 4000), (50, 5000)]


def test_empty_prev(empty_signal):
    assert list(Previous(empty_signal)) == []


def test_next(sig_simple):
    pv = Next(sig_simple)
    assert list(pv) == [((10, 20, 1000), 1000), ((20, 30, 1000), 2000), ((30, 40, 1000), 3000), ((40, 50, 1000), 4000)]
    # Underlying signal remains unchanged
    assert list(sig_simple) == [(10, 1000), (20, 2000), (30, 3000), (40, 4000), (50, 5000)]


def test_empty_next(empty_signal):
    assert list(Next(empty_signal)) == []


def test_join_delta_time_positive(sig_simple):
    jdt = JoinDeltaTime(sig_simple, 1000)
    # With positive delta, length preserved; last pairs with itself
    assert list(jdt) == [
        ((10, 20, 1000), 1000),
        ((20, 30, 1000), 2000),
        ((30, 40, 1000), 3000),
        ((40, 50, 1000), 4000),
        ((50, 50, 0), 5000),
    ]


def test_join_delta_time_negative(sig_simple):
    jdt = JoinDeltaTime(sig_simple, -1000)
    # Should match Previous semantics on this evenly spaced signal
    assert list(jdt) == list(Previous(sig_simple))


def test_join_delta_time_zero(sig_simple):
    jdt = JoinDeltaTime(sig_simple, 0)
    assert list(jdt) == [((10, 10, 0), 1000), ((20, 20, 0), 2000), ((30, 30, 0), 3000), ((40, 40, 0), 4000),
                         ((50, 50, 0), 5000)]


def test_join_delta_time_offset_rounding(sig_simple):
    # Delta that doesn't align with sampling: should carry back
    jdt = JoinDeltaTime(sig_simple, 2500)
    # All elements preserved; last pairs with itself, larger jump for early ones
    assert list(jdt) == [
        ((10, 30, 2000), 1000),
        ((20, 40, 2000), 2000),
        ((30, 50, 2000), 3000),
        ((40, 50, 1000), 4000),
        ((50, 50, 0), 5000),
    ]


def test_join_delta_time_irregular_series():
    ts = np.array([1000, 1300, 2000, 2700, 5000], dtype=np.int64)
    vals = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    sig = DummySignal(ts, vals)

    # Positive delta: floor carry-back across irregular gaps, full length
    jdt_pos = JoinDeltaTime(sig, 1000)
    assert list(jdt_pos) == [
        ((1, 3, 1000), 1000),  # 1000 -> 2000
        ((2, 3, 700), 1300),  # 1300 -> 2300 -> 2000
        ((3, 4, 700), 2000),  # 2000 -> 3000 -> 2700
        ((4, 4, 0), 2700),  # 2700 -> 3700 -> 2700
        ((5, 5, 0), 5000),  # 5000 -> 6000 -> 5000
    ]

    # Negative delta: match to earlier floors with varying diffs
    jdt_neg = JoinDeltaTime(sig, -1000)
    assert list(jdt_neg) == [
        ((3, 1, 1000), 2000),  # 2000 -> 1000
        ((4, 2, 1400), 2700),  # 2700 -> 1300
        ((5, 4, 2300), 5000),  # 5000 -> 2700
    ]

    # Zero delta: identity pairs with zero time difference
    jdt_zero = JoinDeltaTime(sig, 0)
    assert list(jdt_zero) == [
        ((1, 1, 0), 1000),
        ((2, 2, 0), 1300),
        ((3, 3, 0), 2000),
        ((4, 4, 0), 2700),
        ((5, 5, 0), 5000),
    ]


def test_join_delta_time_empty_signal(empty_signal):
    assert list(JoinDeltaTime(empty_signal, 0)) == []
    assert list(JoinDeltaTime(empty_signal, 123456)) == []
    assert list(JoinDeltaTime(empty_signal, -987654)) == []


def test_join_delta_time_very_large_deltas(sig_simple):
    # Negative delta too large: no elements remain
    delta_empty = -(sig_simple.last_ts - sig_simple.start_ts + 1)
    jdt_empty = JoinDeltaTime(sig_simple, delta_empty)
    assert list(jdt_empty) == []


def test_join_delta_time_positive_delta_too_large(sig_simple):
    # Positive delta strictly greater than span -> full length, pairs with last
    delta_too_large = (sig_simple.last_ts - sig_simple.start_ts) + 1
    jdt = JoinDeltaTime(sig_simple, delta_too_large)
    assert list(jdt) == [
        ((10, 50, 4000), 1000),
        ((20, 50, 3000), 2000),
        ((30, 50, 2000), 3000),
        ((40, 50, 1000), 4000),
        ((50, 50, 0), 5000),
    ]
