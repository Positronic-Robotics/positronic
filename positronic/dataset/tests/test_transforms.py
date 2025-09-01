import pytest

from positronic.dataset.transforms import Elementwise, JoinDeltaTime, Interleave, IndexOffsets
from .utils import DummySignal


@pytest.fixture
def sig_simple():
    ts = [1000, 2000, 3000, 4000, 5000]
    vals = [10, 20, 30, 40, 50]
    return DummySignal(ts, vals)


@pytest.fixture
def empty_signal():
    ts = []
    vals = []
    return DummySignal(ts, vals)


def _times10(x):
    if isinstance(x, list):
        return [v * 10 for v in x]
    return x * 10


def test_elementwise(sig_simple):
    ew = Elementwise(sig_simple, _times10)
    # Length preserved; values transformed; timestamps unchanged
    assert list(ew) == [(100, 1000), (200, 2000), (300, 3000), (400, 4000), (500, 5000)]
    # Underlying signal remains unchanged
    assert list(sig_simple) == [(10, 1000), (20, 2000), (30, 3000), (40, 4000), (50, 5000)]


def test_join_relative_index_prev_next_equivalents(sig_simple):
    # [0, 1] (current and next)
    j_next = IndexOffsets(sig_simple, [0, 1])
    assert list(j_next) == [
        ((10, 20, 1000, 2000), 1000),
        ((20, 30, 2000, 3000), 2000),
        ((30, 40, 3000, 4000), 3000),
        ((40, 50, 4000, 5000), 4000),
    ]
    # [0, -1] (current and previous)
    j_prev = IndexOffsets(sig_simple, [0, -1])
    assert list(j_prev) == [
        ((20, 10, 2000, 1000), 2000),
        ((30, 20, 3000, 2000), 3000),
        ((40, 30, 4000, 3000), 4000),
        ((50, 40, 5000, 4000), 5000),
    ]


def test_join_relative_index_basic(sig_simple):
    # [0, 1] returns (v_i, v_{i+1}, t_i, t_{i+1})
    j = IndexOffsets(sig_simple, [0, 1])
    assert list(j) == [
        ((10, 20, 1000, 2000), 1000),
        ((20, 30, 2000, 3000), 2000),
        ((30, 40, 3000, 4000), 3000),
        ((40, 50, 4000, 5000), 4000),
    ]

    # [0, -1] returns (v_i, v_{i-1}, t_i, t_{i-1})
    jprev = IndexOffsets(sig_simple, [0, -1])
    assert list(jprev) == [
        ((20, 10, 2000, 1000), 2000),
        ((30, 20, 3000, 2000), 3000),
        ((40, 30, 4000, 3000), 4000),
        ((50, 40, 5000, 4000), 5000),
    ]


def test_join_relative_index_errors_and_edges(sig_simple, empty_signal):
    # Non-empty relative indices required
    with pytest.raises(ValueError):
        IndexOffsets(sig_simple, [])
    # Empty signal yields empty
    assert list(IndexOffsets(empty_signal, [0, 1])) == []


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
    assert list(jdt) == [
        ((20, 10, 1000), 2000),
        ((30, 20, 1000), 3000),
        ((40, 30, 1000), 4000),
        ((50, 40, 1000), 5000),
    ]


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
    ts = [1000, 1300, 2000, 2700, 5000]
    vals = [1, 2, 3, 4, 5]
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


def test_interleave_basic():
    # s1: 1000..5000 step 1000
    ts1 = [1000, 2000, 3000, 4000, 5000]
    v1 = [10, 20, 30, 40, 50]
    s1 = DummySignal(ts1, v1)
    # s2: shifted by +500 and extended to 5500
    ts2 = [1500, 2500, 3500, 4500, 5500]
    v2 = [1, 2, 3, 4, 5]
    s2 = DummySignal(ts2, v2)

    il = Interleave(s1, s2)
    assert list(il) == [
        ((10, 1, 500), 1500),  # t2_ref 1500 - t1_ref 1000 = +500
        ((20, 1, -500), 2000),  # 1500 - 2000 = -500
        ((20, 2, 500), 2500),  # 2500 - 2000 = +500
        ((30, 2, -500), 3000),  # 2500 - 3000 = -500
        ((30, 3, 500), 3500),  # 3500 - 3000 = +500
        ((40, 3, -500), 4000),  # 3500 - 4000 = -500
        ((40, 4, 500), 4500),  # 4500 - 4000 = +500
        ((50, 4, -500), 5000),  # 4500 - 5000 = -500
        ((50, 5, 500), 5500),  # 5500 - 5000 = +500
    ]


def test_interleave_equal_timestamps_drop_duplicates_default():
    # Overlapping timestamp at 2000; by default duplicates dropped -> single entry at 2000
    s1 = DummySignal([1000, 2000], [1, 2])
    s2 = DummySignal([2000, 3000], [10, 20])
    il = Interleave(s1, s2)  # drop_duplicates=True by default
    assert list(il) == [
        ((2, 10, 0), 2000),  # single entry at 2000
        ((2, 20, 1000), 3000),
    ]


def test_interleave_equal_timestamps_keep_duplicates():
    # With drop_duplicates=False, both entries at 2000 are kept
    s1 = DummySignal([1000, 2000], [1, 2])
    s2 = DummySignal([2000, 3000], [10, 20])
    il = Interleave(s1, s2, drop_duplicates=False)
    assert list(il) == [
        ((2, 10, 0), 2000),  # s1's 2000, s2 at 2000
        ((2, 10, 0), 2000),  # s2's 2000, s1 carried at 2000
        ((2, 20, 1000), 3000),
    ]


def test_interleave_empty():
    empty = DummySignal([], [])
    s1 = DummySignal([1000], [1])
    assert list(Interleave(empty, empty)) == []
    assert list(Interleave(s1, empty)) == []
    assert list(Interleave(empty, s1)) == []


def test_interleave_keep_duplicates_nonoverlap_s1_before_s2():
    # s1 entirely before s2; start at s2.start_ts; union contains only s2 timestamps
    s1 = DummySignal([1000, 1500], [1, 2])
    s2 = DummySignal([3000, 4000], [10, 20])
    il = Interleave(s1, s2, drop_duplicates=False)
    assert list(il) == [
        ((2, 10, 1500), 3000),
        ((2, 20, 2500), 4000),
    ]


def test_interleave_keep_duplicates_nonoverlap_s2_before_s1():
    # s2 entirely before s1; start at s1.start_ts; union contains only s1 timestamps
    s1 = DummySignal([3000, 4000], [30, 40])
    s2 = DummySignal([1000, 1500], [1, 2])
    il = Interleave(s1, s2, drop_duplicates=False)
    assert list(il) == [
        ((30, 2, -1500), 3000),
        ((40, 2, -2500), 4000),
    ]


def test_interleave_keep_duplicates_random_indexing():
    # Ensure k-th selection works for non-consecutive indices
    s1 = DummySignal([1000, 3000, 5000], [1, 3, 5])
    s2 = DummySignal([2000, 4000, 6000], [2, 4, 6])
    il = Interleave(s1, s2, drop_duplicates=False)
    # Pick indices 0,2,4 -> timestamps 2000, 4000, 6000
    sub = il[[0, 2, 4]]
    assert list(sub) == [
        ((1, 2, 1000), 2000),
        ((3, 4, 1000), 4000),
        ((5, 6, 1000), 6000),
    ]
