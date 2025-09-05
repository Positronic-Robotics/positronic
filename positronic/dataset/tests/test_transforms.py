import pytest

from positronic.dataset.transforms import (
    Elementwise,
    TimeOffsets,
    Join,
    IndexOffsets,
    TransformEpisode,
    EpisodeTransform,
    Image,
    concat,
    pairwise,
)
from positronic.dataset.episode import EpisodeContainer
import numpy as np
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
    assert list(sig_simple) == [
        (10, 1000),
        (20, 2000),
        (30, 3000),
        (40, 4000),
        (50, 5000),
    ]


def test_join_relative_index_prev_next_equivalents(sig_simple):
    # [0, 1] (current and next)
    j_next = IndexOffsets(sig_simple, 0, 1)
    assert list(j_next) == [
        ((10, 20), 1000),
        ((20, 30), 2000),
        ((30, 40), 3000),
        ((40, 50), 4000),
    ]
    # [0, -1] (current and previous)
    j_prev = IndexOffsets(sig_simple, 0, -1)
    assert list(j_prev) == [
        ((20, 10), 2000),
        ((30, 20), 3000),
        ((40, 30), 4000),
        ((50, 40), 5000),
    ]


def test_join_relative_index_basic(sig_simple):
    # [0, 1] returns (v_i, v_{i+1}, t_i, t_{i+1})
    j = IndexOffsets(sig_simple, 0, 1)
    assert list(j) == [
        ((10, 20), 1000),
        ((20, 30), 2000),
        ((30, 40), 3000),
        ((40, 50), 4000),
    ]

    # [0, -1] returns (v_i, v_{i-1}, t_i, t_{i-1})
    jprev = IndexOffsets(sig_simple, 0, -1)
    assert list(jprev) == [
        ((20, 10), 2000),
        ((30, 20), 3000),
        ((40, 30), 4000),
        ((50, 40), 5000),
    ]

    # Single offset without ref timestamps -> bare values as first element in (value, ref_ts)
    j_single = IndexOffsets(sig_simple, 1)
    assert list(j_single) == [
        (20, 1000),
        (30, 2000),
        (40, 3000),
        (50, 4000),
    ]

    # Single offset with ref timestamps -> (value, ts_at_offset)
    j_single_ref = IndexOffsets(sig_simple, 1, include_ref_ts=True)
    assert list(j_single_ref) == [
        ((20, 2000), 1000),
        ((30, 3000), 2000),
        ((40, 4000), 3000),
        ((50, 5000), 4000),
    ]


def test_join_relative_index_errors_and_edges(sig_simple, empty_signal):
    # Non-empty relative indices required
    with pytest.raises(ValueError):
        IndexOffsets(sig_simple)
    # Empty signal yields empty
    assert list(IndexOffsets(empty_signal, 0, 1)) == []


def test_time_offsets_positive(sig_simple):
    to = TimeOffsets(sig_simple, 1000)
    # Positive delta: length preserved, last clamps to itself
    assert list(to) == [
        (20, 1000),
        (30, 2000),
        (40, 3000),
        (50, 4000),
        (50, 5000),
    ]
    # With include_ref_ts=True returns (value_at_shift, ts_at_shift)
    to_ref = TimeOffsets(sig_simple, 1000, include_ref_ts=True)
    assert list(to_ref) == [
        ((20, 2000), 1000),
        ((30, 3000), 2000),
        ((40, 4000), 3000),
        ((50, 5000), 4000),
        ((50, 5000), 5000),
    ]

    # Multiple deltas without ref timestamps -> tuple of values, with clamping at the end
    to_multi = TimeOffsets(sig_simple, -1000, 0, 1000)
    assert list(to_multi) == [
        ((10, 20, 30), 2000),
        ((20, 30, 40), 3000),
        ((30, 40, 50), 4000),
        ((40, 50, 50), 5000),
    ]

    # Multiple deltas with ref timestamps -> values then reference timestamps
    to_multi_ref = TimeOffsets(sig_simple, -1000, 0, 1000, include_ref_ts=True)
    assert list(to_multi_ref) == [
        ((10, 20, 30, 1000, 2000, 3000), 2000),
        ((20, 30, 40, 2000, 3000, 4000), 3000),
        ((30, 40, 50, 3000, 4000, 5000), 4000),
        ((40, 50, 50, 4000, 5000, 5000), 5000),
    ]


def test_time_offsets_negative(sig_simple):
    to = TimeOffsets(sig_simple, -1000)
    # Drops first; pairs each with previous at -1000
    assert list(to) == [
        (10, 2000),
        (20, 3000),
        (30, 4000),
        (40, 5000),
    ]


def test_time_offsets_zero(sig_simple):
    to = TimeOffsets(sig_simple, 0)
    assert list(to) == [
        (10, 1000),
        (20, 2000),
        (30, 3000),
        (40, 4000),
        (50, 5000),
    ]


def test_time_offsets_offset_rounding(sig_simple):
    # Delta that doesn't align with sampling: should carry back
    to = TimeOffsets(sig_simple, 2500)
    # All elements preserved; last clamps to itself
    assert list(to) == [
        (30, 1000),
        (40, 2000),
        (50, 3000),
        (50, 4000),
        (50, 5000),
    ]
    to_ref = TimeOffsets(sig_simple, 2500, include_ref_ts=True)
    assert list(to_ref) == [
        ((30, 3000), 1000),
        ((40, 4000), 2000),
        ((50, 5000), 3000),
        ((50, 5000), 4000),
        ((50, 5000), 5000),
    ]


def test_time_offsets_irregular_series():
    ts = [1000, 1300, 2000, 2700, 5000]
    vals = [1, 2, 3, 4, 5]
    sig = DummySignal(ts, vals)

    # Positive delta: floor carry-back across irregular gaps, full length
    to_pos = TimeOffsets(sig, 1000)
    assert list(to_pos) == [
        (3, 1000),  # 1000 -> 2000
        (3, 1300),  # 1300 -> 2300 -> 2000
        (4, 2000),  # 2000 -> 3000 -> 2700
        (4, 2700),  # 2700 -> 3700 -> 2700
        (5, 5000),  # 5000 -> 6000 -> 5000
    ]

    # Negative delta: match to earlier floors with varying diffs
    to_neg = TimeOffsets(sig, -1000)
    assert list(to_neg) == [
        (1, 2000),  # 2000 -> 1000
        (2, 2700),  # 2700 -> 1300
        (4, 5000),  # 5000 -> 2700
    ]

    # Zero delta: identity pairs with zero time difference
    to_zero = TimeOffsets(sig, 0)
    assert list(to_zero) == [
        (1, 1000),
        (2, 1300),
        (3, 2000),
        (4, 2700),
        (5, 5000),
    ]


def test_time_offsets_empty_signal(empty_signal):
    assert list(TimeOffsets(empty_signal, 0)) == []
    assert list(TimeOffsets(empty_signal, 123456)) == []
    assert list(TimeOffsets(empty_signal, -987654)) == []


def test_time_offsets_very_large_deltas(sig_simple):
    # Negative delta too large: no elements remain
    delta_empty = -(sig_simple.last_ts - sig_simple.start_ts + 1)
    to_empty = TimeOffsets(sig_simple, delta_empty)
    assert list(to_empty) == []


def test_time_offsets_positive_delta_too_large(sig_simple):
    # Positive delta strictly greater than span -> full length, pairs with last
    delta_too_large = (sig_simple.last_ts - sig_simple.start_ts) + 1
    to = TimeOffsets(sig_simple, delta_too_large)
    assert list(to) == [
        (50, 1000),
        (50, 2000),
        (50, 3000),
        (50, 4000),
        (50, 5000),
    ]


def test_join_basic():
    # s1: 1000..5000 step 1000
    ts1 = [1000, 2000, 3000, 4000, 5000]
    v1 = [10, 20, 30, 40, 50]
    s1 = DummySignal(ts1, v1)
    # s2: shifted by +500 and extended to 5500
    ts2 = [1500, 2500, 3500, 4500, 5500]
    v2 = [1, 2, 3, 4, 5]
    s2 = DummySignal(ts2, v2)

    jn = Join(s1, s2)
    assert list(jn) == [
        ((10, 1), 1500),
        ((20, 1), 2000),
        ((20, 2), 2500),
        ((30, 2), 3000),
        ((30, 3), 3500),
        ((40, 3), 4000),
        ((40, 4), 4500),
        ((50, 4), 5000),
        ((50, 5), 5500),
    ]


def test_join_basic_no_ref_timestamps():
    # s1: 1000..5000 step 1000
    ts1 = [1000, 2000, 3000, 4000, 5000]
    v1 = [10, 20, 30, 40, 50]
    s1 = DummySignal(ts1, v1)
    # s2: shifted by +500 and extended to 5500
    ts2 = [1500, 2500, 3500, 4500, 5500]
    v2 = [1, 2, 3, 4, 5]
    s2 = DummySignal(ts2, v2)

    jn = Join(s1, s2, include_ref_ts=False)
    assert list(jn) == [
        ((10, 1), 1500),
        ((20, 1), 2000),
        ((20, 2), 2500),
        ((30, 2), 3000),
        ((30, 3), 3500),
        ((40, 3), 4000),
        ((40, 4), 4500),
        ((50, 4), 5000),
        ((50, 5), 5500),
    ]


def test_join_equal_timestamps_drop_duplicates_default():
    # Overlapping timestamp at 2000; by default duplicates dropped -> single entry at 2000
    s1 = DummySignal([1000, 2000], [1, 2])
    s2 = DummySignal([2000, 3000], [10, 20])
    jn = Join(s1, s2)
    assert list(jn) == [
        ((2, 10), 2000),  # single entry at 2000
        ((2, 20), 3000),
    ]


def test_join_empty():
    empty = DummySignal([], [])
    s1 = DummySignal([1000], [1])
    assert list(Join(empty, empty)) == []
    assert list(Join(s1, empty)) == []
    assert list(Join(empty, s1)) == []


def test_join_three_signals_basic():
    # s1: 1000,2000,3000
    s1 = DummySignal([1000, 2000, 3000], [10, 20, 30])
    # s2: shifted by +500 and extends beyond
    s2 = DummySignal([1500, 2500, 3500], [1, 2, 3])
    # s3: shifted by +200 from s1 and extends further
    s3 = DummySignal([1200, 2200, 3200, 4200], [100, 200, 300, 400])

    jn = Join(s1, s2, s3)
    # Start from max starts (1500), union of timestamps with carry-back
    assert list(jn) == [
        ((10, 1, 100), 1500),
        ((20, 1, 100), 2000),
        ((20, 1, 200), 2200),
        ((20, 2, 200), 2500),
        ((30, 2, 200), 3000),
        ((30, 2, 300), 3200),
        ((30, 3, 300), 3500),
        ((30, 3, 400), 4200),
    ]


def test_join_three_signals_no_ref_timestamps():
    # s1: 1000,2000,3000
    s1 = DummySignal([1000, 2000, 3000], [10, 20, 30])
    # s2: shifted by +500 and extends beyond
    s2 = DummySignal([1500, 2500, 3500], [1, 2, 3])
    # s3: shifted by +200 from s1 and extends further
    s3 = DummySignal([1200, 2200, 3200, 4200], [100, 200, 300, 400])

    jn = Join(s1, s2, s3, include_ref_ts=False)
    assert list(jn) == [
        ((10, 1, 100), 1500),
        ((20, 1, 100), 2000),
        ((20, 1, 200), 2200),
        ((20, 2, 200), 2500),
        ((30, 2, 200), 3000),
        ((30, 2, 300), 3200),
        ((30, 3, 300), 3500),
        ((30, 3, 400), 4200),
    ]


def test_pairwise_basic_and_alignment():
    # Two scalar signals with different timestamps
    s1 = DummySignal([1000, 2000, 3000, 4000], [1, 2, 3, 4])
    s2 = DummySignal([1500, 2500, 3500], [10, 20, 30])

    add_sig = pairwise(s1, s2, np.add)
    assert list(add_sig) == [
        (11, 1500),  # 1 + 10
        (12, 2000),  # 2 + 10
        (22, 2500),  # 2 + 20
        (23, 3000),  # 3 + 20
        (33, 3500),  # 3 + 30
        (34, 4000),  # 4 + 30
    ]


def test_pairwise_vectors_and_subtract():
    s1 = DummySignal([1000, 2000, 3000], [[1, 2], [3, 4], [5, 6]])
    s2 = DummySignal([1500, 2500], [[10, 20], [30, 40]])

    sub = pairwise(s1, s2, np.subtract)
    assert [(v.tolist(), t) for v, t in sub] == [
        ([1 - 10, 2 - 20], 1500),
        ([3 - 10, 4 - 20], 2000),
        ([3 - 30, 4 - 40], 2500),
        ([5 - 30, 6 - 40], 3000),
    ]


class _DummyTransform(EpisodeTransform):

    def __init__(self):
        self._keys = ["a", "s"]

    @property
    def keys(self):
        return list(self._keys)

    def transform(self, name: str, episode):
        if name == "a":
            # 10x the base signal
            base = episode["s"]
            return Elementwise(base, lambda seq: np.asarray(seq) * 10)
        if name == "s":
            # Override original 's' by adding 1
            base = episode["s"]
            return Elementwise(base, lambda seq: np.asarray(seq) + 1)
        raise KeyError(name)


def test_transform_episode_keys_and_getitem_pass_through(sig_simple):
    # Build an episode with one signal 's' and two static fields
    ep = EpisodeContainer(signals={"s": sig_simple}, static={"id": 7, "note": "ok"}, meta={"origin": "unit"})
    tf = _DummyTransform()
    te = TransformEpisode(ep, tf, pass_through=True)

    # Keys order: transform keys first, then original non-overlapping keys
    assert list(te.keys) == ["a", "s", "id", "note"]

    # __getitem__ should route to transform for transform keys
    a_vals = [v for v, _ in te["a"]]
    s_vals = [v for v, _ in te["s"]]
    assert a_vals == [x * 10 for x, _ in ep["s"]]
    assert s_vals == [x + 1 for x, _ in ep["s"]]

    # Pass-through static values
    assert te["id"] == 7
    assert te["note"] == "ok"

    # Meta passthrough
    assert te.meta == {"origin": "unit"}

    # Missing key raises
    with pytest.raises(KeyError):
        _ = te["missing"]


def test_transform_episode_no_pass_through(sig_simple):
    ep = EpisodeContainer(signals={"s": sig_simple}, static={"id": 7})
    tf = _DummyTransform()
    te = TransformEpisode(ep, tf, pass_through=False)

    # Only transform keys
    assert list(te.keys) == ["a", "s"]

    # Transform values present
    a_vals = [v for v, _ in te["a"]]
    s_vals = [v for v, _ in te["s"]]
    assert a_vals == [x * 10 for x, _ in ep["s"]]
    assert s_vals == [x + 1 for x, _ in ep["s"]]

    # Non-transform key should not be available
    with pytest.raises(KeyError):
        _ = te["id"]


class _DummyTransform2(EpisodeTransform):

    @property
    def keys(self):
        return ["b", "s"]

    def transform(self, name: str, episode):
        if name == "b":
            base = episode["s"]
            return Elementwise(base, lambda seq: np.asarray(seq) * -1)
        if name == "s":
            base = episode["s"]
            return Elementwise(base, lambda seq: np.asarray(seq) + 100)
        raise KeyError(name)


def test_transform_episode_multiple_transforms_order_and_precedence(sig_simple):
    ep = EpisodeContainer(signals={"s": sig_simple}, static={"id": 42, "z": 9})
    t1 = _DummyTransform()  # defines ["a", "s"] (s -> +1)
    t2 = _DummyTransform2()  # defines ["b", "s"] (s -> +100)

    # Concatenate transform keys in order; first occurrence of duplicates kept
    te = TransformEpisode(ep, t1, t2, pass_through=True)
    assert list(te.keys) == ["a", "s", "b", "id", "z"]

    # 's' should come from the first transform (t1)
    s_vals = [v for v, _ in te["s"]]
    assert s_vals == [x + 1 for x, _ in ep["s"]]

    # Other transform keys are accessible
    a_vals = [v for v, _ in te["a"]]
    b_vals = [v for v, _ in te["b"]]
    assert a_vals == [x * 10 for x, _ in ep["s"]]
    assert b_vals == [-(x) for x, _ in ep["s"]]


def test_image_resize_basic():
    # Create a simple image signal with uniform frames
    h, w = 4, 6
    frame1 = np.full((h, w, 3), 10, dtype=np.uint8)
    frame2 = np.full((h, w, 3), 200, dtype=np.uint8)
    ts = [1000, 2000]
    sig = DummySignal(ts, [frame1, frame2])

    # Resize to (width=3, height=2)
    resized = Image.resize(3, 2, sig)
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


def test_image_resize_with_pad_basic():
    # Frame narrower than target: expect horizontal padding with zeros
    h, w = 4, 2
    frame = np.full((h, w, 3), 255, dtype=np.uint8)  # white
    ts = [1000]
    sig = DummySignal(ts, [frame])

    resized = Image.resize_with_pad(4, 4, sig)  # target H=W=4
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


def test_concat_vectors_batched_and_alignment():
    # Two vector signals with different timestamps
    ts1 = [1000, 2000, 3000]
    v1 = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    s1 = DummySignal(ts1, v1)

    ts2 = [1500, 2500]
    v2 = [
        [10],
        [20],
    ]
    s2 = DummySignal(ts2, v2)
    cat = concat(s1, s2)

    # Expected union starting from max start (1500): [1500, 2000, 2500, 3000]
    # Values are concatenated vectors with carry-back
    expected_rows = np.array([
        [1, 2, 10],
        [3, 4, 10],
        [3, 4, 20],
        [5, 6, 20],
    ])
    expected_ts = [1500, 2000, 2500, 3000]

    # Batched request should return a single 2D array
    rows = cat._values_at(np.arange(len(cat)))  # internal call to validate batching shape
    assert isinstance(rows, np.ndarray)
    assert rows.shape == expected_rows.shape
    assert np.array_equal(rows, expected_rows)

    # Iteration path (one-by-one) should match rows
    vals = [v for v, _ in cat]
    ts = [t for _, t in cat]
    assert np.array_equal(np.stack(vals, axis=0), expected_rows)
    assert ts == expected_ts
