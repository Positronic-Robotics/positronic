import numpy as np
import pytest

from positronic import geom
from positronic.dataset.transforms import (
    Elementwise,
    IndexOffsets,
    Join,
    TimeOffsets,
    concat,
    pairwise,
    recode_rotation,
    recode_transform,
    view,
)
from positronic.geom import Rotation

from ...tests.utils import DummySignal


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


@pytest.fixture
def vector_signal():
    ts = [1000, 2000, 3000]
    vals = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        np.array([7.0, 8.0, 9.0, 10.0], dtype=np.float32),
    ]
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


def test_view_slice_returns_lazy_subset(vector_signal):
    sliced = view(vector_signal, slice(1, 3))

    base_samples = list(vector_signal)
    result_samples = list(sliced)

    assert len(result_samples) == len(base_samples)
    for (vals, ts), (base_vals, base_ts) in zip(result_samples, base_samples, strict=False):
        np.testing.assert_array_equal(vals, base_vals[1:3])
        assert ts == base_ts


def test_recode_rotation_quat_to_rotvec_with_slice():
    ts = [1000, 2000]
    # Each record stores quaternion followed by an auxiliary logit.
    values = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0, 0.2], dtype=np.float32),
    ]
    signal = DummySignal(ts, values)

    recoded = recode_rotation(Rotation.Representation.QUAT, Rotation.Representation.ROTVEC, signal, slice=slice(0, 4))

    samples = list(recoded)
    assert len(samples) == 2

    expected_rotvecs = [
        geom.Rotation.create_from([1.0, 0.0, 0.0, 0.0], Rotation.Representation.QUAT).to(
            Rotation.Representation.ROTVEC
        ),
        geom.Rotation.create_from([0.0, 1.0, 0.0, 0.0], Rotation.Representation.QUAT).to(
            Rotation.Representation.ROTVEC
        ),
    ]

    for (vals, ts_out), expected, ts_in in zip(samples, expected_rotvecs, ts, strict=False):
        np.testing.assert_allclose(vals, expected)
        assert ts_out == ts_in

    recoded_quat = recode_rotation(
        Rotation.Representation.QUAT, Rotation.Representation.QUAT, signal, slice=slice(0, 4)
    )
    quat_samples = list(recoded_quat)
    assert len(quat_samples) == 2
    np.testing.assert_allclose(quat_samples[0][0], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(quat_samples[1][0], np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_recode_transform_rotation_matrix_to_quat():
    transforms = [
        geom.Transform3D(
            translation=np.array([0.1, -0.2, 0.3]), rotation=Rotation.from_euler(np.array([0.1, 0.2, -0.3]))
        ),
        geom.Transform3D(
            translation=np.array([-0.4, 0.5, 0.6]), rotation=Rotation.from_euler(np.array([-0.2, 0.4, 0.1]))
        ),
    ]
    ts = [1000, 2000]
    vals = [t.as_vector(Rotation.Representation.ROTATION_MATRIX) for t in transforms]
    signal = DummySignal(ts, vals)

    recoded = recode_transform(Rotation.Representation.ROTATION_MATRIX, Rotation.Representation.QUAT, signal)
    samples = list(recoded)

    assert len(samples) == len(transforms)
    for (vec, ts_out), transform, ts_in in zip(samples, transforms, ts, strict=False):
        expected = transform.as_vector(Rotation.Representation.QUAT)
        np.testing.assert_allclose(vec, expected)
        assert ts_out == ts_in


def test_recode_transform_identity_returns_original_signal():
    ts = [5000]
    vals = [geom.Transform3D.identity.as_vector(Rotation.Representation.QUAT)]
    signal = DummySignal(ts, vals)

    recoded = recode_transform(Rotation.Representation.QUAT, Rotation.Representation.QUAT, signal)

    assert recoded is signal


def test_join_relative_index_prev_next_equivalents(sig_simple):
    # [0, 1] (current and next)
    j_next = IndexOffsets(sig_simple, 0, 1)
    assert list(j_next) == [((10, 20), 1000), ((20, 30), 2000), ((30, 40), 3000), ((40, 50), 4000)]
    # [0, -1] (current and previous)
    j_prev = IndexOffsets(sig_simple, 0, -1)
    assert list(j_prev) == [((20, 10), 2000), ((30, 20), 3000), ((40, 30), 4000), ((50, 40), 5000)]


def test_join_relative_index_basic(sig_simple):
    # [0, 1] returns (v_i, v_{i+1}, t_i, t_{i+1})
    j = IndexOffsets(sig_simple, 0, 1)
    assert list(j) == [((10, 20), 1000), ((20, 30), 2000), ((30, 40), 3000), ((40, 50), 4000)]

    # [0, -1] returns (v_i, v_{i-1}, t_i, t_{i-1})
    jprev = IndexOffsets(sig_simple, 0, -1)
    assert list(jprev) == [((20, 10), 2000), ((30, 20), 3000), ((40, 30), 4000), ((50, 40), 5000)]

    # Single offset without ref timestamps -> bare values as first element in (value, ref_ts)
    j_single = IndexOffsets(sig_simple, 1)
    assert list(j_single) == [(20, 1000), (30, 2000), (40, 3000), (50, 4000)]

    # Single offset with ref timestamps -> (value, ts_at_offset)
    j_single_ref = IndexOffsets(sig_simple, 1, include_ref_ts=True)
    assert list(j_single_ref) == [((20, 2000), 1000), ((30, 3000), 2000), ((40, 4000), 3000), ((50, 5000), 4000)]


def test_join_relative_index_errors_and_edges(sig_simple, empty_signal):
    # Non-empty relative indices required
    with pytest.raises(ValueError):
        IndexOffsets(sig_simple)
    # Empty signal yields empty
    assert list(IndexOffsets(empty_signal, 0, 1)) == []


def test_time_offsets_positive(sig_simple):
    to = TimeOffsets(sig_simple, 1000)
    # Positive delta: length preserved, last clamps to itself
    assert list(to) == [(20, 1000), (30, 2000), (40, 3000), (50, 4000), (50, 5000)]
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
    assert list(to_multi) == [((10, 20, 30), 2000), ((20, 30, 40), 3000), ((30, 40, 50), 4000), ((40, 50, 50), 5000)]

    # Multiple deltas with ref timestamps -> (values_tuple, np.array(ref_timestamps))
    to_multi_ref = TimeOffsets(sig_simple, -1000, 0, 1000, include_ref_ts=True)
    actual = list(to_multi_ref)
    exp_vals = [(10, 20, 30), (20, 30, 40), (30, 40, 50), (40, 50, 50)]
    exp_ts_arrays = [
        np.array([1000, 2000, 3000], dtype=np.int64),
        np.array([2000, 3000, 4000], dtype=np.int64),
        np.array([3000, 4000, 5000], dtype=np.int64),
        np.array([4000, 5000, 5000], dtype=np.int64),
    ]
    exp_ref_ts = [2000, 3000, 4000, 5000]
    assert len(actual) == 4
    for i, ((vals, ts_arr), ref_ts) in enumerate(actual):
        assert tuple(vals) == exp_vals[i]
        assert np.array_equal(ts_arr, exp_ts_arrays[i])
        assert ref_ts == exp_ref_ts[i]


def test_time_offsets_negative(sig_simple):
    to = TimeOffsets(sig_simple, -1000)
    # Drops first; pairs each with previous at -1000
    assert list(to) == [(10, 2000), (20, 3000), (30, 4000), (40, 5000)]


def test_time_offsets_zero(sig_simple):
    to = TimeOffsets(sig_simple, 0)
    assert list(to) == [(10, 1000), (20, 2000), (30, 3000), (40, 4000), (50, 5000)]


def test_time_offsets_offset_rounding(sig_simple):
    # Delta that doesn't align with sampling: should carry back
    to = TimeOffsets(sig_simple, 2500)
    # All elements preserved; last clamps to itself
    assert list(to) == [(30, 1000), (40, 2000), (50, 3000), (50, 4000), (50, 5000)]
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
    assert list(to_zero) == [(1, 1000), (2, 1300), (3, 2000), (4, 2700), (5, 5000)]


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
    assert list(to) == [(50, 1000), (50, 2000), (50, 3000), (50, 4000), (50, 5000)]


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


def test_index_offsets_multiple_with_ref_timestamps(sig_simple):
    # Offsets [-1, 0, 1] should return grouped timestamps as np.ndarray[int64]
    j = IndexOffsets(sig_simple, -1, 0, 1, include_ref_ts=True)
    actual = list(j)
    exp_vals = [(10, 20, 30), (20, 30, 40), (30, 40, 50)]
    exp_ts_arrays = [
        np.array([1000, 2000, 3000], dtype=np.int64),
        np.array([2000, 3000, 4000], dtype=np.int64),
        np.array([3000, 4000, 5000], dtype=np.int64),
    ]
    exp_ref_ts = [2000, 3000, 4000]
    assert len(actual) == 3
    for i, ((vals, ts_arr), ref_ts) in enumerate(actual):
        assert tuple(vals) == exp_vals[i]
        assert np.array_equal(ts_arr, exp_ts_arrays[i])
        assert ref_ts == exp_ref_ts[i]


def test_join_with_ref_timestamps_grouped():
    # s1: 1000..5000 step 1000
    ts1 = [1000, 2000, 3000, 4000, 5000]
    v1 = [10, 20, 30, 40, 50]
    s1 = DummySignal(ts1, v1)
    # s2: shifted by +500 and extended to 5500
    ts2 = [1500, 2500, 3500, 4500, 5500]
    v2 = [1, 2, 3, 4, 5]
    s2 = DummySignal(ts2, v2)

    jn = Join(s1, s2, include_ref_ts=True)
    actual = list(jn)
    exp_vals = [(10, 1), (20, 1), (20, 2), (30, 2), (30, 3), (40, 3), (40, 4), (50, 4), (50, 5)]
    exp_ts_arrays = [
        np.array([1000, 1500], dtype=np.int64),
        np.array([2000, 1500], dtype=np.int64),
        np.array([2000, 2500], dtype=np.int64),
        np.array([3000, 2500], dtype=np.int64),
        np.array([3000, 3500], dtype=np.int64),
        np.array([4000, 3500], dtype=np.int64),
        np.array([4000, 4500], dtype=np.int64),
        np.array([5000, 4500], dtype=np.int64),
        np.array([5000, 5500], dtype=np.int64),
    ]
    exp_union_ts = [1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
    assert len(actual) == len(exp_vals)
    for i, ((vals, ts_arr), ref_ts) in enumerate(actual):
        assert tuple(vals) == exp_vals[i]
        assert np.array_equal(ts_arr, exp_ts_arrays[i])
        assert ref_ts == exp_union_ts[i]


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


def test_concat_vectors_batched_and_alignment():
    # Two vector signals with different timestamps
    ts1 = [1000, 2000, 3000]
    v1 = [[1, 2], [3, 4], [5, 6]]
    s1 = DummySignal(ts1, v1)

    ts2 = [1500, 2500]
    v2 = [[10], [20]]
    s2 = DummySignal(ts2, v2)
    cat = concat(s1, s2)

    # Expected union starting from max start (1500): [1500, 2000, 2500, 3000]
    # Values are concatenated vectors with carry-back
    expected_rows = np.array([[1, 2, 10], [3, 4, 10], [3, 4, 20], [5, 6, 20]])
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
