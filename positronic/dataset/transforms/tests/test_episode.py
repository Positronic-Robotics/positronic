import numpy as np
import pytest

from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.transforms import Elementwise, TransformedEpisode
from positronic.dataset.transforms.episode import Concat, Derive, Group, Identity, Rename

from ...tests.utils import DummySignal, DummyTransform


@pytest.fixture
def sig_simple():
    ts = [1000, 2000, 3000, 4000, 5000]
    vals = [10, 20, 30, 40, 50]
    return DummySignal(ts, vals)


def test_transform_episode_keys_and_getitem_pass_through(sig_simple):
    # Build an episode with one signal 's' and two static fields
    ep = EpisodeContainer(data={'s': sig_simple, 'id': 7, 'note': 'ok'}, meta={'origin': 'unit'})
    tf = DummyTransform(operations={'a': ('s', lambda x: x * 10), 's': ('s', lambda x: x + 1)}, pass_through=True)
    te = TransformedEpisode(ep, tf)

    # Keys order: transform keys first, then original non-overlapping keys
    assert list(te.keys()) == ['a', 's', 'id', 'note']

    # __getitem__ should route to transform for transform keys
    a_vals = [v for v, _ in te['a']]
    s_vals = [v for v, _ in te['s']]
    assert a_vals == [x * 10 for x, _ in ep['s']]
    assert s_vals == [x + 1 for x, _ in ep['s']]

    # Pass-through static values
    assert te['id'] == 7
    assert te['note'] == 'ok'

    # Meta passthrough
    assert te.meta == {'origin': 'unit'}

    # Missing key raises
    with pytest.raises(KeyError):
        _ = te['missing']


def test_key_func_episode_transform(sig_simple):
    ep = EpisodeContainer(data={'s': sig_simple, 'id': 3})
    tf = Derive(
        double=lambda episode: Elementwise(episode['s'], lambda seq: np.asarray(seq) * 2),
        label=lambda episode: f'id={episode["id"]}',
    )

    transformed = tf(ep)
    assert [val for val, _ in transformed['double']] == [2 * val for val, _ in ep['s']]
    assert transformed['label'] == 'id=3'

    wrapped = TransformedEpisode(ep, tf)
    assert list(wrapped.keys()) == ['double', 'label']
    assert [val for val, _ in wrapped['double']] == [2 * val for val, _ in ep['s']]
    assert wrapped['label'] == 'id=3'


def test_transform_episode_pass_through_selected_keys(sig_simple):
    ep = EpisodeContainer(data={'s': sig_simple, 'id': 7, 'note': 'ok', 'skip': 'nope'})
    tf = DummyTransform(operations={'a': ('s', lambda x: x * 10), 's': ('s', lambda x: x + 1)}, pass_through=['note'])
    te = TransformedEpisode(ep, tf)

    assert list(te.keys()) == ['a', 's', 'note']
    assert [v for v, _ in te['a']] == [x * 10 for x, _ in ep['s']]
    assert te['note'] == 'ok'
    with pytest.raises(KeyError):
        _ = te['id']
    with pytest.raises(KeyError):
        _ = te['skip']


def test_transform_episode_no_pass_through(sig_simple):
    ep = EpisodeContainer(data={'s': sig_simple, 'id': 7})
    tf = DummyTransform(operations={'a': ('s', lambda x: x * 10), 's': ('s', lambda x: x + 1)}, pass_through=False)
    te = TransformedEpisode(ep, tf)

    # Only transform keys
    assert list(te.keys()) == ['a', 's']

    # Transform values present
    a_vals = [v for v, _ in te['a']]
    s_vals = [v for v, _ in te['s']]
    assert a_vals == [x * 10 for x, _ in ep['s']]
    assert s_vals == [x + 1 for x, _ in ep['s']]

    # Non-transform key should not be available
    with pytest.raises(KeyError):
        _ = te['id']


def test_transform_episode_multiple_transforms_order_and_precedence(sig_simple):
    ep = EpisodeContainer(data={'s': sig_simple, 'id': 42, 'z': 9})
    t1 = DummyTransform(operations={'a': ('s', lambda x: x * 10), 's': ('s', lambda x: x + 1)}, pass_through=True)
    t2 = DummyTransform(operations={'b': ('s', lambda x: -x), 's': ('s', lambda x: x + 100)}, pass_through=True)

    # Concatenate transforms to apply them in parallel; first transform takes precedence for duplicate keys
    concat_tf = Group(t1, t2)
    te = TransformedEpisode(ep, concat_tf)
    # Note: Key order reflects dict.update() behavior - later transforms' keys added first, then earlier
    assert set(te.keys()) == {'a', 's', 'b', 'id', 'z'}

    # 's' should come from the first transform (t1) - precedence is maintained
    s_vals = [v for v, _ in te['s']]
    assert s_vals == [x + 1 for x, _ in ep['s']]

    # Other transform keys are accessible
    a_vals = [v for v, _ in te['a']]
    b_vals = [v for v, _ in te['b']]
    assert a_vals == [x * 10 for x, _ in ep['s']]
    assert b_vals == [-(x) for x, _ in ep['s']]


def test_transform_episode_precedence_independent_of_access_order(sig_simple):
    """Verify that transform precedence is based on declaration order, not access order."""
    ep = EpisodeContainer(data={'s': sig_simple, 'id': 1})
    t1 = DummyTransform(operations={'a': ('s', lambda x: x * 10), 's': ('s', lambda x: x + 1)}, pass_through=False)
    t2 = DummyTransform(operations={'b': ('s', lambda x: -x), 's': ('s', lambda x: x + 100)}, pass_through=False)

    # Use Group for parallel application with precedence
    concat_tf = Group(t1, t2)
    te = TransformedEpisode(ep, concat_tf)

    # Access 'b' first (from t2), which should NOT affect 's' precedence
    b_vals = [v for v, _ in te['b']]
    assert b_vals == [-(x) for x, _ in ep['s']]

    # 's' should STILL come from t1 (first transform), not t2
    # Even though we accessed 'b' (from t2) first
    s_vals = [v for v, _ in te['s']]
    assert s_vals == [x + 1 for x, _ in ep['s']], "Key 's' should come from t1 regardless of access order"


def test_rename_transform(sig_simple):
    """Test Rename transform that renames keys in an episode."""
    ep = EpisodeContainer(data={'s': sig_simple, 'id': 42, 'note': 'test'})

    # Rename 's' to 'signal' and 'id' to 'identifier'
    # Mapping format: new_key -> old_key
    tf = Rename(mapping={'signal': 's', 'identifier': 'id'})
    renamed = tf(ep)

    # Only renamed keys should be present
    assert list(renamed.keys()) == ['signal', 'identifier']
    assert [v for v, _ in renamed['signal']] == [v for v, _ in ep['s']]
    assert renamed['identifier'] == 42

    # Original keys should not be accessible
    with pytest.raises(KeyError):
        _ = renamed['s']
    with pytest.raises(KeyError):
        _ = renamed['id']
    with pytest.raises(KeyError):
        _ = renamed['note']


def test_identity_transform(sig_simple):
    """Test Identity transform that selects specific features."""
    ep = EpisodeContainer(data={'s': sig_simple, 'id': 42, 'note': 'test', 'extra': 'skip'})

    # Select only 's' and 'id'
    tf = Identity('s', 'id')
    selected = tf(ep)

    assert set(selected.keys()) == {'s', 'id'}
    assert [v for v, _ in selected['s']] == [v for v, _ in ep['s']]
    assert selected['id'] == 42

    # Unselected keys should not be present
    with pytest.raises(KeyError):
        _ = selected['note']
    with pytest.raises(KeyError):
        _ = selected['extra']


def test_identity_transform_empty_returns_original(sig_simple):
    """Test Identity with no features returns original episode."""
    ep = EpisodeContainer(data={'s': sig_simple, 'id': 42})

    tf = Identity()
    result = tf(ep)

    # Should return the original episode unchanged
    assert result is ep


def test_concat_helper(sig_simple):
    """Test Concat helper that concatenates signals from an episode."""
    sig2 = DummySignal([1000, 2000, 3000, 4000, 5000], [100, 200, 300, 400, 500])
    ep = EpisodeContainer(data={'s1': sig_simple, 's2': sig2})

    # Concatenate s1 and s2
    concat_fn = Concat('s1', 's2')
    result = concat_fn(ep)

    # Should be a signal with concatenated arrays at each timestamp
    vals = [v for v, _ in result]
    assert len(vals) == 5
    # Each value should be an array of [s1_val, s2_val]
    assert np.array_equal(vals[0], np.array([10, 100]))
    assert np.array_equal(vals[1], np.array([20, 200]))
    assert np.array_equal(vals[4], np.array([50, 500]))


def test_concat_with_key_func_transform(sig_simple):
    """Test using Concat helper within Derive."""
    sig2 = DummySignal([1000, 2000, 3000, 4000, 5000], [100, 200, 300, 400, 500])
    ep = EpisodeContainer(data={'s1': sig_simple, 's2': sig2})

    # Use Concat to create a new concatenated signal
    tf = Derive(combined=Concat('s1', 's2'))
    transformed = tf(ep)

    assert 'combined' in transformed.keys()
    vals = [v for v, _ in transformed['combined']]
    assert len(vals) == 5
    # Each value should be an array of [s1_val, s2_val]
    assert np.array_equal(vals[0], np.array([10, 100]))
    assert np.array_equal(vals[2], np.array([30, 300]))
