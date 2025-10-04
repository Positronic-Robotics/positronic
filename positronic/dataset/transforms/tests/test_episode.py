import numpy as np
import pytest

from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.transforms import (
    Elementwise,
    EpisodeTransform,
    KeyFuncEpisodeTransform,
    TransformedEpisode,
)

from ...tests.utils import DummySignal


@pytest.fixture
def sig_simple():
    ts = [1000, 2000, 3000, 4000, 5000]
    vals = [10, 20, 30, 40, 50]
    return DummySignal(ts, vals)


class _DummyTransform(EpisodeTransform):
    def __init__(self):
        self._keys = ['a', 's']

    @property
    def keys(self):
        return list(self._keys)

    def transform(self, name: str, episode):
        if name == 'a':
            # 10x the base signal
            base = episode['s']
            return Elementwise(base, lambda seq: np.asarray(seq) * 10)
        if name == 's':
            # Override original 's' by adding 1
            base = episode['s']
            return Elementwise(base, lambda seq: np.asarray(seq) + 1)
        raise KeyError(name)


def test_transform_episode_keys_and_getitem_pass_through(sig_simple):
    # Build an episode with one signal 's' and two static fields
    ep = EpisodeContainer(signals={'s': sig_simple}, static={'id': 7, 'note': 'ok'}, meta={'origin': 'unit'})
    tf = _DummyTransform()
    te = TransformedEpisode(ep, tf, pass_through=True)

    # Keys order: transform keys first, then original non-overlapping keys
    assert list(te.keys) == ['a', 's', 'id', 'note']

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
    ep = EpisodeContainer(signals={'s': sig_simple}, static={'id': 3})
    tf = KeyFuncEpisodeTransform(
        double=lambda episode: Elementwise(
            episode['s'],
            lambda seq: np.asarray(seq) * 2,
        ),
        label=lambda episode: f'id={episode["id"]}',
    )

    assert list(tf.keys) == ['double', 'label']

    doubled = tf.transform('double', ep)
    assert [val for val, _ in doubled] == [2 * val for val, _ in ep['s']]
    assert tf.transform('label', ep) == 'id=3'

    with pytest.raises(KeyError):
        tf.transform('missing', ep)

    wrapped = TransformedEpisode(ep, tf, pass_through=False)
    assert list(wrapped.keys) == ['double', 'label']
    assert [val for val, _ in wrapped['double']] == [2 * val for val, _ in ep['s']]
    assert wrapped['label'] == 'id=3'


def test_transform_episode_pass_through_selected_keys(sig_simple):
    ep = EpisodeContainer(
        signals={'s': sig_simple},
        static={'id': 7, 'note': 'ok', 'skip': 'nope'},
    )
    tf = _DummyTransform()
    te = TransformedEpisode(ep, tf, pass_through=['note'])

    assert list(te.keys) == ['a', 's', 'note']
    assert [v for v, _ in te['a']] == [x * 10 for x, _ in ep['s']]
    assert te['note'] == 'ok'
    with pytest.raises(KeyError):
        _ = te['id']
    with pytest.raises(KeyError):
        _ = te['skip']


def test_transform_episode_no_pass_through(sig_simple):
    ep = EpisodeContainer(signals={'s': sig_simple}, static={'id': 7})
    tf = _DummyTransform()
    te = TransformedEpisode(ep, tf, pass_through=False)

    # Only transform keys
    assert list(te.keys) == ['a', 's']

    # Transform values present
    a_vals = [v for v, _ in te['a']]
    s_vals = [v for v, _ in te['s']]
    assert a_vals == [x * 10 for x, _ in ep['s']]
    assert s_vals == [x + 1 for x, _ in ep['s']]

    # Non-transform key should not be available
    with pytest.raises(KeyError):
        _ = te['id']


class _DummyTransform2(EpisodeTransform):
    @property
    def keys(self):
        return ['b', 's']

    def transform(self, name: str, episode):
        if name == 'b':
            base = episode['s']
            return Elementwise(base, lambda seq: np.asarray(seq) * -1)
        if name == 's':
            base = episode['s']
            return Elementwise(base, lambda seq: np.asarray(seq) + 100)
        raise KeyError(name)


def test_transform_episode_multiple_transforms_order_and_precedence(sig_simple):
    ep = EpisodeContainer(signals={'s': sig_simple}, static={'id': 42, 'z': 9})
    t1 = _DummyTransform()  # defines ["a", "s"] (s -> +1)
    t2 = _DummyTransform2()  # defines ["b", "s"] (s -> +100)

    # Concatenate transform keys in order; first occurrence of duplicates kept
    te = TransformedEpisode(ep, t1, t2, pass_through=True)
    assert list(te.keys) == ['a', 's', 'b', 'id', 'z']

    # 's' should come from the first transform (t1)
    s_vals = [v for v, _ in te['s']]
    assert s_vals == [x + 1 for x, _ in ep['s']]

    # Other transform keys are accessible
    a_vals = [v for v, _ in te['a']]
    b_vals = [v for v, _ in te['b']]
    assert a_vals == [x * 10 for x, _ in ep['s']]
    assert b_vals == [-(x) for x, _ in ep['s']]
