import numpy as np
import pytest

from positronic.dataset.dataset import Dataset
from positronic.dataset.episode import Episode, EpisodeContainer
from positronic.dataset.transforms import TransformedDataset

from ...tests.utils import DummySignal, DummyTransform


class _DummyDataset(Dataset):
    """Minimal Dataset implementation used for TransformedDataset tests."""

    def __init__(self, episodes):
        self._episodes = list(episodes)
        self.getitem_calls = 0

    def __len__(self):
        return len(self._episodes)

    def _get_episode(self, index: int):
        self.getitem_calls += 1
        return self._episodes[index]


def test_transformed_dataset_wraps_episode_with_transforms():
    sig_simple = DummySignal([1000, 2000, 3000, 4000, 5000], [10, 20, 30, 40, 50])
    base_meta = {'a': sig_simple.meta, 's': sig_simple.meta}
    episode = EpisodeContainer(data={'s': sig_simple, 'id': 99}, meta=base_meta)
    dataset = _DummyDataset([episode])
    tf = DummyTransform(operations={'a': ('s', lambda x: x * 10), 's': ('s', lambda x: x + 1)}, pass_through=True)

    transformed = TransformedDataset(dataset, tf)

    assert len(transformed) == 1
    wrapped = transformed[0]
    assert isinstance(wrapped, Episode)
    assert list(wrapped.keys()) == ['a', 's', 'id']

    a_vals = [v for v, _ in wrapped['a']]
    s_vals = [v for v, _ in wrapped['s']]
    assert a_vals == [x * 10 for x, _ in episode['s']]
    assert s_vals == [x + 1 for x, _ in episode['s']]
    assert wrapped['id'] == 99


def test_transformed_dataset_pass_through_selected_keys():
    sig_simple = DummySignal([1000, 2000], [10, 20])
    episode = EpisodeContainer(data={'s': sig_simple, 'id': 42, 'note': 'keep', 'skip': 'drop'})
    dataset = _DummyDataset([episode])
    tf = DummyTransform(operations={'a': ('s', lambda x: x * 10), 's': ('s', lambda x: x + 1)}, pass_through=['note'])

    transformed = TransformedDataset(dataset, tf)

    wrapped = transformed[0]
    assert list(wrapped.keys()) == ['a', 's', 'note']
    assert wrapped['note'] == 'keep'
    with pytest.raises(KeyError):
        _ = wrapped['id']
    with pytest.raises(KeyError):
        _ = wrapped['skip']


def test_transformed_dataset_sequence_indices_return_transformed_episodes():
    episodes = []
    for idx in range(3):
        ts = [1000, 2000]
        values = [idx * 10 + 1, idx * 10 + 2]
        sig = DummySignal(ts, values)
        episodes.append(EpisodeContainer(data={'s': sig, 'id': idx}))

    dataset = _DummyDataset(episodes)
    tf = DummyTransform(operations={'a': ('s', lambda x: x * 10), 's': ('s', lambda x: x + 1)}, pass_through=True)
    transformed = TransformedDataset(dataset, tf)

    sliced = transformed[1:3]
    assert len(sliced) == 2
    for offset, episode in enumerate(sliced, start=1):
        assert isinstance(episode, Episode)
        base_vals = [val for val, _ in episodes[offset]['s']]
        assert [val for val, _ in episode['s']] == [val + 1 for val in base_vals]
        assert [val for val, _ in episode['a']] == [val * 10 for val in base_vals]
        assert episode['id'] == offset

    idx_array = np.array([0, 2])
    selected = transformed[idx_array]
    assert [ep['id'] for ep in selected] == [0, 2]
    for pos, episode in zip((0, 2), selected, strict=True):
        base_vals = [val for val, _ in episodes[pos]['s']]
        assert [val for val, _ in episode['s']] == [val + 1 for val in base_vals]
        assert [val for val, _ in episode['a']] == [val * 10 for val in base_vals]
