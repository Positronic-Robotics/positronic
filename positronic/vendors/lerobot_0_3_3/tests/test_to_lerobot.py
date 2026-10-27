import pickle

import numpy as np
import pytest

from pimm.time import RECEIVED_WORLD
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.time import Time
from positronic.dataset.transforms import Elementwise, TransformedDataset, lazy_sequence
from positronic.dataset.transforms.episode import Derive
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD

pytest.importorskip('lerobot')

from positronic.vendors.lerobot_0_3_3.to_lerobot import EpisodeDictDataset  # noqa: E402


@pytest.mark.parametrize(
    ('preferred', 'other'),
    [(HARNESS_WORLD, RECEIVED_WORLD), (HARNESS_WORLD, RECORDED_TIME), (RECEIVED_WORLD, RECORDED_TIME)],
)
def test_export_uses_the_preferred_clock(tmp_path, preferred, other):
    with LocalDatasetWriter(tmp_path) as dataset, dataset.new_episode() as writer:
        for i in range(3):
            writer.append('value', i, Time(**{preferred: i * 100_000_000, other: i * 1_000_000_000}))
    rows = EpisodeDictDataset(LocalDataset(tmp_path), fps=20)[0]
    np.testing.assert_array_equal(rows['value'], [0, 0, 1, 1])


def test_export_rows_survive_the_pickle_of_a_dataloader_worker(tmp_path):
    with LocalDatasetWriter(tmp_path) as dataset, dataset.new_episode() as writer:
        for i in range(3):
            writer.append('value', i, Time(**{HARNESS_WORLD: i * 100_000_000}))
    offset = 10
    shifted = Derive(value=lambda episode: Elementwise(episode['value'], lazy_sequence(lambda value: value + offset)))
    rows = EpisodeDictDataset(TransformedDataset(LocalDataset(tmp_path), shifted), fps=20)[0]
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(rows))['value'], [10, 10, 11, 11])


def test_export_rows_of_a_video_survive_the_pickle_of_a_dataloader_worker(tmp_path):
    with LocalDatasetWriter(tmp_path) as dataset, dataset.new_episode() as writer:
        for i in range(3):
            image = np.full((16, 16, 3), 40 * i, dtype=np.uint8)
            writer.append('image', image, Time(**{HARNESS_WORLD: i * 100_000_000}))
    rows = EpisodeDictDataset(LocalDataset(tmp_path), fps=20)[0]
    assert isinstance(rows['image'], list)
    frames = pickle.loads(pickle.dumps(rows))['image']
    np.testing.assert_allclose([np.median(frame) for frame in frames], [0, 0, 40, 40], atol=5)


def test_export_raises_the_error_of_a_frame_that_does_not_decode(tmp_path):
    with LocalDatasetWriter(tmp_path) as dataset, dataset.new_episode() as writer:
        for i in range(3):
            writer.append('value', i, Time(**{HARNESS_WORLD: i * 100_000_000}))

    def decode(value):
        if value == 1:
            raise IndexError('Could not decode frame 1')
        return value

    broken = Derive(value=lambda episode: Elementwise(episode['value'], lazy_sequence(decode)))
    with pytest.raises(IndexError, match='Could not decode frame 1'):
        EpisodeDictDataset(TransformedDataset(LocalDataset(tmp_path), broken), fps=20)[0]
