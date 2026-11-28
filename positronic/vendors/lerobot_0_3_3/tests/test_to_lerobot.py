import json
import pickle
import subprocess
import sys

import numpy as np
import pytest

from pimm.time import RECEIVED_WORLD
from positronic import keys
from positronic.cfg import dagger
from positronic.dagger.rounds import ROUND_FOLDERS
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.time import Time
from positronic.dataset.transforms import Elementwise, TransformedDataset, lazy_sequence
from positronic.dataset.transforms.episode import Derive
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD

pytest.importorskip('lerobot')

from positronic.vendors.lerobot_0_3_3.tests import records  # noqa: E402
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


def test_export_gives_one_row_for_an_episode_of_one_step(tmp_path):
    with LocalDatasetWriter(tmp_path) as dataset, dataset.new_episode() as writer:
        writer.append('value', 7, Time(**{HARNESS_WORLD: 100_000_000}))
    rows = EpisodeDictDataset(LocalDataset(tmp_path), fps=20)[0]
    np.testing.assert_array_equal(rows['value'], [7])


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


@pytest.mark.parametrize(
    ('codec', 'arms', 'cameras'),
    [('ee', keys.ONE_ARM, records.ONE_ARM_CAMERAS), ('ee_bimanual', keys.BIMANUAL_ARMS, records.TWO_ARM_CAMERAS)],
)
def test_the_convert_command_of_the_dagger_model_converts_the_training_data_of_a_round(tmp_path, codec, arms, cameras):
    model = dagger.lerobot_act.override(codec=codec, arms=arms, license='test').instantiate()
    folders = {name: tmp_path / name for name in ROUND_FOLDERS}
    records.write_record(folders['training_data'], arms, cameras)

    python, *arguments = (argument.format(**folders) for argument in model.convert)
    assert python == 'python'
    subprocess.run([sys.executable, *arguments, '--num_workers=0'], check=True)

    info = json.loads((folders['converted'] / 'meta' / 'info.json').read_text())
    assert info['total_episodes'] == 1
    width = 8 * len(arms)
    assert (info['features']['observation.state']['shape'], info['features']['action']['shape']) == ([width], [width])
    assert sum(name.startswith('observation.images.') for name in info['features']) == len(cameras)
