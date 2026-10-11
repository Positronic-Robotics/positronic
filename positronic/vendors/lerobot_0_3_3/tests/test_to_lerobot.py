import numpy as np
import pytest

from pimm.time import RECEIVED_WORLD
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.time import Time
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
