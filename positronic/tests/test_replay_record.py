import numpy as np
import pytest

from pimm.time import EMITTED_WORLD, RECEIVED_WORLD
from positronic import keys
from positronic.dataset import Time
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD
from positronic.replay_record import main
from positronic.simulator.mujoco.sim import MujocoSim
from positronic.utils import package_assets_path


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ('timelines', 'selected'),
    [((HARNESS_WORLD, RECORDED_TIME), None), ((RECEIVED_WORLD,), RECEIVED_WORLD), (('capture',), 'capture')],
)
def test_replay_records_commands_on_the_selected_clock(tmp_path, timelines, selected):
    model = package_assets_path('assets/mujoco/franka_table.xml')
    sim = MujocoSim(model)
    pose = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    with LocalDatasetWriter(tmp_path / 'source') as dataset:
        for timeline in timelines:
            with dataset.new_episode() as episode:
                for name, value in sim.save_state().items():
                    episode.set_static(name, value.tolist())
                for timestamp, grip in [(1_000_000_000, 0.2), (1_100_000_000, 0.7)]:
                    time = Time(**{timeline: timestamp})
                    episode.append(keys.TARGET_EE_POSE, pose, time)
                    episode.append(keys.TARGET_GRIP, grip, time)

    main(
        dataset=LocalDataset(tmp_path / 'source'),
        episodes='all',
        mujoco_model_path=model,
        cameras={},
        loaders=(),
        output_dir=str(tmp_path / 'replay'),
        timeline=selected,
    )

    recorded = LocalDataset(tmp_path / 'replay')
    assert len(recorded) == len(timelines)
    for episode in recorded:
        grips = episode[keys.TARGET_GRIP]
        np.testing.assert_allclose(list(grips.values()), [0.2, 0.7])
        np.testing.assert_array_equal(np.diff(grips.timestamps(EMITTED_WORLD)), [100_000_000])
