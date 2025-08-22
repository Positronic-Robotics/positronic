from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import pytest

import pimm
from pimm.testing import MockClock

from positronic.geom import Transform3D, Rotation
from positronic.data_collection import DataCollectionController as DataCollection
from positronic.dataset.local_dataset import LocalDataset


@pytest.fixture
def clock():
    return MockClock()


@pytest.fixture
def world(clock):
    with pimm.World(clock=clock) as w:
        yield w


@pytest.fixture
def run_interleaved(clock, world):
    def _run(*loops, steps: int = 200):
        it = world.interleave(*loops)
        for _ in range(steps):
            try:
                sleep = next(it)
            except StopIteration:
                break
            clock.advance(sleep.seconds)
    return _run


def build_collection(world, out_dir: Path):
    dc = DataCollection(operator_position=None, output_dir=str(out_dir), fps=30)

    # Wire inputs we will drive in the test
    ctrl_em, dc.controller_positions_reader = world.local_pipe()
    buttons_em, dc.buttons_reader = world.local_pipe()
    grip_em, dc.gripper_state = world.local_pipe()

    return dc, ctrl_em, buttons_em, grip_em


def test_data_collection_basic_recording(tmp_path, world, run_interleaved):
    dc, ctrl_em, buttons_em, grip_em = build_collection(world, tmp_path)

    # A simple right-hand pose and button frames
    right_pose = Transform3D(translation=np.array([0.1, 0.2, 0.3]), rotation=Rotation.identity)

    def driver(_stop, _clk) -> Iterator[pimm.Sleep]:
        # Press right_B to start recording. Also set right_trigger value.
        buttons_em.emit({
            'left': None,
            'right': [0.7, 0.0, 0.0, 0.0, 0.0, 1.0],  # trigger=0.7, B=1.0
        })
        yield pimm.Sleep(0.001)

        # Send controller pose and a gripper state update
        ctrl_em.emit({'left': None, 'right': right_pose})
        grip_em.emit(0.42)
        yield pimm.Sleep(0.001)

        # Release B then press again to stop
        buttons_em.emit({'left': None, 'right': [0.7, 0.0, 0.0, 0.0, 0.0, 0.0]})
        yield pimm.Sleep(0.001)
        buttons_em.emit({'left': None, 'right': [0.7, 0.0, 0.0, 0.0, 0.0, 1.0]})
        yield pimm.Sleep(0.001)

    run_interleaved(dc.run, driver)

    ds = LocalDataset(tmp_path)
    assert len(ds) == 1
    ep = ds[0]

    # Expect these keys to be present and have one record each
    expected_keys = {
        'target_grip',
        'right_controller_translation',
        'right_controller_quaternion',
        'target_robot_position_translation',
        'target_robot_position_quaternion',
        'grip',
    }
    assert expected_keys.issubset(set(ep.keys))

    r_trans = ep['right_controller_translation']
    r_quat = ep['right_controller_quaternion']
    t_trans = ep['target_robot_position_translation']
    t_quat = ep['target_robot_position_quaternion']
    tgt_grip = ep['target_grip']
    grip = ep['grip']

    assert len(r_trans) == len(r_quat) == len(t_trans) == len(t_quat) == 1
    assert len(tgt_grip) >= 1
    assert len(grip) == 1

    np.testing.assert_allclose(r_trans[0][0], right_pose.translation)
    np.testing.assert_allclose(t_trans[0][0], right_pose.translation)
    np.testing.assert_allclose(r_quat[0][0], right_pose.rotation.as_quat)
    np.testing.assert_allclose(t_quat[0][0], right_pose.rotation.as_quat)

    # Target grip should reflect the trigger value used above, and gripper state value should match emitted
    assert any(val == 0.7 for (val, _) in tgt_grip[:])
    assert grip[0][0] == 0.42
