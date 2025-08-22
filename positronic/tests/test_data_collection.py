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

    # Timestamps should be strictly increasing for dynamic signals
    def assert_strictly_increasing(sig):
        for i in range(1, len(sig)):
            assert sig[i][1] > sig[i - 1][1]

    assert_strictly_increasing(r_trans)
    assert_strictly_increasing(tgt_grip)

    # Target grip should reflect the trigger value used above, and gripper state value should match emitted
    assert any(val == 0.7 for (val, _) in tgt_grip[:])
    assert grip[0][0] == 0.42


def test_data_collection_with_mujoco_robot_gripper(tmp_path):
    # Build Mujoco simulation and components
    from positronic.simulator.mujoco.sim import MujocoSim, MujocoFranka, MujocoGripper
    from positronic.data_collection import OperatorPosition

    sim = MujocoSim("positronic/assets/mujoco/franka_table.xml", loaders=())
    robot = MujocoFranka(sim, suffix='_ph')
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')

    # Use sim as the world clock to advance time with physics
    with pimm.World(clock=sim) as world:
        dc = DataCollection(operator_position=OperatorPosition.FRONT.value, output_dir=str(tmp_path), fps=30)

        # Wire minimal inputs: controller positions and buttons, plus robot/gripper channels
        ctrl_em, dc.controller_positions_reader = world.local_pipe()
        buttons_em, dc.buttons_reader = world.local_pipe()

        robot.state, dc.robot_state = world.local_pipe()
        dc.robot_commands, robot.commands = world.local_pipe()

        dc.target_grip_emitter, gripper.target_grip = world.local_pipe()
        gripper.grip, dc.gripper_state = world.local_pipe()

        # Interleave sim, robot, gripper, and data collection
        def driver(_stop, _clk):
            # Start recording with B press and set trigger=0.5
            buttons_em.emit({'left': None, 'right': [0.5, 0.0, 0.0, 0.0, 0.0, 1.0]})
            yield pimm.Sleep(0.01)

            # Enable tracking with A press so target pose is produced
            buttons_em.emit({'left': None, 'right': [0.5, 0.0, 0.0, 0.0, 1.0, 0.0]})
            yield pimm.Sleep(0.01)

            # Send controller pose to move the robot a bit; keep it identity to simplify
            ctrl_em.emit({'left': None, 'right': Transform3D.identity})
            yield pimm.Sleep(0.02)

            # Stop recording with another B press
            buttons_em.emit({'left': None, 'right': [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]})
            yield pimm.Sleep(0.005)
            buttons_em.emit({'left': None, 'right': [0.5, 0.0, 0.0, 0.0, 0.0, 1.0]})
            yield pimm.Sleep(0.005)

        steps = iter(world.interleave(sim.run, robot.run, gripper.run, dc.run, driver))
        # Advance a reasonable number of steps
        for _ in range(400):
            try:
                next(steps)
            except StopIteration:
                break

    # Validate dataset contents
    ds = LocalDataset(tmp_path)
    assert len(ds) == 1
    ep = ds[0]

    expected = {
        'target_grip',
        'right_controller_translation',
        'right_controller_quaternion',
        'target_robot_position_translation',
        'target_robot_position_quaternion',
        'robot_position_translation',
        'robot_position_rotation',
        'robot_joints',
        'robot_joints_velocity',
        'grip',
    }
    assert expected.issubset(set(ep.keys))

    # Robot/gripper signals should have at least one sample
    robot_j = ep['robot_joints']
    robot_pos_t = ep['robot_position_translation']
    robot_pos_q = ep['robot_position_rotation']
    grip_sig = ep['grip']
    assert len(robot_j) >= 1
    assert len(grip_sig) >= 1

    # Controller pose was identity; verify controller signals reflect that
    rc_t = ep['right_controller_translation']
    rc_q = ep['right_controller_quaternion']
    np.testing.assert_allclose(rc_t[0][0], np.zeros(3))
    np.testing.assert_allclose(rc_q[0][0], Rotation.identity.as_quat)

    # When tracking is enabled, target pose initially matches current robot state (due to offset calibration)
    tr_t = ep['target_robot_position_translation']
    tr_q = ep['target_robot_position_quaternion']
    # Sample robot position at or before target ts using episode time indexer
    t_ts = int(tr_t[0][1])
    sample = ep.time[t_ts]
    np.testing.assert_allclose(tr_t[0][0], sample['robot_position_translation'][0], atol=5e-3)
    np.testing.assert_allclose(tr_q[0][0], sample['robot_position_rotation'][0], atol=5e-3)

    # Basic sanity on sizes and monotonic timestamps
    def assert_strictly_increasing(sig):
        for i in range(1, len(sig)):
            assert sig[i][1] > sig[i - 1][1]

    for name in ['robot_joints', 'robot_joints_velocity', 'robot_position_translation', 'grip']:
        assert_strictly_increasing(ep[name])
