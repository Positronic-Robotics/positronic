from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock
from positronic.data_collection import DataCollectionController, controller_positions_serializer
from positronic.dataset.ds_writer_agent import DsWriterAgent, DsWriterCommand, DsWriterCommandType, Serializers
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.geom import Rotation, Transform3D


# TODO: Move these fixtures into a common module so that others can reuse them.
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


def make_buttons(*,
                 trigger: float = 0.0,
                 thumb: float = 0.0,
                 stick: float = 0.0,
                 A: bool = False,
                 B: bool = False) -> Dict:
    """Constructs controller buttons payload matching DataCollection mapping."""
    return {
        'left': None,
        'right': [trigger, thumb, 0.0, stick, 1.0 if A else 0.0, 1.0 if B else 0.0],
    }


def assert_strictly_increasing(sig):
    for i in range(1, len(sig)):
        assert sig[i][1] > sig[i - 1][1]


def build_collection(world, out_dir: Path):
    dc = DataCollectionController(operator_position=None)

    # Build ds_agent with minimal signals: target_grip, controller_positions, grip
    spec = {
        'target_grip': None,
        'controller_positions': controller_positions_serializer,
        'grip': None,
    }
    agent = DsWriterAgent(LocalDatasetWriter(out_dir), spec)

    # Wire controller positions to both DC and agent
    ctrl_em, (ctrl_rd_dc, ctrl_rd_agent) = world.local_one_to_many_pipe(2)
    dc.controller_positions_reader = ctrl_rd_dc
    agent.inputs['controller_positions'] = ctrl_rd_agent

    # Wire buttons to DC
    buttons_em, dc.buttons_reader = world.local_pipe()

    # DC emits target grip -> agent receives
    tg_em, agent.inputs['target_grip'] = world.local_pipe()
    dc.target_grip_emitter = tg_em

    # Directly emit gripper state to agent for test purposes
    grip_em, agent.inputs['grip'] = world.local_pipe()

    # DC emits start/stop commands -> agent receives
    cmd_em, agent.command = world.local_pipe()
    dc.ds_agent_commands = cmd_em

    return dc, agent, ctrl_em, buttons_em, grip_em, cmd_em


def test_data_collection_basic_recording(tmp_path, world, run_interleaved):
    dc, agent, ctrl_em, buttons_em, grip_em, cmd_em = build_collection(world, tmp_path)

    # A simple right-hand pose and button frames
    right_pose = Transform3D(translation=np.array([0.1, 0.2, 0.3]), rotation=Rotation.identity)

    def driver(_stop, _clk) -> Iterator[pimm.Sleep]:
        # Start recording and set right_trigger value.
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        buttons_em.emit(make_buttons(trigger=0.7, B=False))
        yield pimm.Sleep(0.001)

        # Send controller pose and a gripper state update
        ctrl_em.emit({'left': None, 'right': right_pose})
        grip_em.emit(0.42)
        yield pimm.Sleep(0.001)

        # Stop recording
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))
        yield pimm.Sleep(0.001)

    run_interleaved(dc.run, lambda s, c: agent.run(s, c), driver)

    ds = LocalDataset(tmp_path)
    assert len(ds) == 1
    ep = ds[0]

    # Expect these keys to be present and have records under new serializers
    expected_keys = {
        'target_grip',
        'controller_positions.right',
        'grip',
    }
    assert expected_keys.issubset(set(ep.keys))

    right_pose_sig = ep['controller_positions.right']
    tgt_grip = ep['target_grip']
    grip = ep['grip']

    assert len(right_pose_sig) == 1
    assert len(tgt_grip) >= 1
    assert len(grip) == 1

    np.testing.assert_allclose(right_pose_sig[0][0][:3], right_pose.translation)
    np.testing.assert_allclose(right_pose_sig[0][0][3:], right_pose.rotation.as_quat)

    assert_strictly_increasing(tgt_grip)
    assert_strictly_increasing(tgt_grip)

    # Target grip should reflect the trigger value used above, and gripper state value should match emitted
    assert any(val == 0.7 for (val, _) in tgt_grip[:])
    assert grip[0][0] == 0.42


def test_data_collection_with_mujoco_robot_gripper(tmp_path):
    # Build Mujoco simulation and components
    from positronic.data_collection import OperatorPosition
    from positronic.simulator.mujoco.sim import MujocoFranka, MujocoGripper, MujocoSim

    sim = MujocoSim("positronic/assets/mujoco/franka_table.xml", loaders=())
    robot = MujocoFranka(sim, suffix='_ph')
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')

    # Use sim as the world clock to advance time with physics
    with pimm.World(clock=sim) as world:
        dc = DataCollectionController(operator_position=OperatorPosition.FRONT.value)

        # Build ds_agent and wiring analogous to main_sim
        spec = {
            'target_grip': None,
            'robot_commands': Serializers.robot_command,
            'controller_positions': controller_positions_serializer,
            'robot_state': Serializers.robot_state,
            'grip': None,
        }
        agent = DsWriterAgent(LocalDatasetWriter(tmp_path), spec)

        # Controller positions to DC and agent
        ctrl_em, (ctrl_rd_dc, ctrl_rd_agent) = world.local_one_to_many_pipe(2)
        dc.controller_positions_reader = ctrl_rd_dc
        agent.inputs['controller_positions'] = ctrl_rd_agent

        # Buttons to DC
        buttons_em, dc.buttons_reader = world.local_pipe()

        # Robot state to DC and agent
        rstate_em, (rstate_rd_agent, rstate_rd_dc) = world.local_one_to_many_pipe(2)
        agent.inputs['robot_state'] = rstate_rd_agent
        dc.robot_state = rstate_rd_dc
        robot.state = rstate_em

        # Commands from DC to robot and agent
        cmd_em, (cmd_rd_robot, cmd_rd_agent) = world.local_one_to_many_pipe(2)
        dc.robot_commands = cmd_em
        robot.commands = cmd_rd_robot
        agent.inputs['robot_commands'] = cmd_rd_agent

        # Target grip to gripper and agent
        tg_em, (tg_rd_gripper, tg_rd_agent) = world.local_one_to_many_pipe(2)
        dc.target_grip_emitter = tg_em
        gripper.target_grip = tg_rd_gripper
        agent.inputs['target_grip'] = tg_rd_agent

        # Gripper state direct to agent
        grip_em, agent.inputs['grip'] = world.local_pipe()
        gripper.grip = grip_em

        # DC emits start/stop commands -> agent receives
        cmd_em, agent.command = world.local_pipe()
        dc.ds_agent_commands = cmd_em

        # Interleave sim, robot, gripper, and data collection
        def driver(_stop, _clk):
            # Start recording and set trigger=0.5
            cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
            buttons_em.emit(make_buttons(trigger=0.5))
            yield pimm.Sleep(0.01)

            # Enable tracking with A press so target pose is produced
            buttons_em.emit(make_buttons(trigger=0.5, A=True))
            yield pimm.Sleep(0.01)

            # Send controller pose to move the robot a bit; keep it identity to simplify
            ctrl_em.emit({'left': None, 'right': Transform3D.identity})
            yield pimm.Sleep(0.02)

            # Stop recording
            cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))
            yield pimm.Sleep(0.005)

        steps = iter(world.interleave(sim.run, robot.run, gripper.run, dc.run, lambda s, c: agent.run(s, c), driver))
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
        'controller_positions.right',
        'robot_state.q',
        'robot_state.dq',
        'robot_state.ee_pose',
        'grip',
    }
    assert expected.issubset(set(ep.keys))

    # Robot/gripper signals should have at least one sample
    robot_j = ep['robot_state.q']
    grip_sig = ep['grip']
    assert len(robot_j) >= 1
    assert len(grip_sig) >= 1

    # Controller pose was identity; verify controller signals reflect that
    rc = ep['controller_positions.right']
    np.testing.assert_allclose(rc[0][0][:3], np.zeros(3))
    np.testing.assert_allclose(rc[0][0][3:], Rotation.identity.as_quat)

    # When tracking is enabled, target pose initially matches current robot state (due to offset calibration)
    # Verify a robot command was emitted (tracking enabled and a pose sent)
    # We don't assert exact equality with state here; just presence and shape.
    cmd_pose = ep['robot_commands.pose']
    assert len(cmd_pose) >= 1 and cmd_pose[0][0].shape == (7, )

    # Basic sanity on sizes and monotonic timestamps
    def assert_strictly_increasing(sig):
        for i in range(1, len(sig)):
            assert sig[i][1] > sig[i - 1][1]

    for name in ['robot_state.q', 'robot_state.dq', 'grip']:
        assert_strictly_increasing(ep[name])
