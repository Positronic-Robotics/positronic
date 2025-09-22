from pathlib import Path
from typing import Dict

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock
from positronic.data_collection import DataCollectionController, controller_positions_serializer
from positronic.dataset.ds_writer_agent import DsWriterAgent, DsWriterCommand, DsWriterCommandType, Serializers
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.geom import Rotation, Transform3D
from positronic.tests.testing_coutils import ManualDriver, drive_scheduler


# TODO: Move these fixtures into a common module so that others can reuse them.
@pytest.fixture
def clock():
    return MockClock()


@pytest.fixture
def world(clock):
    with pimm.World(clock=clock) as w:
        yield w


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
    writer_cm = LocalDatasetWriter(out_dir)
    writer = writer_cm.__enter__()
    agent = DsWriterAgent(writer, spec)

    world.connect(dc.target_grip_emitter, agent.inputs['target_grip'])
    world.connect(dc.ds_agent_commands, agent.command)

    ctrl_em_dc = world.pair(dc.controller_positions_receiver)
    ctrl_em_agent = world.pair(agent.inputs['controller_positions'])
    buttons_em = world.pair(dc.buttons_receiver)
    grip_em = world.pair(agent.inputs['grip'])

    return dc, agent, ctrl_em_dc, ctrl_em_agent, buttons_em, grip_em, writer_cm


def test_data_collection_basic_recording(tmp_path, world, clock):
    dc, agent, ctrl_em_dc, ctrl_em_agent, buttons_em, grip_em, writer_cm = build_collection(world, tmp_path)

    # A simple right-hand pose and button frames
    right_pose = Transform3D(translation=np.array([0.1, 0.2, 0.3]), rotation=Rotation.identity)

    payload = {'left': None, 'right': right_pose}

    def start_episode():
        dc.ds_agent_commands.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        buttons_em.emit(make_buttons(trigger=0.7, B=False))

    def emit_signals():
        ctrl_em_dc.emit(payload)
        ctrl_em_agent.emit(payload)
        grip_em.emit(0.42)

    def stop_episode():
        dc.ds_agent_commands.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))

    driver = ManualDriver([
        (start_episode, 0.001),
        (emit_signals, 0.001),
        (stop_episode, 0.001),
    ])

    with writer_cm:
        scheduler = world.start([dc, agent, driver])
        drive_scheduler(scheduler, clock=clock)

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
    assert right_pose_sig.names == Serializers.transform_3d.names
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
        writer_cm = LocalDatasetWriter(tmp_path)
        agent = DsWriterAgent(writer_cm.__enter__(), spec)

        world.connect(robot.state, dc.robot_state)
        world.connect(robot.state, agent.inputs['robot_state'])
        world.connect(dc.robot_commands, robot.commands)
        world.connect(dc.robot_commands, agent.inputs['robot_commands'])
        world.connect(dc.target_grip_emitter, gripper.target_grip)
        world.connect(dc.target_grip_emitter, agent.inputs['target_grip'])
        world.connect(gripper.grip, agent.inputs['grip'])
        world.connect(dc.ds_agent_commands, agent.command)

        ctrl_em_dc = world.pair(dc.controller_positions_receiver)
        ctrl_em_agent = world.pair(agent.inputs['controller_positions'])
        buttons_em = world.pair(dc.buttons_receiver)

        def start_episode():
            dc.ds_agent_commands.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
            buttons_em.emit(make_buttons(trigger=0.5))

        def enable_tracking():
            buttons_em.emit(make_buttons(trigger=0.5, A=True))

        def emit_pose():
            payload = {'left': None, 'right': Transform3D.identity}
            ctrl_em_dc.emit(payload)
            ctrl_em_agent.emit(payload)

        def stop_episode():
            dc.ds_agent_commands.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))

        driver = ManualDriver([
            (start_episode, 0.01),
            (enable_tracking, 0.01),
            (emit_pose, 0.02),
            (stop_episode, 0.005),
        ])

        with writer_cm:
            scheduler = world.start([sim, robot, gripper, dc, agent, driver])
            drive_scheduler(scheduler, steps=400)

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
