from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

import pimm
from positronic import data_collection, keys, wire
from positronic.data_collection import DataCollectionController, OperatorPosition, controller_positions_serializer
from positronic.dataset.ds_writer_agent import DsWriterAgent, DsWriterCommand
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.webxr import WebXR
from positronic.geom import Rotation, Transform3D
from positronic.simulator.mujoco.sim import MujocoSim
from positronic.tests.testing_coutils import ManualDriver, RecordingEmitter, drive_scheduler


# TODO: Move these fixtures into a common module so that others can reuse them.
@pytest.fixture
def world():
    with pimm.World(virtual_time=True) as w:
        yield w


def make_buttons(
    *, trigger: float = 0.0, thumb: float = 0.0, stick: float = 0.0, A: bool = False, B: bool = False
) -> dict:
    """Constructs controller buttons payload matching DataCollection mapping."""
    return {'left': None, 'right': [trigger, thumb, 0.0, stick, 1.0 if A else 0.0, 1.0 if B else 0.0]}


def assert_strictly_increasing(sig):
    for i in range(1, len(sig)):
        assert sig[i][1] > sig[i - 1][1]


class DummyRobot(pimm.ControlSystem):
    def __init__(self):
        self.commands = pimm.FakeReceiver(self)
        self.state = pimm.FakeEmitter(self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, _clock: pimm.Clock):
        while not should_stop.value:
            yield pimm.Sleep(0.1)


def build_collection(world, out_dir: Path, *, metadata_getter: Callable[[], dict[str, object]] | None = None):
    dc = DataCollectionController(
        operator_position=None, nominal_joints=(), output_path=out_dir, metadata_getter=metadata_getter
    )
    robot = DummyRobot()

    ds_agent = wire.wire(world, dc, LocalDatasetWriter, {}, robot, None, None)
    assert ds_agent is not None
    ds_agent.add_signal('controller_positions', controller_positions_serializer)

    world.connect(dc.ds_agent_commands, ds_agent.command)

    ctrl_em_dc = world.pair(dc.controller_positions)
    ctrl_em_agent = world.pair(ds_agent.inputs['controller_positions'])
    buttons_em = world.pair(dc.buttons_receiver)

    return dc, ds_agent, ctrl_em_dc, ctrl_em_agent, buttons_em, robot


def test_data_collection_records_task_metadata(tmp_path, world):
    call_count = 0

    def metadata_getter():
        nonlocal call_count
        call_count += 1
        return {keys.TASK: 'stack-blocks'}

    (dc, agent, ctrl_em_dc, ctrl_em_agent, buttons_em, robot) = build_collection(
        world, tmp_path, metadata_getter=metadata_getter
    )

    right_pose = Transform3D(translation=np.array([0.2, 0.1, -0.1]), rotation=Rotation.identity)
    controller_payload = {'left': None, 'right': right_pose}

    def emit_pose():
        ctrl_em_dc.emit(controller_payload)
        ctrl_em_agent.emit(controller_payload)

    def send_buttons(**kwargs):
        buttons_em.emit(make_buttons(**kwargs))

    driver = ManualDriver([
        (lambda: send_buttons(trigger=0.0, B=False), 0.002),
        (lambda: send_buttons(trigger=0.9, B=True), 0.002),
        (emit_pose, 0.002),
        (lambda: send_buttons(trigger=0.1, B=False), 0.002),
        (lambda: send_buttons(trigger=0.8, B=True), 0.002),
        (None, 0.005),
    ])

    scheduler = world.start([dc, agent, robot, driver])
    drive_scheduler(scheduler, steps=400)

    assert call_count == 1

    dataset = LocalDataset(tmp_path)
    assert len(dataset) == 1
    episode = dataset[0]
    assert isinstance(episode, Episode)
    assert episode[keys.TASK] == 'stack-blocks'


def test_data_collection_basic_recording(tmp_path, world):
    dc, agent, ctrl_em_dc, ctrl_em_agent, buttons_em, robot = build_collection(world, tmp_path)

    # A simple right-hand pose and button frames
    right_pose = Transform3D(translation=np.array([0.1, 0.2, 0.3]), rotation=Rotation.identity)

    payload = {'left': None, 'right': right_pose}

    def start_episode():
        dc.ds_agent_commands.emit(DsWriterCommand.START(tmp_path))
        buttons_em.emit(make_buttons(trigger=0.7, B=False))

    def emit_signals():
        ctrl_em_dc.emit(payload)
        ctrl_em_agent.emit(payload)

    def stop_episode():
        dc.ds_agent_commands.emit(DsWriterCommand.STOP())

    driver = ManualDriver([(start_episode, 0.001), (emit_signals, 0.001), (stop_episode, 0.001)])

    scheduler = world.start([dc, agent, robot, driver])
    drive_scheduler(scheduler)

    ds = LocalDataset(tmp_path)
    assert len(ds) == 1
    ep = ds[0]

    expected_keys = {'controller_positions.right'}
    assert expected_keys.issubset(set(ep.keys()))

    right_pose_sig = ep['controller_positions.right']
    assert len(right_pose_sig) == 1

    np.testing.assert_allclose(right_pose_sig[0][0][:3], right_pose.translation)
    np.testing.assert_allclose(right_pose_sig[0][0][3:], right_pose.rotation.as_quat)


NOMINAL_JOINTS = np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0])
JOINTS_SPREAD = np.array([0.03, 0.05, 0.08, 0.08, 0.10, 0.10, 0.10])


def build_teleop_arm(world, spread: Sequence[float] | np.ndarray = JOINTS_SPREAD):
    """The controller with the arm side of its ``sync_move`` paired, and recorders on what it emits."""
    dc = DataCollectionController(OperatorPosition.FRONT.value, NOMINAL_JOINTS, spread)
    grips, sounds = RecordingEmitter(), RecordingEmitter()
    dc.target_grip._bind(grips)
    dc.sound._bind(sounds)
    return dc, world.pair(dc.sync_move), world.pair(dc.buttons_receiver), grips, sounds


@pytest.mark.parametrize(
    ('robot_arm', 'nominal_joints'),
    [(DummyRobot(), ()), (None, NOMINAL_JOINTS)],
    ids=['an arm with no pose to put it at', 'a pose with no arm to put there'],
)
def test_a_station_naming_an_arm_or_a_start_pose_without_the_other_is_refused(robot_arm, nominal_joints):
    """A start pose belongs to the arm it was measured on: a station that swaps its arm and keeps the pose
    would drive the new one to the old one's joints, and one that drops the arm has nothing to move."""
    with pytest.raises(ValueError, match='nominal_joints'):
        data_collection.main(
            robot_arm=robot_arm,
            gripper=None,
            webxr=WebXR(port=0),
            sound=None,
            cameras=None,
            nominal_joints=nominal_joints,
        )


def test_a_spread_that_does_not_cover_every_nominal_joint_is_refused():
    """A spread holds one measurement per joint. Widths that disagree reach numpy on the first right-stick
    press, where a misconfigured station is indistinguishable from an arm that refused the move."""
    with pytest.raises(ValueError, match='joints_spread'):
        data_collection.main(
            robot_arm=DummyRobot(),
            gripper=None,
            webxr=WebXR(port=0),
            sound=None,
            cameras=None,
            nominal_joints=NOMINAL_JOINTS.tolist(),
            joints_spread=JOINTS_SPREAD[:-1].tolist(),
        )


def test_the_right_stick_holds_the_session_until_the_arm_is_at_its_start_pose(world):
    """The right stick asks for a start pose drawn around the nominal joints, and the controller does
    nothing further until the arm answers that it is there."""
    dc, arm, buttons, grips, _ = build_teleop_arm(world)
    asked, marks = [], {}

    driver = ManualDriver([
        (lambda: buttons.emit(make_buttons(stick=0.0)), 0.01),
        (lambda: buttons.emit(make_buttons(stick=1.0)), 0.01),
        (lambda: asked.extend(arm.incoming()), 0.01),
        (lambda: marks.update(asked=len(grips.emitted)), 0.01),
        (lambda: marks.update(waited=len(grips.emitted)), 0.01),
        (lambda: asked[0].set_result(None), 0.01),
        (lambda: marks.update(arrived=len(grips.emitted)), 0.0),
    ])
    drive_scheduler(world.start([dc, driver]), steps=400)

    assert len(asked) == 1
    move = asked[0].request
    assert isinstance(move, roboarm_command.JointPosition)
    np.testing.assert_array_less(np.abs(np.asarray(move.positions) - NOMINAL_JOINTS), JOINTS_SPREAD)
    assert marks['asked'] > 0  # the loop was running before the press
    assert marks['waited'] == marks['asked']  # ... and stood still for the whole move
    assert marks['arrived'] > marks['waited']


def test_a_start_pose_naming_something_that_is_not_an_angle_is_refused():
    """A draw between non-finite bounds raises where the operator hears nothing but a failed move, so the
    station is refused before any hardware starts."""
    with pytest.raises(ValueError, match='finite'):
        data_collection.main(
            robot_arm=DummyRobot(),
            gripper=None,
            webxr=WebXR(port=0),
            sound=None,
            cameras=None,
            nominal_joints=NOMINAL_JOINTS.tolist(),
            joints_spread=[*JOINTS_SPREAD[:-1].tolist(), float('nan')],
        )


def test_the_right_stick_redraws_the_scene_the_arm_is_put_back_into(world):
    """One press readies every device the station has: a sim's scene is drawn again alongside the arm's
    start pose, and the session stays held until both have answered."""
    dc, arm, buttons, grips, _ = build_teleop_arm(world)
    scene = world.pair(dc.redraw_scene)
    asked, marks = [], {}

    driver = ManualDriver([
        (lambda: buttons.emit(make_buttons(stick=0.0)), 0.01),
        (lambda: buttons.emit(make_buttons(stick=1.0)), 0.01),
        (lambda: asked.extend([*scene.incoming(), *arm.incoming()]), 0.01),
        (lambda: marks.update(asked=len(grips.emitted)), 0.01),
        (lambda: asked[0].set_result(None), 0.01),
        (lambda: marks.update(drawn=len(grips.emitted)), 0.01),
        (lambda: asked[1].set_result(None), 0.01),
        (lambda: marks.update(placed=len(grips.emitted)), 0.0),
    ])
    drive_scheduler(world.start([dc, driver]), steps=400)

    assert len(asked) == 2, 'the press reached only one of the scene and the arm'
    assert marks['drawn'] == marks['asked'], 'the session went on with the arm still travelling'
    assert marks['placed'] > marks['drawn']


def test_a_start_pose_the_arm_refuses_is_sounded_to_the_operator(world):
    """A move the arm fails is the operator's to hear about and ask for again; the session goes on."""
    dc, arm, buttons, grips, sounds = build_teleop_arm(world)
    asked, marks = [], {}

    def refuse_the_move():
        marks['refused'] = len(grips.emitted)
        asked[0].set_exception(RuntimeError('joint limit'))

    driver = ManualDriver([
        (lambda: buttons.emit(make_buttons(stick=0.0)), 0.01),
        (lambda: buttons.emit(make_buttons(stick=1.0)), 0.01),
        (lambda: asked.extend(arm.incoming()), 0.01),
        (refuse_the_move, 0.01),
        (lambda: marks.update(after=len(grips.emitted)), 0.0),
    ])
    drive_scheduler(world.start([dc, driver]), steps=400)

    assert [path.name for _, path in sounds.emitted] == ['error-occurred.wav']
    assert marks['after'] > marks['refused']


def test_a_station_that_measured_no_spread_puts_the_arm_at_its_nominal(world):
    """Jitter is a station's to measure, and the arms that have none named are the ones asked for their
    nominal joints themselves."""
    dc, arm, buttons, _, _ = build_teleop_arm(world, spread=())
    asked = []

    driver = ManualDriver([
        (lambda: buttons.emit(make_buttons(stick=0.0)), 0.01),
        (lambda: buttons.emit(make_buttons(stick=1.0)), 0.01),
        (lambda: asked.extend(arm.incoming()), 0.0),
    ])
    drive_scheduler(world.start([dc, driver]), steps=400)

    np.testing.assert_array_equal(asked[0].request.positions, NOMINAL_JOINTS)


def test_every_start_pose_is_a_fresh_per_joint_draw_around_the_nominal():
    draws = [roboarm_command.sampled_joints(NOMINAL_JOINTS, JOINTS_SPREAD).positions for _ in range(64)]
    offsets = (np.array(draws) - NOMINAL_JOINTS) / JOINTS_SPREAD
    assert np.all(np.abs(offsets) < 1)  # inside the spread the nominal allows each joint
    assert np.all(offsets.std(axis=0) > 0)  # a draw of its own each time
    assert np.all(offsets.std(axis=1) > 0)  # each joint drawn on its own, not one offset for the whole vector


def test_data_collection_with_mujoco_robot_gripper(tmp_path):
    sim = MujocoSim('positronic/assets/mujoco/franka_table.xml', loaders=())

    # Virtual time: the sim advances the world clock as physics steps
    with pimm.World(virtual_time=True) as world:
        dc = DataCollectionController(
            operator_position=OperatorPosition.FRONT.value, nominal_joints=sim.initial_joints, output_path=tmp_path
        )

        agent = DsWriterAgent(LocalDatasetWriter)
        agent.add_signal(keys.TARGET_GRIP)
        agent.add_signal(keys.ROBOT_COMMAND, Serializers.robot_command)
        agent.add_signal('controller_positions', controller_positions_serializer)
        agent.add_signal(keys.ROBOT_STATE, Serializers.robot_state)
        agent.add_signal(keys.GRIP)

        world.connect(sim.state, dc.robot_state)
        world.connect(sim.state, agent.inputs[keys.ROBOT_STATE])
        world.connect(dc.robot_commands, sim.commands)
        world.connect(dc.robot_commands, agent.inputs[keys.ROBOT_COMMAND])
        world.connect(dc.target_grip, sim.target_grip)
        world.connect(dc.target_grip, agent.inputs[keys.TARGET_GRIP])
        world.connect(sim.grip, agent.inputs[keys.GRIP])
        world.connect(dc.ds_agent_commands, agent.command)

        ctrl_em_dc = world.pair(dc.controller_positions)
        ctrl_em_agent = world.pair(agent.inputs['controller_positions'])
        buttons_em = world.pair(dc.buttons_receiver)

        def start_episode():
            dc.ds_agent_commands.emit(DsWriterCommand.START(tmp_path))
            buttons_em.emit(make_buttons(trigger=0.5))

        def enable_tracking():
            buttons_em.emit(make_buttons(trigger=0.5, A=True))

        def emit_pose():
            payload = {'left': None, 'right': Transform3D.identity}
            ctrl_em_dc.emit(payload)
            ctrl_em_agent.emit(payload)

        def stop_episode():
            dc.ds_agent_commands.emit(DsWriterCommand.STOP())

        driver = ManualDriver([
            (start_episode, 0.01),
            (enable_tracking, 0.01),
            (emit_pose, 0.02),
            (stop_episode, 0.005),
        ])

        scheduler = world.start([sim, dc, agent, driver])
        drive_scheduler(scheduler, steps=400)

    # Validate dataset contents
    ds = LocalDataset(tmp_path)
    assert len(ds) == 1
    ep = ds[0]
    assert isinstance(ep, Episode)

    expected = {keys.TARGET_GRIP, 'controller_positions.right', keys.JOINTS, keys.JOINT_VEL, keys.EE_POSE, keys.GRIP}
    assert expected.issubset(set(ep.keys()))

    # Robot/gripper signals should have at least one sample
    robot_j = ep[keys.JOINTS]
    grip_sig = ep[keys.GRIP]
    assert len(robot_j) >= 1
    assert len(grip_sig) >= 1

    # Controller pose was identity; verify controller signals reflect that
    rc = ep['controller_positions.right']
    np.testing.assert_allclose(rc[0][0][:3], np.zeros(3))
    np.testing.assert_allclose(rc[0][0][3:], Rotation.identity.as_quat)

    # When tracking is enabled, target pose initially matches current robot state (due to offset calibration)
    # Verify a robot command was emitted (tracking enabled and a pose sent)
    # We don't assert exact equality with state here; just presence and shape.
    cmd_pose = ep[keys.TARGET_EE_POSE]
    assert len(cmd_pose) >= 1 and cmd_pose[0][0].shape == (7,)

    # Basic sanity on sizes and monotonic timestamps
    def assert_strictly_increasing(sig):
        for i in range(1, len(sig)):
            assert sig[i][1] > sig[i - 1][1]

    for name in [keys.JOINTS, keys.JOINT_VEL, keys.GRIP]:
        assert_strictly_increasing(ep[name])


@pytest.mark.parametrize(
    'mode',
    [
        roboarm_command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6),
        roboarm_command.PositionControl(),
    ],
    ids=['impedance', 'position_control'],
)
def test_mujoco_runs_a_command_that_pins_a_control_mode(mode):
    """The actuators run their own law, so what a command pins makes no difference to what it does."""
    sim = MujocoSim('positronic/assets/mujoco/franka_table.xml', loaders=())
    target = np.asarray(sim.initial_ctrl[:7], dtype=np.float64).copy()
    target[0] += 0.1
    cmd = roboarm_command.JointPosition(positions=target, mode=mode)

    with pimm.World(virtual_time=True) as world:
        commands = world.pair(sim.commands)
        driver = ManualDriver([(lambda: commands.emit(cmd), 0.1)])
        scheduler = world.start([sim, driver])
        drive_scheduler(scheduler, steps=50)

    np.testing.assert_allclose(sim.data.ctrl[:7], target)


def test_mujoco_grip_one_is_closed():
    sim = MujocoSim('positronic/assets/mujoco/franka_table.xml', loaders=())

    def finger_qpos() -> float:
        """Physical finger travel: 0 = fingers together (closed), 0.04 m = fully apart (open)."""
        return sim.data.joint('finger_joint1_ph').qpos.item()

    # The pose the scene is built in leaves the gripper open.
    assert finger_qpos() > 0.03

    snapshots = {}
    with pimm.World(virtual_time=True) as world:
        target_grip = world.pair(sim.target_grip)
        grip = world.pair(sim.grip)

        driver = ManualDriver([
            (lambda: target_grip.emit(1.0), 0.5),
            (lambda: snapshots.update(closed=finger_qpos(), closed_grip=grip.read().data), 0.0),
            (lambda: target_grip.emit(0.0), 0.5),
            (lambda: snapshots.update(opened=finger_qpos(), opened_grip=grip.read().data), 0.0),
        ])

        scheduler = world.start([sim, driver])
        drive_scheduler(scheduler, steps=2000)

    assert snapshots['closed'] < 0.005
    assert snapshots['closed_grip'] > 0.9
    assert snapshots['opened'] > 0.035
    assert snapshots['opened_grip'] < 0.1
