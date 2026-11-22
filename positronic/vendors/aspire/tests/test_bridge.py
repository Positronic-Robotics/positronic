import threading
from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition, JointPosition
from positronic.policy.base import PolicyEnd
from positronic.policy.keys import END
from positronic.vendors.aspire.bridge import (
    DIAGNOSTIC_ACCEPTED,
    DIAGNOSTIC_MOVE_END,
    BridgeSettings,
    CameraCalibration,
    CapBridge,
    MoveEnd,
    Program,
)
from positronic.vendors.aspire.contracts import PRIMARY_ARM, EnvironmentAdapter

ARMS = keys.BIMANUAL_ARMS
QUAT = geom.Rotation.Representation.QUAT
WRIST = keys.WRIST_IMAGE.removeprefix(keys.IMAGE_PREFIX)
EXTERIOR = keys.EXTERIOR_IMAGE.removeprefix(keys.IMAGE_PREFIX)


def command(arm: str | None) -> str:
    return keys.arm_channel(keys.ROBOT_COMMAND, arm)


def grip(arm: str | None) -> str:
    return keys.arm_channel(keys.TARGET_GRIP, arm)


class Rig:
    """A robot that reaches each chunk's last target at once, unless a test holds it back."""

    def __init__(self, arms: tuple[str | None, ...] = ARMS, joints: int = 6):
        self.arms = arms
        self.q = {arm: np.full(joints, 0.1 * i) for i, arm in enumerate(arms)}
        self.pose = {arm: geom.Transform3D(np.array([0.3, 0.2 - 0.4 * i, 0.3])) for i, arm in enumerate(arms)}
        self.grip = dict.fromkeys(arms, 0.0)
        # The most each gripper closes: an object in the jaws stops it short.
        self.grip_limit = dict.fromkeys(arms, 1.0)
        # How far short of a pose target each hand stops.
        self.shortfall = np.zeros(3)
        # Arms whose joints creep on and never settle, whatever the command.
        self.creeping: set[str] = set()
        # The camera images, and each depth the rig sends, by observation key.
        self.views: dict[str, np.ndarray] = {keys.EXTERIOR_IMAGE: np.zeros((4, 6, 3), dtype=np.uint8)}

    def obs(self) -> dict:
        obs = {keys.TASK: 'stack the cups', **self.views}
        for arm in self.arms:
            state = keys.arm_channel(keys.ROBOT_STATE, arm)
            obs[f'{state}{keys.JOINTS_SUFFIX}'] = self.q[arm].copy()
            obs[f'{state}{keys.JOINT_VEL_SUFFIX}'] = np.zeros(len(self.q[arm]))
            obs[f'{state}{keys.EE_POSE_SUFFIX}'] = self.pose[arm].as_vector(QUAT)
            obs[keys.arm_channel(keys.GRIP, arm)] = self.grip[arm]
        return obs

    def play(self, chunk: list[dict]) -> None:
        last = chunk[-1]
        for arm in self.arms:
            if arm in self.creeping:
                self.q[arm] = self.q[arm] + 0.01
                continue
            match last[command(arm)]:
                case JointPosition(positions=positions):
                    self.q[arm] = np.asarray(positions, dtype=float)
                case CartesianPosition(pose=pose):
                    self.pose[arm] = geom.Transform3D(pose.translation + self.shortfall, pose.rotation)
            self.grip[arm] = min(last[grip(arm)], self.grip_limit[arm])


def drive(bridge: CapBridge, rig: Rig, rounds: int = 300) -> list[list[dict]]:
    """Answer the rig's observations until the program ends; return every chunk."""
    chunks = []
    for _ in range(rounds):
        chunk = bridge.answer(rig.obs())
        chunks.append(chunk)
        if END in chunk[-1]:
            return chunks
        rig.play(chunk)
    raise AssertionError('the program did not end')


@pytest.fixture
def settings() -> BridgeSettings:
    return BridgeSettings(fps=10.0, round_s=0.2, call_timeout_s=2.0, think_timeout_s=30.0)


@pytest.fixture
def run(settings):
    bridges = []

    def run(
        program: str, rig: Rig | None = None, settings: BridgeSettings = settings
    ) -> tuple[CapBridge, list[list[dict]]]:
        bridge = CapBridge(Program('program.py', dedent(program)), settings)
        bridges.append(bridge)
        return bridge, drive(bridge, rig or Rig())

    yield run
    for bridge in bridges:
        bridge.close()


def completed(bridge: CapBridge) -> None:
    assert bridge.outcome is not None
    assert bridge.outcome.end is PolicyEnd.COMPLETE, bridge.outcome.error


def test_bridge_is_an_aspire_environment_adapter(settings):
    assert isinstance(CapBridge(Program('program.py', 'pass'), settings), EnvironmentAdapter)


def test_two_arm_program_plays_its_targets_and_ends_complete(run):
    bridge, chunks = run("""
        state = get_robot_state()
        assert state.arms == ('left', 'right')
        assert get_task_context().language == 'stack the cups'
        start = state.end_effector_poses['left']
        target = Pose(position=start.position + [0.0, 0.0, 0.1], quaternion_wxyz=start.quaternion_wxyz, frame='world')
        moved = move_to_pose(target, arm='left')
        assert moved.ok and moved.diagnostics['accepted'] and moved.diagnostics['arrived'], moved
        assert abs(get_robot_state().end_effector_poses['left'].position[2] - start.position[2] - 0.1) < 1e-9
        joints = move_to_joints([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], arm='right')
        assert joints.ok and joints.final_errors['right'] == 0.0, joints
        closed = close_gripper(arm='secondary')
        assert closed.ok and get_robot_state().gripper_positions['left'] == 0.0, closed
        both = move_synchronized({'left': {'position': [0.4, 0.1, 0.2], 'rpy_deg': [180, 0, 0], 'frame': 'world'},
                                  'right': [0.0, 0.1, 0.0, 0.0, 0.0, 0.0]})
        assert both.ok and set(both.final_errors) == {'left', 'right'}, both
        home = go_home(arm='right')
        assert home.ok, home
    """)
    completed(bridge)
    assert chunks[-1][-1][END] == PolicyEnd.COMPLETE.value
    poses = [element[command('left')] for chunk in chunks for element in chunk]
    assert any(isinstance(target, CartesianPosition) for target in poses)
    assert chunks[-1][-1][grip('left')] == 1.0
    np.testing.assert_allclose(chunks[-1][-1][command('right')].positions, np.full(6, 0.1))


def test_the_example_program_completes_on_a_compliant_rig(run):
    bridge, _ = run((Path(__file__).parents[1] / 'examples' / 'lift_and_return.py').read_text())
    completed(bridge)


def test_each_answer_holds_at_least_one_round(run, settings):
    _, chunks = run('move_to_joints([0.5, 0, 0, 0, 0, 0], arm="left")')
    assert all(len(chunk) == settings.round_ticks for chunk in chunks[:-1])


@pytest.mark.parametrize(('round_s', 'fps', 'ticks'), [(0.25, 10.0, 3), (1.1, 50.0, 55), (0.01, 10.0, 1)])
def test_a_round_holds_at_least_round_s_and_no_extra_tick(round_s, fps, ticks):
    assert BridgeSettings(fps=fps, round_s=round_s).round_ticks == ticks


def test_a_refused_call_was_not_accepted_and_moves_nothing(run):
    bridge, chunks = run("""
        refused = move_to_pose({'position': [0.3, 0.0, 0.3], 'rpy_deg': [0, 0, 0], 'frame': 'camera'}, arm='left')
        assert not refused.ok and refused.error.code == 'invalid_request', refused
        assert refused.diagnostics == {'accepted': False} and refused.steps_executed == 0, refused
        unknown = open_gripper(arm='middle')
        assert unknown.error.code == 'not_found' and not unknown.diagnostics['accepted'], unknown
    """)
    completed(bridge)
    assert all(isinstance(element[command('left')], JointPosition) for chunk in chunks for element in chunk)


def test_an_arrived_move_outside_the_call_tolerance_did_not_succeed(run):
    rig = Rig()
    rig.shortfall = np.array([0.0, 0.0, 0.005])
    bridge, _ = run(
        """
        start = get_robot_state().end_effector_poses['left']
        loose = move_to_pose(start, tolerance=0.01, arm='left')
        assert loose.ok and loose.diagnostics['arrived'], loose
        tight = move_to_pose(start, tolerance=0.001, arm='left')
        assert tight.diagnostics['accepted'] and tight.diagnostics['arrived'] and not tight.ok, tight
        assert tight.error.code == 'controller_failed' and tight.status == 'controller_failed', tight
        assert abs(tight.final_errors['left'] - 0.005) < 1e-9, tight
    """,
        rig,
    )
    completed(bridge)


def test_a_grasp_that_stalls_on_an_object_ends_as_stopped(run, settings):
    rig = Rig()
    rig.grip_limit['right'] = 0.6
    bridge, _ = run(
        """
        grasp = close_gripper(arm='right')
        assert grasp.diagnostics == {'accepted': True, 'arrived': False, 'move_end': 'stopped'}, grasp
        assert not grasp.ok and grasp.error.code == 'controller_failed', grasp
        assert abs(get_robot_state().gripper_positions['right'] - 0.4) < 1e-9
    """,
        rig,
    )
    completed(bridge)


def test_a_move_that_never_settles_times_out_by_the_chunks_it_answered(run, settings):
    rig = Rig()
    rig.creeping.add('left')
    bridge, chunks = run(
        """
        late = move_to_joints([1, 1, 1, 1, 1, 1], arm='left')
        assert late.error.code == 'timeout' and late.diagnostics['move_end'] == 'timeout', late
        assert late.status == 'timeout' and not late.diagnostics['arrived'], late
    """,
        rig,
    )
    completed(bridge)
    played_s = sum(len(chunk) for chunk in chunks[:-1]) / settings.fps
    assert settings.call_timeout_s <= played_s <= settings.call_timeout_s + 2 * settings.round_s


def test_a_raising_program_gives_up_and_records_its_error(run):
    bridge, chunks = run("""
        open_gripper(arm='left')
        raise ValueError('cup not found')
    """)
    assert bridge.outcome is not None and bridge.outcome.end is PolicyEnd.GIVE_UP
    assert 'ValueError: cup not found' in (bridge.outcome.error or '')
    assert chunks[-1][-1][END] == PolicyEnd.GIVE_UP.value
    assert bridge.answer(Rig().obs())[-1][END] == PolicyEnd.GIVE_UP.value


@pytest.mark.parametrize(
    'program, error',
    [
        ('segment_text("exterior", "cup")', 'NotImplementedError'),
        ('import os', 'ImportError'),
        ('open("/etc/hostname")', 'NameError'),
        ('move_to_joints(', 'SyntaxError'),
    ],
)
def test_a_program_outside_the_contract_gives_up(run, program, error):
    bridge, _ = run(program)
    assert bridge.outcome is not None and bridge.outcome.end is PolicyEnd.GIVE_UP
    assert error in (bridge.outcome.error or '')


def test_a_program_may_import_math(run):
    bridge, _ = run('import math\nassert math.pi > 3')
    completed(bridge)


def test_step_takes_one_period(run, settings):
    bridge, chunks = run("""
        close = ArmCommand(mode='joint_position', target=[0.2] * 6, gripper_position=0.0)
        result = step(RobotAction(arms={'left': close}))
        assert result.ok and result.observation.robot_state.joint_positions['left'][0] == 0.2, result
    """)
    completed(bridge)
    assert chunks[0][0][grip('left')] == 1.0


def test_trajectory_plays_its_waypoints_at_the_chunk_rate(run, settings):
    bridge, chunks = run("""
        state = get_robot_state()
        start = state.joint_positions['right']
        waypoints = [[float(v) + 0.1 * i for v in start] for i in range(5)]
        path = Trajectory(joint_positions=waypoints, dt_s=0.2, joint_names=state.joint_names['right'],
                          planner='test', collision_aware=False, expected_start=state, arm='primary',
                          gripper_positions=[1.0, 1.0, 0.0, 0.0, 0.0])
        result = execute_trajectory(path)
        assert result.ok and result.diagnostics['arrived'], result
        stale = execute_trajectory(path)
        assert stale.error.code == 'stale_plan' and not stale.diagnostics['accepted'], stale
    """)
    completed(bridge)
    played = chunks[0]
    assert len(played) == 10
    np.testing.assert_allclose(
        [element[command('right')].positions[0] for element in played], 0.1 + 0.1 * np.repeat(np.arange(5), 2)
    )
    assert [element[grip('right')] for element in played[::2]] == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_get_observation_carries_the_rgb_views(run):
    bridge, _ = run("""
        view = get_observation('exterior').cameras['exterior']
        assert view.rgb.shape == (4, 6, 3) and view.depth_m is None
    """)
    completed(bridge)


def test_close_stops_a_waiting_program(settings):
    before = set(threading.enumerate())
    bridge = CapBridge(Program('program.py', 'move_to_joints([1, 1, 1, 1, 1, 1], arm="left")'), settings)
    bridge.answer(Rig().obs())
    bridge.close()
    assert bridge.outcome is None
    assert set(threading.enumerate()) <= before


def test_the_trace_records_each_call_with_its_arguments_result_and_end(run, settings):
    bridge, _ = run("""
        get_robot_state()
        close_gripper(arm='left')
        move_to_joints([0.5] * 6, arm='right')
        step(RobotAction(arms={'left': ArmCommand(mode='joint_position', target=[0.0] * 6)}))
        try:
            segment_text('exterior', 'cup')
        except Exception:
            pass
    """)
    completed(bridge)
    calls = bridge.calls
    assert [call.name for call in calls] == [
        'get_robot_state',
        'close_gripper',
        'move_to_joints',
        'step',
        'segment_text',
    ]
    state, grip_call, move, period, unsupported = calls
    assert state.result['gripper_positions'] == {'left': 1.0, 'right': 1.0}
    assert grip_call.kwargs == {'arm': 'left'}
    assert grip_call.result['diagnostics'][DIAGNOSTIC_MOVE_END] == MoveEnd.ARRIVED
    assert move.args == [[0.5] * 6] and move.result['ok'] and move.result['diagnostics'][DIAGNOSTIC_ACCEPTED]
    assert move.result['final_observation']['cameras'] == {}
    assert period.result['diagnostics'][DIAGNOSTIC_MOVE_END] == MoveEnd.PERIOD
    assert unsupported.result is None and 'NotImplementedError' in unsupported.raised
    assert move.ended_robot_s - move.started_robot_s == pytest.approx(settings.round_s)
    assert all(a.ended_wall_s <= b.started_wall_s for a, b in zip(calls, calls[1:], strict=False))


def test_the_trace_records_a_call_that_the_bridge_refused(run):
    bridge, _ = run("move_to_joints([1.0] * 5, arm='left')")
    completed(bridge)
    (call,) = bridge.calls
    assert not call.result['ok'] and not call.result['diagnostics'][DIAGNOSTIC_ACCEPTED]
    assert DIAGNOSTIC_MOVE_END not in call.result['diagnostics']


@pytest.fixture
def one_arm(settings) -> BridgeSettings:
    return replace(settings, arms=keys.ONE_ARM)


def test_the_example_program_completes_on_a_one_arm_rig_with_seven_joints(run, one_arm):
    example = (Path(__file__).parents[1] / 'examples' / 'lift_and_return.py').read_text()
    bridge, chunks = run(example, Rig(keys.ONE_ARM, joints=7), one_arm)
    completed(bridge)
    assert {key for chunk in chunks for element in chunk for key in element} == {command(None), grip(None), END}
    moves = [call for call in bridge.calls if call.name != 'get_robot_state']
    names = ['move_to_pose', 'set_grippers', 'set_grippers', 'move_synchronized', 'go_home']
    assert [call.name for call in moves] == names
    assert all(call.result['diagnostics'][DIAGNOSTIC_MOVE_END] == MoveEnd.ARRIVED for call in moves)


def test_a_one_arm_program_names_its_arm_primary_and_moves_the_joints_the_rig_has(run, one_arm):
    bridge, chunks = run(
        """
        state = get_robot_state()
        assert state.arms == ('primary',) and len(state.joint_positions['primary']) == 7
        assert len(state.joint_names['primary']) == 7
        moved = move_to_joints([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25])
        assert moved.ok and moved.final_errors == {'primary': 0.0}, moved
        short = move_to_joints([0.5] * 6)
        assert short.error.code == 'invalid_request' and not short.diagnostics['accepted'], short
        together = move_synchronized({'primary': [0.5] * 6})
        assert together.error.code == 'invalid_request' and not together.diagnostics['accepted'], together
        other = open_gripper(arm='left')
        assert other.error.code == 'not_found', other
        period = step(RobotAction(arms={'primary': ArmCommand(mode='joint_position', target=[0.1] * 7)}))
        assert period.ok and period.observation.robot_state.joint_positions['primary'][6] == 0.1, period
        wide = step(RobotAction(arms={'primary': ArmCommand(mode='joint_position', target=[0.1] * 6)}))
        assert wide.error.code == 'invalid_request', wide
        start = get_robot_state()
        path = Trajectory(joint_positions=[[0.1] * 7, [0.2] * 7], dt_s=0.2, joint_names=start.joint_names['primary'],
                          planner='test', collision_aware=False, expected_start=start)
        assert execute_trajectory(path).ok
        start = get_robot_state()
        narrow = Trajectory(joint_positions=[[0.2] * 6], dt_s=0.2, joint_names=start.joint_names['primary'],
                            planner='test', collision_aware=False, expected_start=start)
        assert execute_trajectory(narrow).error.code == 'invalid_request'
        assert close_gripper().ok and get_robot_state().gripper_positions['primary'] == 0.0
        assert go_home().ok
    """,
        Rig(keys.ONE_ARM, joints=7),
        one_arm,
    )
    completed(bridge)
    np.testing.assert_allclose(chunks[-1][-1][command(None)].positions, np.zeros(7))
    assert chunks[-1][-1][grip(None)] == 1.0


def test_the_embodiment_of_a_program_is_the_one_the_settings_name(run, one_arm):
    bridge, _ = run(
        """
        assert get_robot_state().embodiment == 'franka'
        command = ArmCommand(mode='joint_position', target=[0.1] * 7)
        assert command.embodiment == 'franka' and step(RobotAction(arms={'primary': command})).ok
        other = ArmCommand(mode='joint_position', target=[0.1] * 7, embodiment='yam_real')
        refused = step(RobotAction(arms={'primary': other}))
        assert refused.error.code == 'invalid_request' and not refused.diagnostics['accepted'], refused
        try:
            RobotAction(arms={'primary': command, 'secondary': other})
            mixed = None
        except ValueError as error:
            mixed = str(error)
        assert 'same embodiment' in mixed, mixed
    """,
        Rig(keys.ONE_ARM, joints=7),
        replace(one_arm, embodiment='franka'),
    )
    completed(bridge)
    assert bridge.embodiment == 'franka'


def test_an_observation_whose_joint_velocities_do_not_match_its_joints_is_refused(one_arm):
    obs = Rig(keys.ONE_ARM, joints=7).obs()
    obs[keys.JOINT_VEL] = np.zeros(6)
    with pytest.raises(ValueError, match='joint_velocities'):
        CapBridge(Program('program.py', 'pass'), one_arm).answer(obs)


def test_settings_refuse_arms_that_do_not_name_each_arm_once():
    with pytest.raises(ValueError, match='arms'):
        BridgeSettings(arms=())
    with pytest.raises(ValueError, match='arms'):
        BridgeSettings(arms=(None, PRIMARY_ARM))


# A wrist camera 0.3 m above the hand of a one-arm rig. It looks down: its optical z is the hand's -z, and its y
# is the hand's -y.
INTRINSICS = ((10.0, 0.0, 3.0), (0.0, 10.0, 1.5), (0.0, 0.0, 1.0))
LOOKS_DOWN = CameraCalibration(intrinsics=INTRINSICS, pose=(0.0, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0), hand=PRIMARY_ARM)
# Depth in millimetres, with no depth at one pixel.
DEPTH_MM = np.full((4, 6, 1), 500, dtype=np.uint16)
DEPTH_MM[:, 3] = 400
DEPTH_MM[0, 0] = 0


def wrist_rig() -> Rig:
    rig = Rig(keys.ONE_ARM, joints=7)
    rig.views[keys.WRIST_IMAGE] = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    rig.views[f'{keys.DEPTH_PREFIX}{WRIST}'] = DEPTH_MM
    return rig


@pytest.fixture
def one_arm_bridge(one_arm):
    """Make a bridge of a one-arm rig that read one observation; its wrist camera is calibrated by default."""
    bridges = []

    def one_arm_bridge(rig: Rig | None = None, **bridge_args) -> CapBridge:
        bridge_args.setdefault('calibration', {WRIST: LOOKS_DOWN})
        bridge = CapBridge(Program('program.py', 'pass'), one_arm, **bridge_args)
        bridges.append(bridge)
        bridge.answer((rig or wrist_rig()).obs())
        return bridge

    yield one_arm_bridge
    for bridge in bridges:
        bridge.close()


def test_a_calibrated_view_carries_its_intrinsics_and_metric_depth(one_arm_bridge):
    cameras = one_arm_bridge().get_observation().cameras
    view = cameras[WRIST]
    np.testing.assert_array_equal(view.intrinsics, INTRINSICS)
    assert view.depth_m.shape == (4, 6) and view.depth_m.dtype == np.float32
    np.testing.assert_allclose(view.depth_m, DEPTH_MM[..., 0] / 1000.0)
    assert view.depth_m[0, 0] == 0.0
    exterior = cameras[EXTERIOR]
    assert exterior.depth_m is None and exterior.intrinsics is None and exterior.camera_pose is None


def test_the_pose_of_a_wrist_camera_follows_the_hand(one_arm_bridge):
    rig = wrist_rig()
    offset = geom.Transform3D(np.array([0.02, -0.05, 0.1]), geom.Rotation.from_euler(np.array([0.3, -0.4, 1.2])))
    wrist = CameraCalibration(intrinsics=INTRINSICS, pose=tuple(offset.as_vector(QUAT)), hand=PRIMARY_ARM)
    fixed = CameraCalibration(intrinsics=INTRINSICS, pose=tuple(offset.as_vector(QUAT)))
    bridge = one_arm_bridge(rig, calibration={WRIST: wrist, EXTERIOR: fixed})
    for hand in (
        rig.pose[None],
        geom.Transform3D(np.array([0.5, -0.1, 0.4]), geom.Rotation.from_euler([2.0, 0.5, -1.0])),
    ):
        rig.pose[None] = hand
        bridge.answer(rig.obs())
        cameras = bridge.get_observation().cameras
        assert cameras[WRIST].camera_pose.frame == 'world'
        np.testing.assert_allclose(cameras[WRIST].camera_pose.as_matrix(), hand.as_matrix @ offset.as_matrix, atol=1e-9)
        np.testing.assert_allclose(cameras[EXTERIOR].camera_pose.as_matrix(), offset.as_matrix, atol=1e-9)


@pytest.mark.parametrize(
    ('depth', 'metres'),
    [
        (np.array([[1250, 0]], dtype=np.uint16), [[1.25, 0.0]]),
        (np.array([[[1250], [0]]], dtype=np.uint16), [[1.25, 0.0]]),
        (np.array([[0.75, np.nan], [np.inf, -1.0]], dtype=np.float32), [[0.75, 0.0], [0.0, 0.0]]),
        (np.array([[0.75, 0.5]], dtype=np.float64), [[0.75, 0.5]]),
    ],
)
def test_depth_reads_as_metres_with_zero_for_no_depth(one_arm_bridge, depth, metres):
    rig = wrist_rig()
    rig.views[keys.WRIST_IMAGE] = np.zeros((*np.shape(metres), 3), dtype=np.uint8)
    rig.views[f'{keys.DEPTH_PREFIX}{WRIST}'] = depth
    view = one_arm_bridge(rig).get_observation().cameras[WRIST]
    assert view.depth_m.dtype == np.float32
    np.testing.assert_allclose(view.depth_m, metres)


@pytest.mark.parametrize('depth', [np.full((4, 6), 128, dtype=np.uint8), np.full((2, 3), 500, dtype=np.uint16)])
def test_a_depth_that_is_not_metric_or_not_the_size_of_its_image_is_refused(one_arm_bridge, depth):
    rig = wrist_rig()
    rig.views[f'{keys.DEPTH_PREFIX}{WRIST}'] = depth
    with pytest.raises(ValueError, match='[Dd]epth'):
        one_arm_bridge(rig)


@pytest.mark.parametrize(
    ('intrinsics', 'pose'),
    [
        (((10.0, 0.0, 2.0), (0.0, 10.0, 1.0), (0.0, 0.0, 2.0)), LOOKS_DOWN.pose),
        (((0.0, 0.0, 2.0), (0.0, 10.0, 1.0), (0.0, 0.0, 1.0)), LOOKS_DOWN.pose),
        (((10.0, 0.0, 2.0), (0.0, 10.0, 1.0)), LOOKS_DOWN.pose),
        (INTRINSICS, (0.0, 0.0, 0.3, 0.0, 2.0, 0.0, 0.0)),
        (INTRINSICS, (0.0, 0.0, 0.3, 1.0, 0.0, 0.0)),
        (INTRINSICS, (float('nan'), 0.0, 0.3, 1.0, 0.0, 0.0, 0.0)),
    ],
)
def test_a_calibration_that_is_not_a_camera_is_refused(intrinsics, pose):
    with pytest.raises(ValueError):
        CameraCalibration(intrinsics=intrinsics, pose=pose)


def test_a_calibration_on_the_hand_of_no_arm_is_refused(settings, one_arm):
    on_left = CameraCalibration(intrinsics=INTRINSICS, pose=LOOKS_DOWN.pose, hand=keys.LEFT_ARM)
    with pytest.raises(ValueError, match='hand'):
        CapBridge(Program('program.py', 'pass'), one_arm, calibration={WRIST: on_left})
    unknown = CameraCalibration(intrinsics=INTRINSICS, pose=LOOKS_DOWN.pose, hand='tertiary')
    with pytest.raises(ValueError, match='hand'):
        CapBridge(Program('program.py', 'pass'), settings, calibration={WRIST: unknown})


def test_a_camera_on_the_primary_hand_of_a_two_arm_rig_follows_the_arm_that_primary_names(settings):
    rig = Rig()
    rig.views[keys.WRIST_IMAGE] = np.zeros((4, 6, 3), dtype=np.uint8)
    bridge = CapBridge(Program('program.py', 'pass'), settings, calibration={WRIST: LOOKS_DOWN})
    try:
        bridge.answer(rig.obs())
        pose = bridge.get_observation().cameras[WRIST].camera_pose
        expected = rig.pose[keys.RIGHT_ARM].as_matrix @ LOOKS_DOWN.transform.as_matrix
        np.testing.assert_allclose(pose.as_matrix(), expected)
    finally:
        bridge.close()
