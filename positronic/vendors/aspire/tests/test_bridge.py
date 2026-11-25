import math
import threading
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition, JointPosition
from positronic.policy.base import PolicyEnd
from positronic.policy.keys import END
from positronic.vendors.aspire.bridge import BridgeSettings, CapBridge
from positronic.vendors.aspire.contracts import EnvironmentAdapter, Pose, RobotState, SynchronizedTrajectory, Trajectory

ARMS = keys.BIMANUAL_ARMS
QUAT = geom.Rotation.Representation.QUAT


def command(arm: str) -> str:
    return keys.arm_channel(keys.ROBOT_COMMAND, arm)


def grip(arm: str) -> str:
    return keys.arm_channel(keys.TARGET_GRIP, arm)


class Rig:
    """A bimanual robot that reaches each chunk's last target at once, unless a test holds it back."""

    def __init__(self):
        self.q = {arm: np.full(6, 0.1 * i) for i, arm in enumerate(ARMS)}
        self.pose = {arm: geom.Transform3D(np.array([0.3, 0.2 - 0.4 * i, 0.3])) for i, arm in enumerate(ARMS)}
        self.grip = dict.fromkeys(ARMS, 0.0)
        # The most each gripper closes: an object in the jaws stops it short.
        self.grip_limit = dict.fromkeys(ARMS, 1.0)
        # How far short of a pose target each hand stops.
        self.shortfall = np.zeros(3)
        # Arms whose joints creep on and never settle, whatever the command.
        self.creeping: set[str] = set()

    def obs(self) -> dict:
        obs = {keys.TASK: 'stack the cups', keys.EXTERIOR_IMAGE: np.zeros((4, 6, 3), dtype=np.uint8)}
        for arm in ARMS:
            state = keys.arm_channel(keys.ROBOT_STATE, arm)
            obs[f'{state}{keys.JOINTS_SUFFIX}'] = self.q[arm].copy()
            obs[f'{state}{keys.JOINT_VEL_SUFFIX}'] = np.zeros(6)
            obs[f'{state}{keys.EE_POSE_SUFFIX}'] = self.pose[arm].as_vector(QUAT)
            obs[keys.arm_channel(keys.GRIP, arm)] = self.grip[arm]
        return obs

    def play(self, chunk: list[dict]) -> None:
        last = chunk[-1]
        for arm in ARMS:
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

    def run(program: str, rig: Rig | None = None) -> tuple[CapBridge, list[list[dict]]]:
        bridge = CapBridge(compile(dedent(program), 'program.py', 'exec'), settings)
        bridges.append(bridge)
        return bridge, drive(bridge, rig or Rig())

    yield run
    for bridge in bridges:
        bridge.close()


def completed(bridge: CapBridge) -> None:
    assert bridge.outcome is not None
    assert bridge.outcome.end is PolicyEnd.COMPLETE, bridge.outcome.error


def test_bridge_is_an_aspire_environment_adapter(settings):
    assert isinstance(CapBridge(compile('pass', 'program.py', 'exec'), settings), EnvironmentAdapter)


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


MOTION_THRESHOLDS = (
    'position_tolerance_m',
    'rotation_tolerance_rad',
    'joint_tolerance_rad',
    'grip_tolerance',
    'still_joint_rad',
    'still_grip',
)


@pytest.mark.parametrize('name', MOTION_THRESHOLDS)
@pytest.mark.parametrize('value', [0.0, -0.01, math.nan, math.inf])
def test_settings_refuse_a_motion_threshold_that_is_not_finite_and_positive(name, value):
    with pytest.raises(ValueError, match=name):
        BridgeSettings(**{name: value})


@pytest.mark.parametrize('name', MOTION_THRESHOLDS)
def test_settings_take_the_smallest_positive_motion_threshold(name):
    assert getattr(BridgeSettings(**{name: math.ulp(0.0)}), name) == math.ulp(0.0)


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


def test_a_trajectory_waits_for_its_final_gripper_target(run):
    rig = Rig()
    rig.grip_limit['right'] = 0.6
    bridge, _ = run(
        """
        state = get_robot_state()
        start = [float(v) for v in state.joint_positions['right']]
        path = Trajectory(joint_positions=[start, [v + 0.1 for v in start]], dt_s=0.2,
                          joint_names=state.joint_names['right'], planner='test', collision_aware=False,
                          expected_start=state, arm='primary', gripper_positions=[1.0, 0.0])
        result = execute_trajectory(path)
        assert result.diagnostics['move_end'] == 'stopped' and not result.diagnostics['arrived'], result
        assert abs(result.final_observation.robot_state.gripper_positions['right'] - 0.4) < 1e-9, result
    """,
        rig,
    )
    completed(bridge)


def test_a_trajectory_with_joint_names_other_than_the_measured_ones_is_refused(run):
    bridge, chunks = run("""
        state = get_robot_state()
        start = state.joint_positions['right']
        names = list(reversed(state.joint_names['right']))
        one = Trajectory(joint_positions=[start], dt_s=0.2, joint_names=names, planner='test', collision_aware=False,
                         expected_start=state, arm='primary')
        single = execute_trajectory(one)
        assert single.error.code == 'invalid_request' and not single.diagnostics['accepted'], single
        both = SynchronizedTrajectory(joint_positions={'left': [state.joint_positions['left']], 'right': [start]},
                                      dt_s=0.2, joint_names={'left': state.joint_names['left'], 'right': names},
                                      planner='test', collision_aware=False, expected_start=state)
        synchronized = execute_trajectory(both)
        assert synchronized.error.code == 'invalid_request' and not synchronized.diagnostics['accepted'], synchronized
    """)
    completed(bridge)
    assert all(isinstance(element[command('right')], JointPosition) for chunk in chunks for element in chunk)


def a_trajectory(kind: type, dt_s: float) -> Trajectory | SynchronizedTrajectory:
    pose = Pose(position=np.zeros(3), quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]), frame='world')
    state = RobotState({'right': np.zeros(6)}, {'right': np.zeros(6)}, {'right': pose}, {'right': 1.0}, 'world')
    common = {'dt_s': dt_s, 'planner': 'test', 'collision_aware': False, 'expected_start': state}
    names = state.joint_names['right']
    if kind is Trajectory:
        return Trajectory(joint_positions=[np.zeros(6)], joint_names=names, arm='right', **common)
    return SynchronizedTrajectory(joint_positions={'right': [np.zeros(6)]}, joint_names={'right': names}, **common)


@pytest.mark.parametrize('kind', [Trajectory, SynchronizedTrajectory])
@pytest.mark.parametrize('dt_s', [0.0, -0.2, math.nan, math.inf])
def test_a_trajectory_refuses_a_period_that_is_not_finite_and_positive(kind, dt_s):
    with pytest.raises(ValueError, match='dt_s'):
        a_trajectory(kind, dt_s)


@pytest.mark.parametrize('kind', [Trajectory, SynchronizedTrajectory])
def test_a_trajectory_takes_the_smallest_positive_period(kind):
    assert a_trajectory(kind, math.ulp(0.0)).dt_s == math.ulp(0.0)


def test_get_observation_carries_the_rgb_views(run):
    bridge, _ = run("""
        view = get_observation('exterior').cameras['exterior']
        assert view.rgb.shape == (4, 6, 3) and view.depth_m is None
    """)
    completed(bridge)


def test_close_stops_a_waiting_program(settings):
    before = set(threading.enumerate())
    bridge = CapBridge(compile('move_to_joints([1, 1, 1, 1, 1, 1], arm="left")', 'program.py', 'exec'), settings)
    bridge.answer(Rig().obs())
    bridge.close()
    assert bridge.outcome is None
    assert set(threading.enumerate()) <= before
