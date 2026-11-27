import inspect
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from eval_vocabulary.episode import GT_STATE_GRIPPER_CONTACT, GT_STATE_POSE, GT_STATE_SIGNAL

from positronic import geom, keys
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.drivers.roboarm.models import GRIPPER_SITE, parse_ee_frames
from positronic.policy.base import Obs
from positronic.policy.executor import Executor
from positronic.policy.keys import (
    CONTROL,
    GROUND_TRUTH,
    HELPER_FIELDS,
    HELPER_HAND_BACK,
    HELPER_TAKE_OVER,
    LICENSE,
    POLICY_SOURCE,
    TELEOPERATOR,
    TYPE,
)
from positronic.policy.takeover import ControlObservation
from positronic.simulator.expert import DOWN, object_key, wrap, yaw
from positronic.simulator.robolab import expert, yam_spec
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab.expert import BANANA, BOWL, Phase, ReachStatics

LEFT, RIGHT = keys.BIMANUAL_ARMS
# The two arms of a YAM in the rig's world frame, whose grip point is off ``GRIPPER_SITE``, with a wrist that has no
# limit.
_GRIP_POINT = geom.Transform3D(np.array([0.0, 0.04, -0.003]))
_PARK = np.array([0.2, 0.0, 0.2])
_ARMS = {
    LEFT: ReachStatics(
        _GRIP_POINT, np.array([0.3, 0.305, 0.31]), reach=0.52, rest_yaw=0.0, wrist_turn=np.pi, park=_PARK
    ),
    RIGHT: ReachStatics(
        _GRIP_POINT, np.array([0.3, -0.305, 0.31]), reach=0.52, rest_yaw=0.0, wrist_turn=np.pi, park=_PARK
    ),
}
_TABLE = 0.293
# A banana that lies along y, near the right mount, and a bowl near the left mount.
_BANANA = np.array([0.81, -0.116, _TABLE + 0.018, 1.0, 0.0, 0.0, 0.0])
_BOWL = np.array([0.86, 0.153, _TABLE + expert.BOWL_HALF_HEIGHT, 1.0, 0.0, 0.0, 0.0])
_PERIOD_NS = 66_666_667
_HELD_GRIP = 0.6


def _control(source: str = TELEOPERATOR) -> dict:
    """The ``CONTROL`` observation of a step where ``source`` drives."""
    return ControlObservation(source=source, event=None).model_dump(mode='json')


def _truth(banana: np.ndarray, bowl: np.ndarray, contact: bool) -> dict:
    """The ``GROUND_TRUTH`` observation of a scene with ``banana`` and ``bowl``, as the harness gives it."""
    return {
        object_key(BANANA, GT_STATE_POSE): banana.astype(np.float32),
        object_key(BOWL, GT_STATE_POSE): bowl.astype(np.float32),
        object_key(BANANA, GT_STATE_GRIPPER_CONTACT): contact,
        object_key(BOWL, GT_STATE_GRIPPER_CONTACT): False,
    }


def _ee_pose(arm: str) -> str:
    return f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.EE_POSE_SUFFIX}'


def _arm(arm: str, grip_point: geom.Transform3D, grip: float) -> Obs:
    """The observation of ``arm`` with its grip point at ``grip_point``, as the harness gives it."""
    ee_pose = grip_point * _ARMS[arm].grip_point.inv
    return {_ee_pose(arm): ee_pose.as_vector(geom.Rotation.Representation.QUAT), keys.arm_channel(keys.GRIP, arm): grip}


def _down_at(position, turn: float = 0.0) -> geom.Transform3D:
    return geom.Transform3D(np.asarray(position, dtype=np.float64), geom.Rotation.from_euler([0.0, 0.0, turn]) * DOWN)


_PARKED = _down_at(_ARMS[LEFT].mount + _PARK)


def _obs(
    right: geom.Transform3D,
    grip: float = 0.0,
    banana: np.ndarray = _BANANA,
    contact: bool = False,
    left: geom.Transform3D = _PARKED,
    left_grip: float = 0.0,
    bowl: np.ndarray = _BOWL,
    source: str = TELEOPERATOR,
) -> Obs:
    return {
        **_arm(RIGHT, right, grip),
        **_arm(LEFT, left, left_grip),
        GROUND_TRUTH: _truth(banana, bowl, contact),
        CONTROL: _control(source),
    }


class _Run:
    """An expert that a test feeds one observation per control step of RoboLab."""

    def __init__(self, takes_over_at_start: bool = False):
        self.time_ns = 0
        self.runtime = Executor(lambda: self.time_ns, simulated=True, charge_inference_time=False)
        policy = expert.BananaInBowlExpert(_ARMS, license='test', takes_over_at_start=takes_over_at_start)
        self.run = self.runtime.start(policy)

    def element(self, obs: Obs) -> dict:
        """The element of the expert for ``obs``, one control step later."""
        step = self.run.send(obs)
        assert step is not None
        self.time_ns += _PERIOD_NS
        return dict(step.commands)

    def local(self, name: str) -> Any:
        """The local ``name`` of the run of the expert, or ``None`` before the run sets it."""
        return inspect.getgeneratorlocals(self.run).get(name)

    def send(self, obs: Obs) -> dict:
        """The element of the expert one control step after it gets ``obs``. The first command of an expert does not
        move the arm, so a new expert gets ``obs`` twice."""
        if self.local('task') is None:
            self.element(obs)
        return self.element(obs)

    @property
    def phase(self) -> Phase:
        return self.local('task').phase

    def close(self) -> None:
        self.run.close()
        self.runtime.close()


@pytest.fixture
def run():
    made = _Run()
    yield made
    made.close()


def _pose(commands: dict, arm: str = RIGHT) -> geom.Transform3D:
    command = commands[keys.arm_channel(keys.ROBOT_COMMAND, arm)]
    assert isinstance(command, CartesianPosition)
    return command.pose * _ARMS[arm].grip_point


def _grip(commands: dict, arm: str = RIGHT) -> float:
    return commands[keys.arm_channel(keys.TARGET_GRIP, arm)]


def _grasp(banana: np.ndarray = _BANANA) -> geom.Transform3D:
    return _down_at(expert.grasp_point(banana) + [0.0, 0.0, expert.GRASP_HEIGHT], expert.grasp_yaw(banana))


def test_the_expert_states_its_identity_its_license_and_the_frame_of_each_pose_of_each_arm():
    meta = expert.BananaInBowlExpert(_ARMS, license='CC-BY-4.0').meta()

    assert (meta[TYPE], meta[LICENSE]) == (expert.BananaInBowlExpert.WIRE_NAME, 'CC-BY-4.0')
    frames = parse_ee_frames(meta[roboarm_keys.EE_FRAMES])
    poses = {_ee_pose(arm) for arm in _ARMS} | {
        f'{keys.arm_channel(keys.ROBOT_COMMAND, arm)}{keys.POSE_SUFFIX}' for arm in _ARMS
    }
    assert frames.keys() == poses
    assert {frame.site for frame in frames.values()} == {GRIPPER_SITE}


def test_the_expert_of_the_yam_has_each_arm_at_the_mount_that_the_env_server_sends_with_the_statics_it_gets():
    statics = {'grip_point': _GRIP_POINT, 'reach': 0.48, 'rest_yaw': 1.5, 'wrist_turn': 1.7, 'park': _PARK}
    arms = expert.BananaInBowlExpert.for_yam('test', **statics).arms

    assert set(arms) == set(keys.BIMANUAL_ARMS)
    for arm, each in arms.items():
        assert np.allclose(each.mount, np.add(yam_spec.mount_xyz(arm), yam_spec.ROOT_IN_RIG_WORLD))
        assert each.grip_point == _GRIP_POINT and np.array_equal(each.park, _PARK)
        assert (each.reach, each.rest_yaw, each.wrist_turn) == (0.48, 1.5, 1.7)


@pytest.mark.parametrize('wrist_turn', [1.5, 3.2, 100.0])
def test_a_wrist_turn_that_is_not_in_rad_above_pi_over_2_and_not_above_pi_fails(wrist_turn):
    with pytest.raises(ValueError, match='wrist turn'):
        replace(_ARMS[RIGHT], wrist_turn=wrist_turn)


def test_the_arm_whose_mount_is_nearer_the_banana_goes_above_it_with_the_grip_open_and_the_other_arm_parks(run):
    start = _down_at([0.6, -0.2, 0.55])
    commands = run.send(_obs(start, left=_down_at([0.5, 0.3, 0.5])))

    assert run.phase == Phase.ABOVE
    assert _grip(commands) == 0.0 and _grip(commands, LEFT) == 0.0
    point = expert.grasp_point(_BANANA)
    to_goal = point[:2] - start.translation[:2]
    moved = _pose(commands).translation[:2] - start.translation[:2]
    assert np.dot(moved, to_goal) > 0.0
    to_park = _PARKED.translation - [0.5, 0.3, 0.5]
    assert np.dot(_pose(commands, LEFT).translation - [0.5, 0.3, 0.5], to_park) > 0.0


@pytest.mark.parametrize('turn', [0.0, 0.7, -2.0])
def test_the_grip_closes_across_the_long_axis_of_the_banana(turn):
    banana = np.concatenate([_BANANA[:3], geom.Rotation.from_euler([0.0, 0.0, turn]).as_quat])
    closing = geom.Rotation.from_euler([0.0, 0.0, expert.grasp_yaw(banana)]).as_rotation_matrix[:, 0]
    long_axis = geom.Rotation.from_quat(banana[3:])(expert.LONG_AXIS)
    assert abs(np.dot(closing, long_axis)) < 1e-6


def test_above_the_banana_and_aligned_the_expert_goes_down_and_at_the_grasp_pose_it_closes(run):
    above = _grasp() * geom.Transform3D(np.array([0.0, 0.0, -expert.APPROACH_HEIGHT]))
    commands = run.send(_obs(above))
    assert run.phase == Phase.DOWN
    assert _pose(commands).translation[2] < above.translation[2]

    commands = run.send(_obs(_grasp()))
    assert run.phase == Phase.CLOSE and _grip(commands) == 1.0


def test_a_grip_that_closes_late_at_the_grasp_pose_stays_in_close(run):
    phases = []
    for index in range(30):  # 2 s: open for 1 s, then closing
        grip = 0.0 if index < 15 else min(0.05 * (index - 14), 0.25)
        run.send(_obs(_grasp(), grip=grip))
        phases.append(run.phase)
    # The check of the hold comes CLOSE_NS after the grip got past OPEN, at step 17.
    closing = 17 + -(-expert.CLOSE_NS // _PERIOD_NS)
    assert phases[:closing] == [Phase.CLOSE] * closing
    assert phases[closing] != Phase.CLOSE


def test_the_expert_with_no_pose_of_the_banana_in_the_ground_truth_says_to_wire_gt_state(run):
    with pytest.raises(ValueError, match=GT_STATE_SIGNAL):
        run.send({**_obs(_grasp()), GROUND_TRUTH: {'sim_state': np.zeros(3)}})


def test_a_grip_on_the_banana_lifts_it_then_carries_it_toward_the_bowl(run):
    at_grasp = _grasp()
    commands = run.send(_obs(at_grasp, grip=_HELD_GRIP, contact=True))
    assert run.phase == Phase.LIFT and _grip(commands) == 1.0
    assert _pose(commands).translation[2] > at_grasp.translation[2]

    carry_z = _BOWL[2] + expert.BOWL_HALF_HEIGHT + expert.CARRY_CLEAR
    high = _down_at(np.append(at_grasp.translation[:2], carry_z), expert.grasp_yaw(_BANANA))
    lifted = np.append(_BANANA[:2], carry_z - at_grasp.translation[2] + _BANANA[2]).tolist() + [1.0, 0.0, 0.0, 0.0]
    commands = run.send(_obs(high, grip=_HELD_GRIP, banana=np.array(lifted), contact=True))
    assert run.phase == Phase.CARRY and _grip(commands) == 1.0
    moved = _pose(commands).translation[:2] - high.translation[:2]
    assert np.dot(moved, _BOWL[:2] - high.translation[:2]) > 0.0


def test_the_grip_carries_the_banana_with_the_yaw_from_the_heading_of_the_grasp(run):
    def from_heading(pose: geom.Transform3D) -> float:
        away = pose.translation[:2] - _ARMS[RIGHT].mount[:2]
        return wrap(yaw(pose.rotation) - np.arctan2(away[1], away[0]))

    # A bowl within the reach of the arm, so that the carry pose does not tilt.
    bowl = np.array([0.6, 0.05, _BOWL[2], 1.0, 0.0, 0.0, 0.0])
    at_grasp = _grasp()
    carry_z = bowl[2] + expert.BOWL_HALF_HEIGHT + expert.CARRY_CLEAR
    high = geom.Transform3D(np.append(at_grasp.translation[:2], carry_z), at_grasp.rotation)
    lifted = np.concatenate([_BANANA[:2], [carry_z - at_grasp.translation[2] + _BANANA[2]], _BANANA[3:]])
    run.send(_obs(high, grip=_HELD_GRIP, banana=lifted, contact=True, bowl=bowl))
    assert run.phase == Phase.CARRY

    goal = run.local('task')._goal(high, lifted, bowl)
    assert abs(wrap(yaw(goal.rotation) - yaw(high.rotation))) > 0.3
    assert from_heading(goal) == pytest.approx(from_heading(high), abs=1e-6)


def test_a_banana_above_the_bowl_goes_down_and_the_grip_opens_at_the_release_height(run):
    rim = _BOWL[2] + expert.BOWL_HALF_HEIGHT
    banana = np.concatenate([_BOWL[:2], [rim + 0.08], _BANANA[3:]])
    commands = run.send(_obs(_down_at(banana[:3] + [0.0, 0.0, 0.02]), grip=_HELD_GRIP, banana=banana, contact=True))
    assert run.phase == Phase.LOWER and _grip(commands) == 1.0

    release = _down_at(np.append(_BOWL[:2], rim + expert.RELEASE_CLEAR))
    commands = run.send(_obs(release, grip=_HELD_GRIP, banana=banana, contact=True))
    assert run.phase == Phase.OPEN and _grip(commands) == 0.0


def test_after_the_banana_falls_into_the_bowl_the_expert_retreats_up_with_the_grip_open(run):
    rim = _BOWL[2] + expert.BOWL_HALF_HEIGHT
    release = _down_at(np.append(_BOWL[:2], rim + expert.RELEASE_CLEAR))
    in_bowl = np.concatenate([_BOWL[:3], _BANANA[3:]])
    run.send(_obs(release, grip=_HELD_GRIP, banana=in_bowl, contact=True))
    run.time_ns += expert.OPEN_NS

    commands = run.send(_obs(release, grip=0.0, banana=in_bowl))
    assert run.phase == Phase.RETREAT and _grip(commands) == 0.0
    assert _pose(commands).translation[2] > release.translation[2]


# Where a banana rests after the grip opens above the bowl and the banana does not get in: on the table far from the
# bowl, on the table beside the bowl after it falls off the rim, and on the rim.
_OUT_OF_BOWL = {
    'on the table': _BANANA,
    'beside the bowl': np.concatenate([_BOWL[:2] - [0.06, 0.0], _BANANA[2:]]),
    'on the rim': np.concatenate([_BOWL[:2] + [0.03, 0.0], [_BOWL[2] + 0.03], _BANANA[3:]]),
}


@pytest.mark.parametrize('banana', _OUT_OF_BOWL.values(), ids=_OUT_OF_BOWL.keys())
def test_a_banana_that_does_not_get_in_the_bowl_after_the_grip_opens_is_grasped_again(run, banana):
    rim = _BOWL[2] + expert.BOWL_HALF_HEIGHT
    release = _down_at(np.append(_BOWL[:2], rim + expert.RELEASE_CLEAR))
    run.send(_obs(release, grip=_HELD_GRIP, banana=np.concatenate([_BOWL[:3], _BANANA[3:]]), contact=True))
    run.time_ns += expert.OPEN_NS

    run.send(_obs(release, grip=0.0, banana=banana))
    assert run.phase in (Phase.RISE, Phase.ABOVE)


def test_a_new_expert_carries_a_banana_that_the_grip_holds_with_no_new_grasp(run):
    carry_z = _BOWL[2] + expert.BOWL_HALF_HEIGHT + expert.CARRY_CLEAR
    high = _down_at([0.75, 0.0, carry_z])
    banana = np.concatenate([high.translation - [0.0, 0.0, 0.01], _BANANA[3:]])
    commands = run.send(_obs(high, grip=_HELD_GRIP, banana=banana, contact=True))
    assert run.phase == Phase.CARRY and _grip(commands) == 1.0


def test_the_arm_that_holds_the_banana_carries_it_and_the_arm_nearer_the_banana_parks():
    banana = np.concatenate([[0.7, -0.05, 0.4], _BANANA[3:]])
    left = _down_at(banana[:3] + [0.0, 0.0, 0.01])
    obs = _obs(_down_at([0.6, -0.2, 0.55]), banana=banana, contact=True, left=left, left_grip=_HELD_GRIP)
    held = _Run()
    try:
        commands = held.send(obs)
        assert held.local('arm') == LEFT
        assert _grip(commands, LEFT) == 1.0 and _grip(commands, RIGHT) == 0.0
    finally:
        held.close()


def test_an_arm_in_the_way_holds_the_other_arm_in_park(run):
    commands = run.send(_obs(_down_at([0.6, -0.2, 0.55]), left=_down_at(_BANANA[:3] + [0.0, 0.05, 0.1])))
    assert run.phase == Phase.PARK and _grip(commands) == 0.0


def test_past_the_reach_of_the_arm_the_approach_axis_tilts_away_from_the_mount():
    mount = _ARMS[RIGHT].mount
    near = expert.BananaInBowl(_down_at([0.6, -0.2, 0.5]), _ARMS[RIGHT], 0)
    flat = near._tilted(0.0, mount + [0.4, 0.0, 0.0])
    tilted = near._tilted(0.0, mount + [0.65, 0.0, 0.0])
    down_axis = np.array([0.0, 0.0, -1.0])
    assert np.allclose(flat(np.array([0.0, 0.0, 1.0])), down_axis)
    approach = tilted(np.array([0.0, 0.0, 1.0]))
    assert approach[0] > 0.0 and approach[2] < 0.0 and abs(approach[1]) < 1e-9


def test_an_arm_that_stays_still_short_of_the_approach_pose_goes_down_from_where_it_is(run):
    above = _grasp() * geom.Transform3D(np.array([0.0, 0.0, -expert.APPROACH_HEIGHT]))
    short = geom.Transform3D(above.translation - [0.018, 0.0, 0.0], above.rotation)
    run.send(_obs(short))
    assert run.phase == Phase.ABOVE

    for _ in range(expert.STUCK_NS // _PERIOD_NS + 1):
        run.send(_obs(short))
    assert run.phase == Phase.DOWN


def test_an_arm_that_stays_still_above_the_release_height_opens_the_grip(run):
    rim = _BOWL[2] + expert.BOWL_HALF_HEIGHT
    banana = np.concatenate([_BOWL[:2], [rim + 0.08], _BANANA[3:]])
    above = _down_at(banana[:3] + [0.0, 0.0, 0.02])
    run.send(_obs(above, grip=_HELD_GRIP, banana=banana, contact=True))
    assert run.phase == Phase.LOWER

    for _ in range(expert.STUCK_NS // _PERIOD_NS):
        run.send(_obs(above, grip=_HELD_GRIP, banana=banana, contact=True))
    commands = run.send(_obs(above, grip=_HELD_GRIP, banana=banana, contact=True))
    assert run.phase == Phase.OPEN and _grip(commands) == 0.0


@pytest.mark.parametrize(
    ('away', 'phase'), [([0.05, 0.0, expert.APPROACH_HEIGHT], Phase.ABOVE), ([0.0, 0.0, 0.03], Phase.DOWN)]
)
def test_an_arm_that_stays_still_short_of_its_goal_takes_the_other_grasp_rotation(run, away, phase):
    grasp = _grasp()
    short = geom.Transform3D(grasp.translation + away, grasp.rotation)
    run.send(_obs(short))
    assert run.phase == phase
    task = run.local('task')
    before = task._yaw

    for _ in range(expert.STUCK_NS // _PERIOD_NS + 1):
        run.send(_obs(short))
    assert abs(wrap(task._yaw - before)) == pytest.approx(np.pi)


@pytest.mark.parametrize('turned', [False, True])
def test_the_grasp_takes_the_rotation_within_the_wrist_turn_unless_the_arm_took_the_other(turned):
    statics = _ARMS[RIGHT]
    point = expert.grasp_point(_BANANA)
    away = point[:2] - statics.mount[:2]
    across = wrap(expert.grasp_yaw(_BANANA) - np.arctan2(away[1], away[0]))
    limited = replace(statics, rest_yaw=across + np.pi, wrist_turn=1.75)
    task = expert.BananaInBowl(_grasp(), limited, 0)
    task._turned = turned

    task._grasp(_BANANA)
    assert abs(wrap(task._yaw - across)) == pytest.approx(0.0 if turned else np.pi, abs=1e-6)


def _root_poses(state: dict) -> tuple[np.ndarray, np.ndarray]:
    objects = state[robolab_keys.STATE_RIGID_OBJECTS]
    return np.array(objects[BANANA][robolab_keys.STATE_ROOT_POSE]), np.array(
        objects[BOWL][robolab_keys.STATE_ROOT_POSE]
    )


@pytest.mark.parametrize('seed', range(5))
def test_a_layout_moves_the_banana_and_the_bowl_within_the_move_apart_and_turns_the_banana_about_z(seed):
    banana, bowl = _root_poses(expert.layout(np.random.default_rng(seed), 0.05, np.radians(30.0)))

    for pose, scene_file in ((banana, expert.BANANA_AT), (bowl, expert.BOWL_AT)):
        assert np.all(np.abs(pose[:2] - scene_file[0][:2]) <= 0.05)
        assert pose[2] == pytest.approx(scene_file[0][2])
    assert np.linalg.norm(banana[:2] - bowl[:2]) > expert.LAYOUT_APART_M
    np.testing.assert_allclose(bowl[3:], expert.BOWL_AT[1])
    turn = geom.Rotation.from_quat(banana[3:]) * geom.Rotation.from_quat(expert.BANANA_AT[1]).inv
    assert turn.angle <= np.radians(30.0) + 1e-6
    np.testing.assert_allclose(turn(np.array([0.0, 0.0, 1.0])), [0.0, 0.0, 1.0], atol=1e-6)


def test_with_takes_over_at_start_the_expert_presses_take_over_at_its_second_element_and_never_hands_back():
    made = _Run(takes_over_at_start=True)
    start = _down_at([0.6, -0.2, 0.55])
    try:
        fields = [made.element(_obs(start, source=POLICY_SOURCE)) for _ in range(2)]
        fields += [made.element(_obs(start)) for _ in range(60)]
    finally:
        made.close()

    assert fields[0][HELPER_TAKE_OVER] is None
    assert [each[HELPER_TAKE_OVER] for each in fields[1:]] == [fields[1][HELPER_TAKE_OVER]] * (len(fields) - 1)
    assert all(each[HELPER_HAND_BACK] is None for each in fields)


def test_while_another_source_drives_the_expert_commands_the_pose_where_the_arm_is(run):
    start = _down_at([0.6, -0.2, 0.55])

    commands = run.send(_obs(start, source=POLICY_SOURCE))

    assert run.phase == Phase.ABOVE
    np.testing.assert_allclose(_pose(commands).as_matrix, start.as_matrix, atol=1e-9)
    assert commands[HELPER_TAKE_OVER] is None


def test_the_expert_takes_over_from_a_source_that_takes_the_arm_back_to_an_earlier_phase(run):
    above = _grasp() * geom.Transform3D(np.array([0.0, 0.0, -0.03]))
    commands = run.send(_obs(above, source=POLICY_SOURCE))
    assert run.phase == Phase.DOWN and commands[HELPER_TAKE_OVER] is None

    commands = run.send(_obs(_down_at([0.6, -0.2, 0.55]), source=POLICY_SOURCE))
    assert run.phase == Phase.ABOVE
    assert commands[HELPER_TAKE_OVER] == run.time_ns - _PERIOD_NS


def test_after_a_takeover_the_expert_drives_and_hands_back_at_a_later_phase(run):
    above = _grasp() * geom.Transform3D(np.array([0.0, 0.0, -0.03]))
    run.send(_obs(above, source=POLICY_SOURCE))
    run.send(_obs(_down_at([0.6, -0.2, 0.55]), source=POLICY_SOURCE))
    commands = run.send(_obs(_down_at([0.6, -0.2, 0.55])))
    assert run.phase == Phase.ABOVE and commands[HELPER_HAND_BACK] is None

    commands = run.send(_obs(above))
    assert run.phase == Phase.DOWN and commands[HELPER_HAND_BACK] == run.time_ns - _PERIOD_NS


# The env origin of the kinematic scene in the rig's world frame, and the reach of a grip that closes on the banana.
_ORIGIN = np.array([0.3, 0.0, _TABLE - expert.BANANA_AT[0][2] + 0.018])
_GRAB_M = 0.02
_START = {arm: _down_at(statics.mount + [0.2, 0.0, 0.15]) for arm, statics in _ARMS.items()}


class _KinematicScene:
    """A kinematic stand-in of ``BananaInBowlTask``: each arm takes the pose of its command at once, so the test does
    not depend on how a pose is solved. A grip that closes near the grasp point holds the banana. An open grip lets go
    of the banana, which falls into the bowl or onto the table.

    A fake takeover picks the source that drives from the press fields, in the step of the press: Take over gives the
    arm to the expert, and Hand back gives it back to the policy."""

    def __init__(self, seed: int):
        banana, bowl = _root_poses(expert.layout(np.random.default_rng(seed), 0.05, np.radians(30.0)))
        banana[:3] += _ORIGIN
        bowl[:3] += _ORIGIN
        self.banana, self.bowl = banana, bowl
        self.rest_z = self.banana[2]
        self.arms = dict(_START)
        self.grips = dict.fromkeys(_ARMS, 0.0)
        self.held: tuple[str, geom.Transform3D] | None = None  # the arm that holds the banana, and its banana pose
        self.source = POLICY_SOURCE
        self.presses: dict = {HELPER_TAKE_OVER: None, HELPER_HAND_BACK: None}
        self.sources = [self.source]

    def obs(self) -> Obs:
        observed = {arm: self.grips[arm] * (_HELD_GRIP if self.held and self.held[0] == arm else 1.0) for arm in _ARMS}
        return {
            **_arm(RIGHT, self.arms[RIGHT], observed[RIGHT]),
            **_arm(LEFT, self.arms[LEFT], observed[LEFT]),
            GROUND_TRUTH: _truth(self.banana, self.bowl, self.held is not None),
            CONTROL: _control(self.source),
        }

    def select(self, fields: Mapping[str, Any]) -> None:
        pressed = {name: fields[name] != self.presses[name] for name in self.presses}
        self.presses = {name: fields[name] for name in self.presses}
        if self.source == POLICY_SOURCE and pressed[HELPER_TAKE_OVER]:
            self.source = TELEOPERATOR
        elif self.source == TELEOPERATOR and pressed[HELPER_HAND_BACK]:
            self.source = POLICY_SOURCE
        if self.source != self.sources[-1]:
            self.sources.append(self.source)

    def apply(self, commands: dict) -> None:
        for arm, statics in _ARMS.items():
            command = commands[keys.arm_channel(keys.ROBOT_COMMAND, arm)]
            self.arms[arm] = command.pose * statics.grip_point
            self.grips[arm] = 1.0 if commands[keys.arm_channel(keys.TARGET_GRIP, arm)] >= 0.5 else 0.0
        if self.held is not None:
            arm, in_grip = self.held
            if self.grips[arm] == 0.0:
                self._let_go()
            else:
                pose = self.arms[arm] * in_grip
                self.banana = np.concatenate([pose.translation, pose.rotation.as_quat])
        for arm in _ARMS:
            near = np.linalg.norm(self.arms[arm].translation - expert.grasp_point(self.banana)) <= _GRAB_M
            if self.held is None and self.grips[arm] == 1.0 and near:
                banana = geom.Transform3D(self.banana[:3], geom.Rotation.from_quat(self.banana[3:]))
                self.held = arm, self.arms[arm].inv * banana

    def _let_go(self) -> None:
        self.held = None
        inside = np.linalg.norm(self.banana[:2] - self.bowl[:2]) <= expert.IN_BOWL_M
        self.banana[2] = self.bowl[2] if inside else self.rest_z

    def done(self) -> bool:
        return self.held is None and expert.in_bowl(self.banana, self.bowl)


def _drive(scene: _KinematicScene, policy, takes_over_at_start: bool, steps: int = 600) -> int | None:
    """Run ``scene`` for not more than ``steps`` control steps, with ``policy(index, scene)`` as the source and the
    expert as the helper. The step at which the banana rests in the bowl, or None."""
    time_ns = 0
    runtime = Executor(lambda: time_ns, simulated=True, charge_inference_time=False)
    run = runtime.start(expert.BananaInBowlExpert(_ARMS, license='test', takes_over_at_start=takes_over_at_start))
    try:
        for index in range(steps):
            step = run.send(scene.obs())
            assert step is not None
            scene.select(step.commands)
            if scene.source == TELEOPERATOR:
                scene.apply({name: value for name, value in step.commands.items() if name not in HELPER_FIELDS})
            else:
                scene.apply(policy(index, scene))
            time_ns += _PERIOD_NS
            if scene.done():
                return index + 1
        return None
    finally:
        run.close()
        runtime.close()


def _hold(index: int, scene: _KinematicScene) -> dict:
    """A policy that commands each arm where it is, with the grip as it is."""
    return {
        name: value
        for arm in _ARMS
        for name, value in {
            keys.arm_channel(keys.ROBOT_COMMAND, arm): CartesianPosition(scene.arms[arm] * _ARMS[arm].grip_point.inv),
            keys.arm_channel(keys.TARGET_GRIP, arm): scene.grips[arm],
        }.items()
    }


def _pull_back(index: int, scene: _KinematicScene) -> dict:
    """A policy that moves the grip point of each arm back toward its mount and up for 30 steps, and then holds."""
    commands = _hold(index, scene)
    if index < 30:
        for arm in _ARMS:
            pose = scene.arms[arm]
            moved = geom.Transform3D(pose.translation + [-0.005, 0.0, 0.004], pose.rotation)
            commands[keys.arm_channel(keys.ROBOT_COMMAND, arm)] = CartesianPosition(moved * _ARMS[arm].grip_point.inv)
    return commands


@pytest.mark.parametrize('seed', range(5))
def test_the_expert_takes_over_at_the_start_and_puts_the_banana_in_the_bowl_as_in_round_0(seed):
    scene = _KinematicScene(seed)

    assert _drive(scene, _hold, takes_over_at_start=True) is not None
    assert scene.sources == [POLICY_SOURCE, TELEOPERATOR]


@pytest.mark.parametrize('seed', range(5))
def test_the_expert_takes_over_from_a_policy_that_leaves_its_path_hands_back_and_puts_the_banana_in_the_bowl(seed):
    scene = _KinematicScene(seed)

    assert _drive(scene, _pull_back, takes_over_at_start=False) is not None
    assert scene.sources[:4] == [POLICY_SOURCE, TELEOPERATOR, POLICY_SOURCE, TELEOPERATOR]
