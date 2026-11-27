"""The success profiles of MolmoSpaces episodes, on small MuJoCo scenes."""

import math
import types

import mujoco
import numpy as np
import pytest

from positronic.simulator.molmo_spaces import success_profile as sp

# A floor, a wall, a ledge, a ramp, a tray, a cube, a rod, a robot hand and a cabinet with two doors. The hand is a
# mocap palm; turned about y, it is an upright plate. The palm's contacts start 2 mm out and push only on overlap.
_SCENE = """
<mujoco>
  <option timestep="0.002"/>
  <default><geom friction="1 0.005 0.0001"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <body name="wall" pos="0.03 -1 0.3"><geom type="box" size="0.01 0.3 0.3"/></body>
    <body name="ledge" pos="0 1 0.02"><geom type="box" size="0.03 0.03 0.02"/></body>
    <body name="ramp" pos="0 2 0.05" euler="10 0 0"><geom type="box" size="0.2 0.2 0.05"/></body>
    <body name="robot" mocap="true" pos="0 -2 0.5">
      <geom type="box" size="0.03 0.03 0.005" margin="0.002" gap="0.002"/>
    </body>
    <body name="cube" pos="0 0 0.02">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
    <body name="rod" pos="0 -3 0.01">
      <freejoint/>
      <geom type="box" size="0.08 0.01 0.01" mass="0.05"/>
    </body>
    <body name="tray" pos="0.5 0 0.005">
      <freejoint/>
      <geom type="box" size="0.08 0.08 0.005" mass="0.3"/>
      <geom type="box" size="0.005 0.08 0.015" pos="0.075 0 0.02" mass="0.05"/>
      <geom type="box" size="0.005 0.08 0.015" pos="-0.075 0 0.02" mass="0.05"/>
      <geom type="box" size="0.08 0.005 0.015" pos="0 0.075 0.02" mass="0.05"/>
      <geom type="box" size="0.08 0.005 0.015" pos="0 -0.075 0.02" mass="0.05"/>
    </body>
    <body name="cabinet" pos="-1 0 0.3">
      <geom type="box" size="0.2 0.2 0.3"/>
      <body name="door_a" pos="0.21 -0.2 0">
        <joint name="hinge_a" type="hinge" axis="0 0 1" range="0 1.57"/>
        <geom type="box" size="0.005 0.1 0.25" pos="0 0.1 0" contype="0" conaffinity="0"/>
      </body>
      <body name="door_b" pos="0.21 0.2 0">
        <joint name="hinge_b" type="hinge" axis="0 0 1" range="-1.57 0"/>
        <geom type="box" size="0.005 0.1 0.25" pos="0 -0.1 0" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_UPRIGHT = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # the palm turned to a plate, thin along x
_PALM_HALF_THICKNESS = 0.005
_CUBE_HALF = 0.02


class _Scene:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(_SCENE)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def body(self, name: str) -> int:
        return self.model.body(name).id

    def joint(self, name: str) -> int:
        return self.model.joint(name).id

    def place(self, name: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Put the free body ``name`` at ``pos`` and stop it."""
        joint = self.model.body_jntadr[self.body(name)]
        adr, vel = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
        self.data.qpos[adr : adr + 7] = [*pos, *quat]
        self.data.qvel[vel : vel + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def hand(self, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        mocap = self.model.body_mocapid[self.body('robot')]
        self.data.mocap_pos[mocap], self.data.mocap_quat[mocap] = pos, quat
        mujoco.mj_forward(self.model, self.data)

    def settle(self, seconds: float = 0.5) -> None:
        for _ in range(round(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def open_joint(self, name: str, opening: float) -> None:
        joint = self.joint(name)
        low, high = self.model.jnt_range[joint]
        # A negative ``opening`` is past the closed position.
        self.data.qpos[self.model.jnt_qposadr[joint]] = opening * (high - low) * math.copysign(1.0, low + high)
        mujoco.mj_forward(self.model, self.data)

    def touch_side(self, name: str) -> None:
        """Hold the upright palm 1 mm from the -x face of the cube ``name``: a contact with no force.

        The palm's lower edge is 1.5 cm below the cube's center, clear of a tray's base under the cube.
        """
        x, y, z = self.data.xpos[self.body(name)]
        self.hand((x - _CUBE_HALF - _PALM_HALF_THICKNESS - 0.001, y, z + 0.015), _UPRIGHT)

    def touching(self, name: str, other: str) -> bool:
        pair = {self.body(name), self.body(other)}
        roots = self.model.body_rootid[self.model.geom_bodyid]
        return any({roots[c.geom1], roots[c.geom2]} == pair for c in self.data.contact)


@pytest.fixture
def scene() -> _Scene:
    scene = _Scene()
    scene.settle(0.2)
    return scene


def _pick(scene: _Scene) -> sp.Pick:
    return sp.Pick(scene.data, scene.body('cube'), scene.body('robot'))


def test_a_floor_resting_object_rests_with_its_full_weight_on_the_floor(scene):
    facts = _pick(scene).facts({})
    assert facts.scene_support_frac == pytest.approx(1.0, abs=0.02)
    assert facts.scene_supports and not facts.holds


def _carry_cube(scene: _Scene, at) -> None:
    """Rest the cube on the palm with its center at ``at``, and let it settle."""
    scene.hand((at[0], at[1], at[2] - _CUBE_HALF - _PALM_HALF_THICKNESS))
    scene.place('cube', at)
    scene.settle()


def test_a_pick_holds_with_the_object_in_the_hand_well_above_its_start(scene):
    check = _pick(scene)
    _carry_cube(scene, (0.0, -2.0, 0.1))
    facts = check.facts({})
    assert facts.lift_m == pytest.approx(0.08, abs=0.005)
    assert facts.robot_contact and not facts.scene_supports
    assert facts.holds


# A lift of 2.4 cm passes MolmoSpaces' own pick check, which needs 1 cm.
@pytest.mark.parametrize('lift', [0.015, 0.024])
def test_a_pick_lifted_less_than_three_centimetres_does_not_hold(scene, lift):
    check = _pick(scene)
    _carry_cube(scene, (0.0, -2.0, _CUBE_HALF + lift))
    facts = check.facts({})
    assert facts.lift_m == pytest.approx(lift, abs=0.003)
    assert facts.robot_contact and not facts.holds


def test_a_held_pick_that_brushes_a_wall_holds(scene):
    check = _pick(scene)
    _carry_cube(scene, (-0.001, -1.0, 0.1))
    # The cube's +x face touches the wall face at x = 0.02.
    scene.place('cube', (0.0005, -1.0, scene.data.xpos[scene.body('cube')][2]))
    facts = check.facts({})
    assert scene.touching('cube', 'wall')
    assert facts.robot_contact and not facts.scene_supports
    assert facts.holds


def test_a_pick_that_rests_on_a_ledge_above_its_start_does_not_hold(scene):
    check = _pick(scene)
    scene.place('cube', (0.0, 1.0, 0.061))
    scene.settle()
    scene.touch_side('cube')
    facts = check.facts({})
    assert facts.lift_m == pytest.approx(0.04, abs=0.003)
    assert facts.robot_contact and facts.scene_supports
    assert not facts.holds


def _place(scene: _Scene, obj: str = 'cube', max_tilt_rad: float = math.radians(45)) -> sp.Place:
    return sp.Place(scene.data, scene.body(obj), scene.body('tray'), scene.body('robot'), max_tilt_rad)


_TRAY = (0.5, 0.0, 0.005)


def _cube_in_tray(scene: _Scene, tray=_TRAY) -> None:
    scene.place('tray', tray)
    scene.place('cube', (tray[0], tray[1], tray[2] + 0.005 + _CUBE_HALF + 0.001))
    scene.settle()


def test_a_place_holds_with_the_released_object_resting_in_the_receptacle(scene):
    check = _place(scene)
    _cube_in_tray(scene)
    facts = check.facts({})
    assert facts.support_frac == pytest.approx(1.0, abs=0.02)
    assert facts.in_footprint and not facts.robot_contact
    assert facts.holds


def test_a_place_with_the_object_leaning_on_the_rim_from_outside_does_not_hold(scene):
    check = _place(scene, 'rod')
    # The rod's inner end lies on the top of the rim at x = 0.58, and its outer end on the floor.
    angle = math.asin(0.04 / 0.16)
    scene.place(
        'rod', (0.58 + 0.08 * math.cos(angle), 0.0, 0.02 + 0.01), (math.cos(angle / 2), 0, math.sin(angle / 2), 0)
    )
    scene.settle()
    facts = check.facts({})
    assert scene.touching('rod', 'tray')
    assert not facts.in_footprint
    assert not facts.holds


_CONTROL_DT = 0.066  # MolmoSpaces' DROID policy period


class _Task:
    """A MolmoSpaces task stand-in that answers its verdict and its end, and records each call."""

    def __init__(self, success: bool = False, done: bool = False, terminate_upon_success: bool = False):
        self.success, self.done = success, done
        self.config = types.SimpleNamespace(terminate_upon_success=terminate_upon_success)
        self.calls: list[str] = []

    def judge_success(self) -> bool:
        self.calls.append('judge_success')
        return self.success

    def is_done(self) -> bool:
        self.calls.append('is_done')
        return self.done


def test_a_place_released_over_the_edge_of_the_rim_never_succeeds(scene):
    # The cube is let go with its center 1.5 cm past the rim and its bottom 0.2 mm into the rim's top.
    judge = sp.StrictJudge(sp.SuccessProfile.STRICT_V1, _Task(), _place(scene), _CONTROL_DT)
    rim_top = scene.data.xpos[scene.body('tray')][2] + 0.035
    scene.place('cube', (0.595, 0.0, rim_top + _CUBE_HALF - 0.0002))
    first = judge.step({})
    assert scene.touching('cube', 'tray')
    assert first.metrics['strict_v1.in_footprint'] is False
    judged = [first]
    for _ in range(40):
        scene.settle(_CONTROL_DT)
        judged.append(judge.step({}))
    assert not any(j.success for j in judged)
    assert not scene.touching('cube', 'tray') or not judged[-1].metrics['strict_v1.receptacle_supports']


def test_a_place_with_the_object_still_in_the_hand_does_not_hold(scene):
    check = _place(scene)
    _cube_in_tray(scene)
    scene.touch_side('cube')
    facts = check.facts({})
    assert facts.receptacle_supports and facts.in_footprint
    assert facts.robot_contact and not facts.holds


def test_a_place_in_a_receptacle_that_the_robot_lifts_does_not_hold(scene):
    check = _place(scene)
    # The palm carries the tray 5.5 cm up, with the cube in it.
    scene.hand((0.5, 0.0, 0.05))
    _cube_in_tray(scene, (0.5, 0.0, 0.055 + 0.0005))
    facts = check.facts({})
    assert facts.receptacle_supports and facts.in_footprint and not facts.robot_contact
    assert facts.receptacle_lift_m == pytest.approx(0.055, abs=0.003)
    assert not facts.holds


def test_a_place_in_a_receptacle_dragged_far_across_the_floor_holds(scene):
    check = _place(scene)
    _cube_in_tray(scene, (0.68, 0.0, 0.005))
    facts = check.facts({})
    assert np.linalg.norm(facts.receptacle_pose[:2] - [0.5, 0.0]) == pytest.approx(0.18, abs=0.005)
    assert facts.holds


def test_a_place_in_a_receptacle_holds_only_within_the_tilt_limit_of_the_task(scene):
    # The tray starts level, at the height where it then rests on the ramp, which tilts 10 degrees about x.
    tilt = (math.cos(math.radians(5)), math.sin(math.radians(5)), 0.0, 0.0)
    ramp_top = np.array([0.0, 2.0, 0.05])
    up = np.array([0.0, -math.sin(math.radians(10)), math.cos(math.radians(10))])
    scene.place('tray', ramp_top + up * (0.05 + 0.0055))
    within, beyond = _place(scene, max_tilt_rad=math.radians(15)), _place(scene, max_tilt_rad=math.radians(5))
    scene.place('tray', ramp_top + up * (0.05 + 0.0055), tilt)
    scene.place('cube', ramp_top + up * (0.05 + 0.01 + _CUBE_HALF + 0.001), tilt)
    scene.settle()
    facts = within.facts({})
    assert abs(facts.receptacle_lift_m) <= sp.MAX_RECEPTACLE_LIFT_M
    assert facts.receptacle_tilt_rad == pytest.approx(math.radians(10), abs=math.radians(0.5))
    assert facts.holds
    assert not beyond.facts({}).holds


def _next_to(scene: _Scene) -> sp.PlaceNextTo:
    return sp.PlaceNextTo(scene.data, scene.body('cube'), scene.body('tray'), scene.body('robot'))


def test_a_next_to_place_held_beside_the_target_does_not_hold(scene):
    check = _next_to(scene)
    scene.place('cube', (0.62, 0.0, 0.0205))
    scene.settle()
    scene.touch_side('cube')
    facts = check.facts({sp.MOLMO_INFO_SUCCESS: True})
    assert facts.molmo_success and facts.robot_contact
    assert not facts.holds


def test_a_next_to_place_holds_once_released_beside_the_target(scene):
    check = _next_to(scene)
    scene.place('cube', (0.62, 0.0, 0.0205))
    scene.settle()
    assert check.facts({sp.MOLMO_INFO_SUCCESS: True}).holds
    assert not check.facts({sp.MOLMO_INFO_SUCCESS: False}).holds


_STRICT_PROFILES = [sp.SuccessProfile.STRICT_V1, sp.SuccessProfile.STRICT_V2]


@pytest.mark.parametrize('profile', _STRICT_PROFILES)
@pytest.mark.parametrize(('opening', 'holds'), [(0.25, False), (0.35, True)])
def test_an_open_holds_from_thirty_percent_of_the_named_joint(scene, opening, holds, profile):
    check = sp.Open(scene.data, scene.joint('hinge_a'), profile)
    scene.open_joint('hinge_a', opening)
    facts = check.facts({})
    assert facts.opening == pytest.approx(opening)
    assert facts.joint == scene.joint('hinge_a')
    assert facts.holds is holds


@pytest.mark.parametrize('profile', _STRICT_PROFILES)
def test_an_open_of_another_door_of_the_same_cabinet_does_not_hold(scene, profile):
    check = sp.Open(scene.data, scene.joint('hinge_a'), profile)
    scene.open_joint('hinge_b', 0.8)
    facts = check.facts({})
    assert facts.opening == 0.0 and not facts.holds


# The open fraction, per step, of a door beside the named one, whose joint stays shut. MolmoSpaces' open reward
# takes the largest open fraction over every joint of the object, so it passes on this door at the last step.
_KNOCKED_DOOR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0035, 0.0479, 0.1317, 0.1437, 0.124, 0.1219, 0.1842]


def test_a_door_knocked_open_beside_the_named_one_never_succeeds(scene):
    profile = sp.SuccessProfile.STRICT_V1
    judge = sp.StrictJudge(profile, _Task(), sp.Open(scene.data, scene.joint('hinge_a'), profile), _CONTROL_DT)
    judged = []
    for opening in _KNOCKED_DOOR + [_KNOCKED_DOOR[-1]] * 40:
        scene.open_joint('hinge_b', opening)
        judged.append(judge.step({}))
    assert not any(j.success for j in judged)
    assert {j.metrics['strict_v1.opening'] for j in judged} == {0.0}


@pytest.mark.parametrize('profile', _STRICT_PROFILES)
@pytest.mark.parametrize(('opening', 'holds'), [(0.10, False), (0.06, False), (0.04, True), (0.0, True)])
def test_a_close_holds_from_ninety_five_percent_closed(scene, opening, holds, profile):
    check = sp.Close(scene.data, scene.joint('hinge_b'), profile)
    scene.open_joint('hinge_b', opening)
    assert check.facts({}).holds is holds


# A push can take a joint past its closed position, as far as the softness of its limit lets it go.
@pytest.mark.parametrize('joint', ['hinge_a', 'hinge_b'])
@pytest.mark.parametrize('past', [0.06, 0.4])
def test_a_joint_pushed_past_its_closed_position_reads_closed_under_strict_v2(scene, joint, past):
    profile = sp.SuccessProfile.STRICT_V2
    scene.open_joint(joint, -past)
    opened = sp.Open(scene.data, scene.joint(joint), profile).facts({})
    assert opened.opening == pytest.approx(-past)
    assert not opened.holds
    assert sp.Close(scene.data, scene.joint(joint), profile).facts({}).holds


@pytest.mark.parametrize('joint', ['hinge_a', 'hinge_b'])
def test_strict_v1_reads_a_joint_pushed_past_its_closed_position_by_its_distance_from_closed(scene, joint):
    profile = sp.SuccessProfile.STRICT_V1
    scene.open_joint(joint, -0.4)
    opened = sp.Open(scene.data, scene.joint(joint), profile).facts({})
    assert opened.opening == pytest.approx(0.4)
    assert opened.holds
    scene.open_joint(joint, -0.06)
    assert not sp.Close(scene.data, scene.joint(joint), profile).facts({}).holds


@pytest.mark.parametrize(
    ('success', 'done', 'judged', 'calls'),
    [
        (True, False, (True, True), ['judge_success']),
        (False, True, (False, True), ['judge_success', 'is_done']),
        (False, False, (False, False), ['judge_success', 'is_done']),
    ],
)
def test_the_native_profile_takes_the_task_verdict_and_ends_at_the_first_success(success, done, judged, calls):
    # A MolmoSpaces task changes its own state when asked, so the native profile keeps MolmoSpaces' call sequence.
    task = _Task(success, done)
    judge = sp.MolmoNativeJudge(task)
    assert judge.observe({sp.MOLMO_INFO_SUCCESS: success}) == {}
    assert judge.step({sp.MOLMO_INFO_SUCCESS: not success}) == sp.Judged(judged[0], judged[1], {})
    assert task.calls == calls


_AFTER_STEPS = 16  # 1 s at 66 ms


def _run_native_tail(verdicts: list[bool], task: _Task | None = None) -> list[sp.Judged]:
    task = task or _Task()
    judge = sp.MolmoNativeTailJudge(task, _CONTROL_DT)
    judged = []
    for verdict in verdicts:
        task.success = verdict
        judged.append(judge.step({}))
    return judged


def test_the_native_tail_profile_succeeds_at_the_task_verdict_and_records_a_second_more():
    judged = _run_native_tail([False] * 3 + [True] * 40)
    assert [j.success for j in judged].index(True) == 3
    assert [j.done for j in judged].index(True) == 3 + _AFTER_STEPS


def test_a_native_tail_success_stays_after_the_task_verdict_breaks():
    judged = _run_native_tail([True] + [False] * 40)
    assert all(j.success for j in judged)
    assert not any(j.done for j in judged[:_AFTER_STEPS])
    assert judged[_AFTER_STEPS].done


def test_the_task_end_ends_a_native_tail_episode_before_and_after_a_success():
    assert _run_native_tail([False], _Task(done=True)) == [sp.Judged(False, True, {})]
    assert _run_native_tail([True], _Task(done=True)) == [sp.Judged(True, True, {})]


def test_the_native_tail_profile_asks_judge_success_before_is_done_at_each_step():
    task = _Task()
    _run_native_tail([False, True], task)
    assert task.calls == ['judge_success', 'is_done'] * 2


def test_a_native_tail_episode_refuses_a_task_that_ends_at_its_own_native_success():
    with pytest.raises(ValueError, match='terminate_upon_success'):
        sp.MolmoNativeTailJudge(_Task(terminate_upon_success=True), _CONTROL_DT)


class _Scripted:
    """A check whose predicate holds at each step as ``holds`` lists."""

    def __init__(self, holds: list[bool]):
        self._holds = iter(holds)

    def facts(self, info) -> sp.JointFacts:
        return sp.JointFacts(holds=next(self._holds), joint=3, opening=0.5)


def _run(holds: list[bool], task: _Task | None = None) -> list[sp.Judged]:
    judge = sp.StrictJudge(sp.SuccessProfile.STRICT_V1, task or _Task(), _Scripted(holds), _CONTROL_DT)
    return [judge.step({}) for _ in holds]


_HOLD_STEPS = 16  # 1 s at 66 ms


def test_a_predicate_held_one_step_short_of_a_second_is_no_success():
    judged = _run([True] * (_HOLD_STEPS - 1) + [False] * 40)
    assert not any(j.success or j.done for j in judged)


def test_a_predicate_held_for_a_second_succeeds_and_the_episode_records_a_second_more():
    judged = _run([True] * 60)
    first = next(i for i, j in enumerate(judged) if j.success)
    assert first == _HOLD_STEPS - 1
    assert all(j.success for j in judged[first:])
    ends = [i for i, j in enumerate(judged) if j.done]
    assert ends[0] == first + _AFTER_STEPS


def test_a_success_stays_while_the_episode_records_after_the_predicate_breaks():
    judged = _run([True] * _HOLD_STEPS + [False] * 40)
    assert judged[_HOLD_STEPS - 1].success
    assert all(j.success for j in judged[_HOLD_STEPS - 1 :])
    assert judged[_HOLD_STEPS - 1 + _AFTER_STEPS].done


def test_one_step_where_the_predicate_breaks_starts_the_hold_window_again():
    judged = _run([True] * (_HOLD_STEPS - 1) + [False] + [True] * _HOLD_STEPS)
    assert not any(j.success for j in judged[:-1])
    assert judged[-1].success


def test_the_task_end_ends_the_episode_before_and_after_a_success():
    assert _run([True] * 3, _Task(done=True))[0] == (False, True, _run([True])[0].metrics)
    judged = _run([True] * (_HOLD_STEPS + 2), _Task(done=True))
    assert judged[-1].success and judged[-1].done


@pytest.mark.parametrize('profile', _STRICT_PROFILES)
def test_a_strict_episode_records_each_fact_under_its_profile_with_its_hold_count_and_verdict(profile):
    judge = sp.StrictJudge(profile, _Task(), _Scripted([True, True]), _CONTROL_DT)
    assert judge.observe({}) == {
        f'{profile}.holds': True,
        f'{profile}.joint': 3,
        f'{profile}.opening': 0.5,
        f'{profile}.held_steps': 0,
        f'{profile}.success': False,
    }
    assert judge.step({}).metrics[f'{profile}.held_steps'] == 1


def test_the_native_task_verdict_is_not_asked_by_a_strict_episode():
    task = _Task(success=True)
    sp.StrictJudge(sp.SuccessProfile.STRICT_V1, task, _Scripted([True]), _CONTROL_DT).step({})
    assert task.calls == ['is_done']


def test_a_strict_episode_refuses_a_task_that_ends_at_its_own_native_success():
    with pytest.raises(ValueError, match='terminate_upon_success'):
        sp.StrictJudge(sp.SuccessProfile.STRICT_V1, _Task(terminate_upon_success=True), _Scripted([]), _CONTROL_DT)
