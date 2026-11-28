import sys
import types

import configuronic as cfn
import numpy as np
import pytest

import pimm
import positronic.cfg.embodiment as embodiment_cfg
from positronic import keys
from positronic.drivers.roboarm import RobotStatus, command
from positronic.drivers.roboarm.yam import driver as yam_driver
from positronic.drivers.webxr import LEFT, RIGHT, ControllerButtons
from positronic.eval import keys as eval_keys
from positronic.geom import Rotation, Transform3D
from positronic.teleop.clutch import CONTROLLER_AXES, OperatorPosition
from positronic.teleop.debug import Arm, ArmControl, Panel, commands, debug, yam_bimanual_arms
from positronic.teleop.service import Button
from positronic.tests.testing_coutils import ManualDriver, RecordingEmitter, drive_scheduler

START_JOINTS = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
ROBOT_POSE = Transform3D(np.array([0.4, 0.0, 0.3]), Rotation.identity)
SILENCE_SEC = 0.5
HAND = Transform3D(np.array([0.1, 1.0, -0.3]), Rotation.identity)


def _state(status: RobotStatus = RobotStatus.AVAILABLE) -> types.SimpleNamespace:
    return types.SimpleNamespace(ee_pose=ROBOT_POSE, status=status)


def _values(trigger: float = 0.0, a: float = 0.0, b: float = 0.0, stick: float = 0.0) -> tuple[float, ...]:
    """The button values of a controller: A and B are X and Y on the left controller."""
    values = [0.0] * (max(Button) + 1)
    values[Button.TRIGGER], values[Button.A], values[Button.B], values[Button.STICK] = trigger, a, b, stick
    return tuple(values)


def _raised(pose: Transform3D, dz: float) -> Transform3D:
    return Transform3D(pose.translation + np.array([0.0, 0.0, dz]), pose.rotation)


@pytest.fixture
def world():
    with pimm.World(virtual_time=True) as w:
        yield w


class _Rig:
    """An ``ArmControl`` with each of its ports paired, and recorders on what it emits."""

    def __init__(self, world: pimm.World, side: str = RIGHT):
        self.control = ArmControl(side, START_JOINTS, OperatorPosition.FRONT, CONTROLLER_AXES, SILENCE_SEC)
        self.poses = world.pair(self.control.controller_positions)
        self.buttons = world.pair(self.control.buttons)
        self.state = world.pair(self.control.robot_state)
        self.sync_move = world.pair(self.control.sync_move)
        self.commands, self.grips, self.reports = RecordingEmitter(), RecordingEmitter(), RecordingEmitter()
        self.control.robot_commands._bind(self.commands, clock=world.clock)
        self.control.target_grip._bind(self.grips, clock=world.clock)
        self.control.report._bind(self.reports, clock=world.clock)
        self.calls: list = []
        # The buttons that the page sent last for each controller
        self._sent = {side: ControllerButtons(_values(), (0,) * len(_values())) for side in (LEFT, RIGHT)}

    def take_calls(self) -> None:
        self.calls.extend(self.sync_move.incoming())

    def hand(self, pose: Transform3D | None, side: str = RIGHT) -> None:
        self.poses.emit({LEFT: None, RIGHT: None, side: pose})

    def press(self, side: str = RIGHT, more: dict[Button, int] | None = None, **values: float) -> None:
        """Send the buttons of the controller on ``side`` with ``values``. Count a press of each button that goes
        down, as the page does, and ``more`` presses that came and went between two samples."""
        sent, now = self._sent[side], _values(**values)
        counts = [
            count + int(value >= 0.5 > before)
            for count, value, before in zip(sent.presses, now, sent.values, strict=True)
        ]
        for button, count in (more or {}).items():
            counts[button] += count
        self._sent[side] = ControllerButtons(now, tuple(counts))
        self.buttons.emit({LEFT: None, RIGHT: None, side: self._sent[side]})

    def events(self) -> list[str]:
        """The texts of the events in the last report, with no time."""
        return [event.split(' ', 1)[1] for event in self.reports.emitted[-1][1].events]

    def control_on(self) -> list:
        """The steps that turn control on with the hand at ``HAND``."""
        return [
            (lambda: self.state.emit(_state()), 0.01),
            (lambda: self.hand(HAND), 0.01),
            (lambda: self.press(), 0.01),
            (lambda: self.press(a=1.0), 0.01),
        ]

    def states(self, count: int = ArmControl.SETTLE_STATES) -> list:
        """The steps that send ``count`` new states of the arm."""
        return [(lambda: self.state.emit(_state()), 0.01)] * count


def test_control_moves_the_arm_from_where_it_came_on_and_a_second_press_turns_it_off(world):
    rig = _Rig(world)
    marks = {}
    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
        (lambda: marks.update(on=len(rig.commands.emitted)), 0.01),
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(a=1.0), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.10)), 0.01),
        (lambda: marks.update(off=len(rig.commands.emitted)), 0.0),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    # The press that turns control off replaces the target the driver has not read with the last pose of the arm
    assert marks['on'] > 0 and marks['off'] == marks['on'] + 1
    shift = rig.commands.emitted[marks['on'] - 1][1].pose.translation - ROBOT_POSE.translation
    assert np.linalg.norm(shift) == pytest.approx(0.05)
    np.testing.assert_allclose(rig.commands.emitted[-1][1].pose.translation, ROBOT_POSE.translation)
    assert rig.events()[-2:] == ['control on', 'control off']


def test_two_presses_of_a_between_two_samples_leave_control_off_when_it_was_off(world):
    rig = _Rig(world)
    driver = ManualDriver([
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.hand(HAND), 0.01),
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(more={Button.A: 2}), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert not rig.commands.emitted
    assert rig.events()[-2:] == ['control on', 'control off']


def test_two_presses_of_a_between_two_samples_turn_control_on_again_from_the_new_place_of_the_hand(world):
    rig = _Rig(world)
    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
        (lambda: rig.press(more={Button.A: 2}), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.08)), 0.01),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    shift = rig.commands.emitted[-1][1].pose.translation - ROBOT_POSE.translation
    assert np.linalg.norm(shift) == pytest.approx(0.03)
    assert rig.events()[-3:] == ['control on', 'control off', 'control on']


def test_a_lost_controller_turns_control_off_and_its_return_does_not_move_the_arm(world):
    rig = _Rig(world)
    marks = {}
    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
        (lambda: marks.update(on=len(rig.commands.emitted)), 0.01),
        (lambda: rig.poses.emit({LEFT: None, RIGHT: None}), 0.01),
        (lambda: rig.buttons.emit({LEFT: None, RIGHT: None}), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.10)), 0.01),
        (lambda: rig.press(), 0.01),
        (lambda: marks.update(back=len(rig.commands.emitted)), 0.0),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert marks['on'] > 0 and marks['back'] == marks['on'] + 1, 'one hold of the unread target, then nothing'
    np.testing.assert_allclose(rig.commands.emitted[-1][1].pose.translation, ROBOT_POSE.translation)
    assert rig.events()[-1] == 'control off: the controller is not tracked'


def test_the_stick_turns_control_off_clears_the_errors_and_then_puts_the_arm_at_its_start_pose(world):
    rig = _Rig(world)
    # The handlers of the arm and of its gripper
    readies = [world.pair(rig.control.ready['0']), world.pair(rig.control.ready['1'])]
    cleared: list = []
    marks = {}

    def clear() -> None:
        for ready in readies:
            cleared.extend(ready.incoming())

    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(stick=1.0), 0.01),
        (clear, 0.01),
        (lambda: rig.take_calls(), 0.01),
        (lambda: marks.update(moves_before_clear=len(rig.calls)), 0.01),
        (lambda: [call.set_result(None) for call in cleared], 0.01),
        *rig.states(),
        (lambda: rig.take_calls(), 0.01),
        (lambda: rig.calls[0].set_result(None), 0.01),
        (lambda: marks.update(at_start=len(rig.commands.emitted)), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
        (lambda: marks.update(after=len(rig.commands.emitted)), 0.0),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert len(cleared) == 2 and marks['moves_before_clear'] == 0 and len(rig.calls) == 1
    move = rig.calls[0].request
    assert isinstance(move, command.JointPosition)
    np.testing.assert_allclose(move.positions, START_JOINTS)
    assert rig.events()[-2:] == ['to the start pose', 'at the start pose']
    assert marks['after'] == marks['at_start'], 'control is off at the start pose'


def test_a_press_of_the_stick_and_of_a_in_one_sample_moves_the_arm_to_its_start_pose_with_control_off(world):
    rig = _Rig(world)
    driver = ManualDriver([
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.hand(HAND), 0.01),
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(more={Button.A: 1, Button.STICK: 1}), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert not rig.commands.emitted
    assert 'control on' not in rig.events()
    assert 'to the start pose' in rig.events()


def test_presses_made_during_the_start_move_do_nothing_after_it(world):
    rig = _Rig(world)
    readies = [world.pair(rig.control.ready['0']), world.pair(rig.control.ready['1'])]
    cleared: list = []

    def clear() -> None:
        for ready in readies:
            cleared.extend(ready.incoming())

    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(stick=1.0), 0.01),
        (clear, 0.01),
        (lambda: [call.set_result(None) for call in cleared], 0.01),
        *rig.states(),
        (lambda: rig.take_calls(), 0.01),
        (lambda: rig.press(more={Button.A: 1, Button.STICK: 1}), 0.01),
        (lambda: rig.calls[0].set_result(None), 0.01),
        (lambda: rig.press(), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    events = rig.events()
    after = events[events.index('at the start pose') + 1 :]
    assert 'control on' not in after and 'to the start pose' not in after
    assert len(rig.calls) == 1


def test_the_start_move_waits_for_the_settle_states_of_the_arm_after_the_press(world):
    rig = _Rig(world)
    marks = {}
    driver = ManualDriver([
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(stick=1.0), 0.01),
        *rig.states(ArmControl.SETTLE_STATES - 1),
        (lambda: rig.take_calls(), 0.01),
        (lambda: marks.update(before_state=len(rig.calls)), 0.01),
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.take_calls(), 0.0),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert marks['before_state'] == 0 and len(rig.calls) == 1


def test_the_start_move_fails_with_no_move_when_the_arm_sends_no_state(world):
    rig = _Rig(world)
    driver = ManualDriver([
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(stick=1.0), ArmControl.SETTLE_SEC + 0.1),
        (lambda: rig.take_calls(), 0.0),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=1000)

    assert not rig.calls
    assert rig.events()[-1] == f'the start pose failed: the arm sent 0 new states in {ArmControl.SETTLE_SEC} s'


def test_the_grip_goes_to_the_gripper_once_for_each_sample_of_the_buttons_while_control_is_on(world):
    rig = _Rig(world)
    driver = ManualDriver([*rig.control_on(), (lambda: rig.press(trigger=0.6), 0.05)])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert [grip for _, grip in rig.grips.emitted] == [0.0, pytest.approx(0.6)]


def test_the_trigger_moves_no_gripper_while_control_is_off_or_the_arm_is_in_an_error(world):
    rig = _Rig(world)
    marks = {}
    driver = ManualDriver([
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.hand(HAND), 0.01),
        (lambda: rig.press(trigger=0.6), 0.01),
        (lambda: marks.update(off=len(rig.grips.emitted)), 0.01),
        (lambda: rig.press(a=1.0), 0.01),
        (lambda: rig.state.emit(_state(RobotStatus.ERROR)), 0.01),
        (lambda: marks.update(on=len(rig.grips.emitted)), 0.01),
        (lambda: rig.press(trigger=0.9), 0.01),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert marks['off'] == 0 and marks['on'] == 1
    assert len(rig.grips.emitted) == marks['on']


def test_a_start_pose_that_the_arm_refuses_is_an_event_and_the_teleoperation_goes_on(world):
    rig = _Rig(world)
    driver = ManualDriver([
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(stick=1.0), 0.01),
        *rig.states(),
        (lambda: rig.take_calls(), 0.01),
        (lambda: rig.calls[0].set_exception(RuntimeError('out of reach')), 0.01),
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.hand(HAND), 0.01),
        (lambda: rig.press(a=1.0), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert rig.events()[-2:] == ['the start pose failed: out of reach', 'control on']
    assert isinstance(rig.commands.emitted[-1][1], command.CartesianPosition)


def test_an_error_turns_control_off_so_the_arm_stays_when_the_error_clears(world):
    rig = _Rig(world)
    marks = {}
    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.state.emit(_state(RobotStatus.ERROR)), 0.01),
        (lambda: marks.update(error=len(rig.commands.emitted)), 0.01),
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
        (lambda: marks.update(after=len(rig.commands.emitted)), 0.0),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert marks['after'] == marks['error']
    assert rig.events()[-2:] == [
        'the arm reports an error: a press of the stick clears it',
        'control off: the arm is not available',
    ]


def test_control_goes_off_when_the_headset_sends_nothing_for_a_while(world):
    rig = _Rig(world)
    marks = {}
    driver = ManualDriver([
        *rig.control_on(),
        (lambda: marks.update(on=len(rig.commands.emitted)), 0.01),
        # The arm keeps sending its state while the headset sends nothing
        *[(lambda: rig.state.emit(_state()), 0.05)] * round((SILENCE_SEC + 0.1) / 0.05),
        (lambda: rig.hand(_raised(HAND, 0.10)), 0.01),
        (lambda: marks.update(after=len(rig.commands.emitted)), 0.0),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=1000)

    assert marks['after'] == marks['on']
    assert 'control off: the headset sends no sample' in rig.events()


def test_control_goes_off_when_the_arm_sends_no_new_state_for_a_while(world):
    rig = _Rig(world)
    marks = {}
    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.hand(_raised(HAND, 0.02)), 0.01),
        (lambda: marks.update(on=len(rig.commands.emitted)), 0.01),
        # The driver stalls: its last state stays, and the hand keeps moving
        *[(lambda: rig.hand(_raised(HAND, 0.03)), 0.05)] * round(2 * ArmControl.STALE_STATE_SEC / 0.05),
        (lambda: marks.update(stalled=len(rig.commands.emitted)), 0.01),
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.10)), 0.01),
        (lambda: marks.update(after=len(rig.commands.emitted)), 0.0),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=2000)

    assert marks['on'] > 0
    assert marks['after'] == marks['stalled'], 'control stays off when the driver sends states again'
    assert 'control off: the arm sends no state' in rig.events()
    # The target that the stalled driver has not read is replaced by the last pose the arm sent
    np.testing.assert_allclose(rig.commands.emitted[-1][1].pose.translation, ROBOT_POSE.translation)


def test_a_driver_that_stalls_while_the_headset_is_silent_takes_no_target_from_before_the_stall(world):
    rig = _Rig(world)
    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
        # The headset goes silent, and the driver stalls before it read the last target: control goes off for the
        # headset first
        *rig.states(ArmControl.SETTLE_STATES - 1),
        (lambda: None, 2 * ArmControl.STALE_STATE_SEC),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=2000)

    assert 'control off: the headset sends no sample' in rig.events()
    np.testing.assert_allclose(rig.commands.emitted[-1][1].pose.translation, ROBOT_POSE.translation)


def test_a_driver_that_stalls_after_it_read_the_last_target_gets_no_new_target(world):
    rig = _Rig(world)
    marks = {}
    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
        *rig.states(),
        (lambda: marks.update(before=len(rig.commands.emitted)), 2 * ArmControl.STALE_STATE_SEC),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=2000)

    assert marks['before'] > 0
    assert len(rig.commands.emitted) == marks['before']


def test_the_stick_replaces_a_target_that_a_stalled_driver_has_not_read(world):
    rig = _Rig(world)
    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
        # The driver stalls before it read the target, and the stick goes before the state goes stale
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(stick=1.0), 0.01),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    np.testing.assert_allclose(rig.commands.emitted[-1][1].pose.translation, ROBOT_POSE.translation)


def test_control_does_not_come_on_while_the_arm_moves_by_itself(world):
    rig = _Rig(world)
    driver = ManualDriver([
        (lambda: rig.state.emit(_state(RobotStatus.BUSY)), 0.01),
        (lambda: rig.hand(HAND), 0.01),
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(a=1.0), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert not rig.commands.emitted
    assert rig.events()[-1] == 'control refused: the arm is not available'


def test_the_controller_on_the_other_side_drives_nothing(world):
    rig = _Rig(world)
    driver = ManualDriver([
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.hand(HAND, side=LEFT), 0.01),
        (lambda: rig.press(side=LEFT), 0.01),
        (lambda: rig.press(side=LEFT, a=1.0, trigger=0.9), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05), side=LEFT), 0.01),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert not rig.commands.emitted and not rig.grips.emitted


def test_the_left_arm_reads_x_and_the_trigger_of_the_left_controller(world):
    rig = _Rig(world, side=LEFT)
    driver = ManualDriver([
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.hand(HAND, side=LEFT), 0.01),
        (lambda: rig.press(side=LEFT), 0.01),
        (lambda: rig.press(side=LEFT, a=1.0, trigger=0.3), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05), side=LEFT), 0.01),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert rig.commands.emitted and rig.grips.emitted[-1][1] == pytest.approx(0.3)


def test_y_of_the_left_controller_keeps_control_of_the_left_arm_and_moves_it_to_no_start_pose(world):
    rig = _Rig(world, side=LEFT)
    marks = {}
    driver = ManualDriver([
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.hand(HAND, side=LEFT), 0.01),
        (lambda: rig.press(side=LEFT), 0.01),
        (lambda: rig.press(side=LEFT, a=1.0), 0.01),
        (lambda: rig.press(side=LEFT), 0.01),
        (lambda: rig.press(side=LEFT, b=1.0), 0.01),  # Y: the camera and the panel of the headset page
        *rig.states(),
        (lambda: rig.take_calls(), 0.01),
        (lambda: marks.update(before=len(rig.commands.emitted)), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05), side=LEFT), 0.01),
        (lambda: marks.update(after=len(rig.commands.emitted)), 0.0),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert not rig.calls
    assert marks['after'] > marks['before'], 'control of the left arm is still on'


def test_the_panel_shows_an_arm_that_sent_no_report():
    image = Panel([LEFT, RIGHT]).view()
    assert image.shape == (540, 960, 3) and image.any()


@pytest.mark.parametrize(
    ('arms', 'match'),
    [
        ({}, 'the arms are on'),
        ({'middle': Arm(ManualDriver([]), START_JOINTS)}, 'the arms are on'),
        ({RIGHT: Arm(ManualDriver([]), [0.0, float('nan')])}, 'finite angle'),
    ],
    ids=['no arm', 'an arm on no side', 'a start pose that is not finite'],
)
def test_a_station_that_names_its_arms_wrong_is_refused(arms, match):
    with pytest.raises(ValueError, match=match):
        debug(arms, webxr=None, operator_position=OperatorPosition.FRONT)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('silence_sec', [0.0, -1.0, float('nan'), float('inf')])
def test_a_silence_limit_that_is_not_finite_and_positive_is_refused(silence_sec):
    arms = {RIGHT: Arm(ManualDriver([]), START_JOINTS)}
    with pytest.raises(ValueError, match='silence_sec'):
        debug(arms, webxr=None, operator_position=OperatorPosition.FRONT, silence_sec=silence_sec)  # pyright: ignore[reportArgumentType]


def test_the_yam_bimanual_arms_are_the_arms_of_the_yambox_station():
    station = embodiment_cfg.yam_bimanual_yambox.override(cameras={}).instantiate()
    mounts = station.static_meta[eval_keys.MOUNTS]
    arms = yam_bimanual_arms.instantiate()

    for side, arm in ((LEFT, keys.LEFT_ARM), (RIGHT, keys.RIGHT_ARM)):
        robot = arms[side].robot
        assert isinstance(robot, yam_driver.Robot) and arms[side].gripper is robot
        np.testing.assert_allclose(robot.base_pose.translation, mounts[f'{keys.ROBOT_STATE}.{arm}.q'])


@pytest.mark.parametrize(
    'args',
    [*([name] for name in commands), ['yam_bimanual', '--arms.station.left_channel=can2']],
    ids=[*commands, 'yam_bimanual with a channel'],
)
def test_the_help_of_each_robot_shows_its_whole_config(args, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['positronic', *args, '--help'])
    cfn.cli(commands)

    shown = capsys.readouterr().out
    assert 'Config:' in shown and '<REQUIRED>' not in shown
    assert all(arg.split('=')[1] in shown for arg in args[1:])
