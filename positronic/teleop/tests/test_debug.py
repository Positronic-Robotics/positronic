import types

import numpy as np
import pytest

import pimm
from positronic.drivers.roboarm import RobotStatus, command
from positronic.drivers.webxr import LEFT, RIGHT
from positronic.geom import Rotation, Transform3D
from positronic.teleop.clutch import OperatorPosition
from positronic.teleop.debug import Arm, ArmControl, Panel, debug
from positronic.teleop.service import Button
from positronic.tests.testing_coutils import ManualDriver, RecordingEmitter, drive_scheduler

START_JOINTS = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
ROBOT_POSE = Transform3D(np.array([0.4, 0.0, 0.3]), Rotation.identity)
HAND = Transform3D(np.array([0.1, 1.0, -0.3]), Rotation.identity)


def _state(status: RobotStatus = RobotStatus.AVAILABLE) -> types.SimpleNamespace:
    return types.SimpleNamespace(ee_pose=ROBOT_POSE, status=status)


def _buttons(trigger: float = 0.0, a: float = 0.0, b: float = 0.0) -> list[float]:
    """The values of a controller: A and B are X and Y on the left controller."""
    values = [0.0] * (max(Button) + 1)
    values[Button.TRIGGER], values[Button.A], values[Button.B] = trigger, a, b
    return values


def _raised(pose: Transform3D, dz: float) -> Transform3D:
    return Transform3D(pose.translation + np.array([0.0, 0.0, dz]), pose.rotation)


@pytest.fixture
def world():
    with pimm.World(virtual_time=True) as w:
        yield w


class _Rig:
    """An ``ArmControl`` with each of its ports paired, and recorders on what it emits."""

    def __init__(self, world: pimm.World, side: str = RIGHT):
        self.control = ArmControl(side, START_JOINTS, OperatorPosition.FRONT)
        self.poses = world.pair(self.control.controller_positions)
        self.buttons = world.pair(self.control.buttons)
        self.state = world.pair(self.control.robot_state)
        self.sync_move = world.pair(self.control.sync_move)
        self.commands, self.grips, self.reports = RecordingEmitter(), RecordingEmitter(), RecordingEmitter()
        self.control.robot_commands._bind(self.commands, clock=world.clock)
        self.control.target_grip._bind(self.grips, clock=world.clock)
        self.control.report._bind(self.reports, clock=world.clock)
        self.calls: list = []

    def take_calls(self) -> None:
        self.calls.extend(self.sync_move.incoming())

    def hand(self, pose: Transform3D | None, side: str = RIGHT) -> None:
        self.poses.emit({LEFT: None, RIGHT: None, side: pose})

    def press(self, side: str = RIGHT, **values: float) -> None:
        self.buttons.emit({LEFT: None, RIGHT: None, side: _buttons(**values)})

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

    assert marks['on'] > 0 and marks['off'] == marks['on']
    shift = rig.commands.emitted[-1][1].pose.translation - ROBOT_POSE.translation
    assert np.linalg.norm(shift) == pytest.approx(0.05)
    assert rig.events()[-2:] == ['control on', 'control off']


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

    assert marks['back'] == marks['on'] > 0
    assert rig.events()[-1] == 'control off: the controller is not tracked'


def test_b_turns_control_off_clears_the_error_and_then_puts_the_arm_at_its_start_pose(world):
    rig = _Rig(world)
    ready = world.pair(rig.control.ready)
    cleared: list = []
    marks = {}
    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(b=1.0), 0.01),
        (lambda: cleared.extend(ready.incoming()), 0.01),
        (lambda: rig.take_calls(), 0.01),
        (lambda: marks.update(moves_before_clear=len(rig.calls)), 0.01),
        (lambda: cleared[0].set_result(None), 0.01),
        (lambda: rig.take_calls(), 0.01),
        (lambda: rig.calls[0].set_result(None), 0.01),
        (lambda: marks.update(at_start=len(rig.commands.emitted)), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
        (lambda: marks.update(after=len(rig.commands.emitted)), 0.0),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert len(cleared) == 1 and marks['moves_before_clear'] == 0 and len(rig.calls) == 1
    move = rig.calls[0].request
    assert isinstance(move, command.JointPosition)
    np.testing.assert_allclose(move.positions, START_JOINTS)
    assert rig.events()[-2:] == ['to the start pose', 'at the start pose']
    assert marks['after'] == marks['at_start'], 'control is off at the start pose'


def test_a_start_pose_that_the_arm_refuses_is_an_event_and_the_teleoperation_goes_on(world):
    rig = _Rig(world)
    driver = ManualDriver([
        (lambda: rig.state.emit(_state()), 0.01),
        (lambda: rig.press(), 0.01),
        (lambda: rig.press(b=1.0), 0.01),
        (lambda: rig.take_calls(), 0.01),
        (lambda: rig.calls[0].set_exception(RuntimeError('out of reach')), 0.01),
        (lambda: rig.press(trigger=0.4), 0.01),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert rig.events()[-1] == 'the start pose failed: out of reach'
    assert rig.grips.emitted[-1][1] == pytest.approx(0.4)


def test_an_arm_in_error_gets_no_target_and_its_error_is_an_event(world):
    rig = _Rig(world)
    marks = {}
    driver = ManualDriver([
        *rig.control_on(),
        (lambda: rig.state.emit(_state(RobotStatus.ERROR)), 0.01),
        (lambda: marks.update(error=len(rig.commands.emitted)), 0.01),
        (lambda: rig.hand(_raised(HAND, 0.05)), 0.01),
        (lambda: marks.update(after=len(rig.commands.emitted)), 0.0),
    ])
    drive_scheduler(world.start([rig.control, driver]), steps=400)

    assert marks['after'] == marks['error']
    assert rig.events()[-1] == 'the arm reports an error: B or Y clears it'
    assert rig.reports.emitted[-1][1].status is RobotStatus.ERROR


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
    assert not rig.reports.emitted[-1][1].control


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
