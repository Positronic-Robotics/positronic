"""Teleoperate the arms of a station with no eval, no task and no recording: to bring up a new robot, or to find a
fault.

``positronic teleop-debug <robot>`` drives one arm from the right controller, or two arms from the two controllers.
Each arm has its own controls:

- A (right controller) or X (left controller): control of the arm on or off.
- B or Y: control off, the errors of the arm and its gripper cleared, and the arm at its start pose.
- The trigger: the grip.

Control of an arm goes off when its controller is not tracked, when the headset sends nothing for a while, and when
the arm is not available: in an error, busy with a move of its own, or with no state. A press of A or X turns it on
again from where the hand and the arm are then. The headset video shows a panel with the state of each arm and its
last events, and the log has the same events.
"""

import collections
import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import configuronic as cfn
import cv2
import numpy as np
from configuronic.cli import CommandTree

import pimm
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.webxr
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, command
from positronic.drivers.webxr import LEFT, RIGHT, WebXR

from .clutch import OperatorPosition, Tracker
from .service import Button, ButtonMap


@dataclass(frozen=True)
class Arm:
    """An arm that one controller drives: the robot, its start pose, its gripper, and the handlers that clear errors.

    ``robot`` emits ``state`` and takes ``commands`` and ``sync_move``. ``gripper`` emits ``grip`` and takes
    ``target_grip``, and can be ``robot`` itself. ``ready`` holds the handler of each device that clears its error.
    """

    robot: pimm.ControlSystem
    start_joints: Sequence[float]
    gripper: pimm.ControlSystem | None = None
    ready: tuple[pimm.calls.ControlSystemHandler[None, None], ...] = ()


@dataclass(frozen=True)
class ArmReport:
    """What the panel shows of an arm. ``status`` is ``None`` while the arm gives no state. The shifts are in metres
    in the robot frame, from where control came on."""

    status: RobotStatus | None
    control: bool
    events: tuple[str, ...]
    trigger: float | None = None
    grip: float | None = None
    target_shift: np.ndarray | None = None
    arm_shift: np.ndarray | None = None


class ArmControl(pimm.ControlSystem):
    """The teleoperation of one arm from the controller on ``side``."""

    # The events that a report keeps, and how often the arm sends its report
    EVENTS = 4
    REPORT_SEC = 0.2
    # Control goes off when no sample of the controllers comes for this long, as in the teleop service
    SILENCE_SEC = 0.5

    def __init__(self, side: str, start_joints: Sequence[float], operator_position: OperatorPosition):
        self.side = side
        self._start_joints = np.asarray(start_joints, dtype=np.float64)
        self._tracker = Tracker(operator_position.value)
        # A and B, which are X and Y on the left controller. Only the presses count, so the hold time does not matter.
        self._buttons = ButtonMap(1.0, buttons=(Button.A, Button.B))
        self._events: collections.deque[str] = collections.deque(maxlen=self.EVENTS)
        # The tool point when control came on, and the last target since then
        self._start: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._trigger: float | None = None

        self.controller_positions = pimm.DefaultingReceiver[dict[str, geom.Transform3D | None]](self, default={})
        self.buttons = pimm.DefaultingReceiver[dict[str, Sequence[float] | None]](self, default={})
        self.robot_state = pimm.ControlSystemReceiver(self)
        self.grip = pimm.ControlSystemReceiver[float](self)
        self.robot_commands = pimm.ControlSystemEmitter[command.CommandType](self)
        self.target_grip = pimm.ControlSystemEmitter[float](self)
        self.sync_move = pimm.calls.ControlSystemCaller[command.CommandType, None](self)
        self.ready = pimm.calls.CallerDict[None, None](self)
        self.report = pimm.ControlSystemEmitter[ArmReport](self)

    def _state(self) -> Any:
        """The state of the arm, or ``None`` while it gives none."""
        message = self.robot_state.read()
        return None if message is None or isinstance(message.data, pimm.SignalError) else message.data

    def _send_report(self) -> None:
        state = self._state()
        grip = self.grip.read()
        target_shift = arm_shift = None
        if self._tracker.on and self._start is not None:
            target_shift = None if self._target is None else self._target - self._start
            arm_shift = None if state is None else np.asarray(state.ee_pose.translation) - self._start
        self.report.emit(
            ArmReport(
                None if state is None else state.status,
                self._tracker.on,
                tuple(self._events),
                self._trigger,
                None if grip is None or isinstance(grip.data, pimm.SignalError) else float(grip.data),
                target_shift,
                arm_shift,
            )
        )

    def _note(self, text: str, level: int = logging.INFO) -> None:
        """Log ``text`` for this arm, and show it at once in the events of its report."""
        logging.log(level, '%s arm: %s', self.side, text)
        self._events.append(f'{time.strftime("%H:%M:%S")} {text}')
        self._send_report()

    def _control_off(self, reason: str) -> None:
        if self._tracker.on:
            self._tracker.turn_off()
            self._note(f'control off: {reason}', logging.WARNING)

    def _toggle_control(self, hand: geom.Transform3D | None) -> None:
        if self._tracker.on:
            self._tracker.turn_off()
            self._note('control off')
            return
        state = self._state()
        if hand is None or state is None or state.status is not RobotStatus.AVAILABLE:
            self._note('control refused: the controller is not tracked, or the arm is not available', logging.WARNING)
            return
        self._tracker.turn_on(state.ee_pose, hand)
        self._start, self._target = np.array(state.ee_pose.translation, dtype=np.float64), None
        self._note('control on')

    @staticmethod
    def _wait(answer: pimm.calls.Answer, should_stop: pimm.SignalReceiver) -> Iterator[pimm.Sleep]:
        """Yield until ``answer`` comes, and raise what it failed on. Return early when the world stops."""
        while not answer.done():
            if should_stop.value:
                return
            yield pimm.Sleep(0.001)
        answer.result()

    def _to_start(self, should_stop: pimm.SignalReceiver) -> Iterator[pimm.Sleep]:
        """Clear the errors of the arm and its gripper, and put the arm at its start pose. Raise what a device failed
        on."""
        yield from self._wait(pimm.calls.all_of([ready(None) for ready in self.ready.values()]), should_stop)
        if should_stop.value:
            return
        yield from self._wait(self.sync_move(command.JointPosition(self._start_joints)), should_stop)
        if not should_stop.value:
            self._note('at the start pose')

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        next_report = clock.now()
        last_sample = clock.now()
        in_error = False
        while not should_stop.value:
            positions = self.controller_positions.read()
            if positions.updated:
                last_sample = clock.now()
            hand = positions.data.get(self.side)
            values = self.buttons.value.get(self.side)
            presses: set[Button] = set()
            if values is None:
                self._buttons.lose()
            else:
                presses = self._buttons.presses(values, clock.now_ns())
                self._trigger = ButtonMap.grip(values)
                self.target_grip.emit(self._trigger)

            state = self._state()
            error = state is not None and state.status is RobotStatus.ERROR
            if error and not in_error:
                self._note('the arm reports an error: B or Y clears it', logging.ERROR)
            in_error = error
            if hand is None:
                self._control_off('the controller is not tracked')
            elif clock.now() - last_sample > self.SILENCE_SEC:
                self._control_off('the headset sends no sample')
            elif state is None or state.status is not RobotStatus.AVAILABLE:
                self._control_off('the arm is not available')

            if Button.A in presses:
                self._toggle_control(hand)
            elif Button.B in presses:
                self._tracker.turn_off()
                self._note('to the start pose')
                try:
                    yield from self._to_start(should_stop)
                # rules-allow: swallowed-error — the operator reads the failure on the panel and presses again
                except Exception as e:
                    self._note(f'the start pose failed: {e}', logging.ERROR)
            elif self._tracker.on and positions.updated and hand is not None:
                target = self._tracker.update(hand)
                self.robot_commands.emit(command.CartesianPosition(target))
                self._target = np.array(target.translation, dtype=np.float64)

            if clock.now() >= next_report:
                self._send_report()
                next_report = clock.now() + self.REPORT_SEC
            yield pimm.Sleep(0.001)


def _shift_text(shift: np.ndarray | None) -> str:
    if shift is None:
        return '-'
    x, y, z = np.round(np.asarray(shift) * 100).astype(int)
    return f'x{x:+d} y{y:+d} z{z:+d} cm'


class Panel(pimm.ControlSystem):
    """The state of each arm as text on black, for the headset video."""

    # The size of the image, and how often the panel sends it
    SIZE = (960, 540)
    PERIOD_SEC = 0.2
    WHITE, GREEN, RED, YELLOW = (255, 255, 255), (80, 255, 80), (255, 80, 80), (255, 230, 80)

    def __init__(self, sides: Sequence[str]):
        self._sides = tuple(sides)
        self.reports = pimm.ReceiverDict(self)
        self.frame = pimm.ControlSystemEmitter[np.ndarray](self)

    def _lines(self, side: str) -> list[tuple[str, tuple[int, int, int]]]:
        message = self.reports[side].read()
        if message is None:
            return [(f'{side}: no report', self.RED)]
        report: ArmReport = message.data
        status = 'no state' if report.status is None else report.status.name
        error = report.status is None or report.status is RobotStatus.ERROR
        color = self.RED if error else (self.GREEN if report.control else self.WHITE)
        grip = '' if report.trigger is None else f', trigger {report.trigger:.2f}'
        if report.grip is not None:
            grip += f', grip {report.grip:.2f}'
        lines = [(f'{side}: {status}, control {"ON" if report.control else "off"}{grip}', color)]
        if report.control:
            lines.append((f'  target {_shift_text(report.target_shift)}', self.YELLOW))
            lines.append((f'  arm    {_shift_text(report.arm_shift)}', self.YELLOW))
        return lines + [(f'  {event}', self.WHITE) for event in report.events]

    def view(self) -> np.ndarray:
        width, height = self.SIZE
        image = np.zeros((height, width, 3), dtype=np.uint8)
        row = 0
        for side in self._sides:
            for text, color in self._lines(side):
                cv2.putText(image, text, (10, 34 + 32 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                row += 1
            row += 1
        return image

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        while not should_stop.value:
            self.frame.emit(self.view())
            yield pimm.Sleep(self.PERIOD_SEC)


def debug(arms: Mapping[str, Arm], webxr: WebXR, operator_position: OperatorPosition) -> None:
    """Run the teleoperation of ``arms`` until Ctrl-C. The controller on the side of each arm drives it."""
    if not arms or not set(arms) <= {LEFT, RIGHT}:
        raise ValueError(f'the arms are on the {LEFT} side, the {RIGHT} side or both, got {list(arms)}')
    for side, arm in arms.items():
        if not len(arm.start_joints) or not np.all(np.isfinite(arm.start_joints)):
            raise ValueError(f'the {side} arm needs a finite angle for each joint of its start pose')
    with pimm.World() as world:
        panel = Panel(list(arms))
        world.connect(panel.frame, webxr.frame)
        controls: list[pimm.ControlSystem | None] = [panel]
        devices: list[pimm.ControlSystem | None] = [webxr]
        for side, arm in arms.items():
            control = ArmControl(side, arm.start_joints, operator_position)
            world.connect(webxr.controller_positions, control.controller_positions)
            world.connect(webxr.buttons, control.buttons)
            # A driver's ports are its own: `pimm.ControlSystem` declares none for the checker to find
            robot: Any = arm.robot
            world.connect(robot.state, control.robot_state)
            world.connect(control.robot_commands, robot.commands)
            world.connect(control.sync_move, robot.sync_move)
            for index, ready in enumerate(arm.ready):
                world.connect(control.ready[str(index)], ready)
            devices.append(arm.robot)
            if arm.gripper is not None:
                gripper: Any = arm.gripper
                world.connect(control.target_grip, gripper.target_grip)
                world.connect(gripper.grip, control.grip)
                if arm.gripper is not arm.robot:
                    devices.append(arm.gripper)
            world.connect(control.report, panel.reports[side])
            controls.append(control)
        # Each device runs in a process of its own: the vendor client of a CRON arm keeps one connection a process
        world.run(controls, devices)


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.franka_droid,
    gripper=positronic.cfg.hardware.gripper.robotiq,
    start_joints=positronic.cfg.hardware.roboarm.FRANKA_NOMINAL_JOINTS,
    webxr=positronic.cfg.webxr.oculus,
    operator_position=OperatorPosition.BACK,
)
def franka(robot_arm, gripper, start_joints: Sequence[float], webxr: WebXR, operator_position: OperatorPosition):
    """A Franka arm with a Robotiq gripper."""
    arm = Arm(robot_arm, start_joints, gripper, (robot_arm.ready, gripper.ready))
    debug({RIGHT: arm}, webxr, operator_position)


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.yam,
    start_joints=positronic.cfg.hardware.roboarm.YAM_NOMINAL_JOINTS,
    webxr=positronic.cfg.webxr.oculus,
    operator_position=OperatorPosition.BACK,
)
def yam(robot_arm, start_joints: Sequence[float], webxr: WebXR, operator_position: OperatorPosition):
    """One YAM arm. Its driver also drives its gripper, on the same CAN chain."""
    debug({RIGHT: Arm(robot_arm, start_joints, robot_arm, (robot_arm.ready,))}, webxr, operator_position)


@cfn.config(
    left_arm=positronic.cfg.hardware.roboarm.yam.override(channel='can0'),
    right_arm=positronic.cfg.hardware.roboarm.yam.override(channel='can1'),
    start_joints=positronic.cfg.hardware.roboarm.YAM_NOMINAL_JOINTS,
    webxr=positronic.cfg.webxr.oculus,
    operator_position=OperatorPosition.BACK,
)
def yam_bimanual(left_arm, right_arm, start_joints: Sequence[float], webxr: WebXR, operator_position: OperatorPosition):
    """Two YAM arms on two CAN chains. The driver of each arm also drives its gripper. A station whose chains have
    other names sets ``--left_arm.channel`` and ``--right_arm.channel``."""
    arms = {
        side: Arm(robot, start_joints, robot, (robot.ready,)) for side, robot in ((LEFT, left_arm), (RIGHT, right_arm))
    }
    debug(arms, webxr, operator_position)


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.so101,
    start_joints=positronic.cfg.hardware.roboarm.SO101_NOMINAL_JOINTS,
    webxr=positronic.cfg.webxr.oculus,
    operator_position=OperatorPosition.BACK,
)
def so101(robot_arm, start_joints: Sequence[float], webxr: WebXR, operator_position: OperatorPosition):
    """One SO-101 arm. Its driver also drives its gripper, on the same motor bus."""
    debug({RIGHT: Arm(robot_arm, start_joints, robot_arm)}, webxr, operator_position)


# TODO: Drive the DH gripper on each CRON arm, once the DH driver reaches it through the bridge project on the box.
@cfn.config(
    left_arm=positronic.cfg.hardware.roboarm.cron.override(ip='192.168.68.67'),
    right_arm=positronic.cfg.hardware.roboarm.cron.override(ip='192.168.68.66'),
    left_start_joints=positronic.cfg.hardware.roboarm.CRON_LEFT_NOMINAL_JOINTS,
    right_start_joints=positronic.cfg.hardware.roboarm.CRON_RIGHT_NOMINAL_JOINTS,
    webxr=positronic.cfg.webxr.oculus,
    operator_position=OperatorPosition.FRONT,
)
def cron_bimanual(
    left_arm,
    right_arm,
    left_start_joints: Sequence[float],
    right_start_joints: Sequence[float],
    webxr: WebXR,
    operator_position: OperatorPosition,
):
    """Two CRON arms. A move of a CRON arm clears its error before it starts."""
    arms = {LEFT: Arm(left_arm, left_start_joints), RIGHT: Arm(right_arm, right_start_joints)}
    debug(arms, webxr, operator_position)


# The subcommands of `positronic teleop-debug`: one for each robot
commands: CommandTree = {
    'franka': franka,
    'yam': yam,
    'yam_bimanual': yam_bimanual,
    'so101': so101,
    'cron_bimanual': cron_bimanual,
}
