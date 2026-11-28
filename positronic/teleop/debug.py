"""Teleoperate the arms of a station with no eval, no task and no recording: to bring up a new robot, or to find a
fault.

``positronic teleop-debug <robot>`` drives one arm from the right controller, or two arms from the two controllers.
Each arm has its own controls:

- A (right controller) or X (left controller): control of the arm on or off.
- A press of the stick: control off, a call of the ready handler of each device of the arm that has one, and a move
  of the arm to its start pose. No control stops this move.
- Y on the left controller stays with the headset page: the camera and the panel.
- The trigger: the grip, while control of the arm is on.

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
import positronic.cfg.embodiment
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.webxr
from positronic import geom, keys
from positronic.drivers.roboarm import RobotStatus, command
from positronic.drivers.webxr import LEFT, RIGHT, ControllerButtons, WebXR
from positronic.eval import Embodiment
from positronic.eval import keys as eval_keys
from positronic.utils import check_finite_positive

from .clutch import CONTROLLER_AXES, OperatorPosition, Tracker
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
    """The teleoperation of one arm from the controller on ``side``. Control goes off when no sample of the
    controllers comes for ``silence_sec``."""

    # The events that a report keeps, how long an event is on the panel, and how often the arm sends its report
    EVENTS = 4
    EVENT_CHARS = 60
    REPORT_SEC = 0.2
    # A driver reads a move before a target and keeps the target until the move ends, so a start move goes only after
    # the driver read the last target. Each tick of a driver reads its requests and sends one state, in either order.
    # The third new state comes from a tick that began after the second came, so after the last target. The start
    # move fails when the states do not come in `SETTLE_SEC`.
    SETTLE_STATES = 3
    SETTLE_SEC = 0.5
    # An arm whose driver sends no new state for this long gives no state: its last state can be from before a stall
    STALE_STATE_SEC = 0.5
    # How often the arm reads its signals: twice in a frame of a headset at 120 Hz, the highest rate of a Quest
    POLL_SEC = 1 / 240

    def __init__(
        self,
        side: str,
        start_joints: Sequence[float],
        operator_position: OperatorPosition,
        controller_axes: geom.Rotation | None,
        silence_sec: float,
    ):
        self.side = side
        self._start_joints = np.asarray(start_joints, dtype=np.float64)
        self._tracker = Tracker(operator_position.value, controller_axes)
        self._silence_sec = silence_sec
        # A (X on the left controller) and the stick. The headset page uses Y. Only the presses count, so the hold time
        # does not matter.
        self._buttons = ButtonMap(1.0, buttons=(Button.A, Button.STICK))
        self._events: collections.deque[str] = collections.deque(maxlen=self.EVENTS)
        # The tool point when control came on, and the last target since then
        self._start: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._trigger: float | None = None
        self._in_error = False
        self._clock: pimm.Clock | None = None  # the clock of `run`
        self._state_at: float | None = None  # when the arm sent its last new state
        # The new states since the last target, until `SETTLE_STATES` of them show the driver read it; None after that
        self._states_since_target: int | None = None

        self.controller_positions = pimm.DefaultingReceiver[dict[str, geom.Transform3D | None]](self, default={})
        self.buttons = pimm.DefaultingReceiver[dict[str, ControllerButtons | None]](self, default={})
        self.robot_state = pimm.ControlSystemReceiver(self)
        self.grip = pimm.ControlSystemReceiver[float](self)
        self.robot_commands = pimm.ControlSystemEmitter[command.CommandType](self)
        self.target_grip = pimm.ControlSystemEmitter[float](self)
        self.sync_move = pimm.calls.ControlSystemCaller[command.CommandType, None](self)
        self.ready = pimm.calls.CallerDict[None, None](self)
        self.report = pimm.ControlSystemEmitter[ArmReport](self)

    def _state(self) -> Any:
        """The state of the arm, or ``None`` while it gives none: before its first state, while its signal is an error,
        and when it sent no new state for ``STALE_STATE_SEC``."""
        message = self.robot_state.read()
        if message is None or isinstance(message.data, pimm.SignalError) or self._clock is None:
            return None
        now = self._clock.now()
        if message.updated:
            self._state_at = now
            if self._states_since_target is not None:
                self._states_since_target += 1
                if self._states_since_target >= self.SETTLE_STATES:
                    self._states_since_target = None
        if self._state_at is None or now - self._state_at > self.STALE_STATE_SEC:
            return None
        return message.data

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
        """Log ``text`` for this arm, and show its start at once in the events of its report."""
        logging.log(level, '%s arm: %s', self.side, text)
        shown = text.splitlines()[0] if text else text
        self._events.append(f'{time.strftime("%H:%M:%S")} {shown[: self.EVENT_CHARS]}')
        self._send_report()

    def _hold_unread_target(self) -> None:
        """Replace the last target, while the driver may not have read it, with the last pose the arm sent, so a
        stalled driver does not move the arm to it when it comes back."""
        # rules-allow: hidden-dependency — the hold replaces only a target the driver has not read. A target it read
        # came while its states were fresh, as every target of control does, and only the driver can drop it
        if self._states_since_target is None:
            return
        self._states_since_target = None
        message = self.robot_state.read()
        if message is not None and not isinstance(message.data, pimm.SignalError):
            self.robot_commands.emit(command.CartesianPosition(message.data.ee_pose))

    def _turn_off(self) -> None:
        """Turn control off, and replace the target the driver may not have read."""
        self._tracker.turn_off()
        self._hold_unread_target()

    def _control_off(self, reason: str) -> None:
        if self._tracker.on:
            self._turn_off()
            self._note(f'control off: {reason}', logging.WARNING)

    def _toggle_control(self, hand: geom.Transform3D | None) -> None:
        if self._tracker.on:
            self._turn_off()
            self._note('control off')
            return
        state = self._state()
        if hand is None:
            self._note('control refused: no controller', logging.WARNING)
        elif state is None:
            self._note('control refused: the arm sends no state', logging.WARNING)
        elif state.status is not RobotStatus.AVAILABLE:
            self._note('control refused: the arm is not available', logging.WARNING)
        else:
            self._tracker.turn_on(state.ee_pose, hand)
            self._start, self._target = np.array(state.ee_pose.translation, dtype=np.float64), None
            self._note('control on')

    def _read_buttons(
        self, message: pimm.Message[dict[str, ControllerButtons | None]], now_ns: int, silent: bool
    ) -> tuple[dict[Button, int], float | None]:
        """The number of presses of each button of the controller in ``message``, and the grip of a new sample."""
        buttons = message.data.get(self.side)
        if buttons is None or silent:
            self._buttons.cancel_holds()
            return {}, None
        grip = None
        if message.updated:
            grip = self._trigger = ButtonMap.grip(buttons.values)
        return self._buttons.new_presses(buttons, now_ns), grip

    def _check_availability(self, hand: geom.Transform3D | None, silent: bool) -> None:
        """Turn control off when the controller or the arm is not available, and note an error of the arm once."""
        state = self._state()
        error = state is not None and state.status is RobotStatus.ERROR
        if error and not self._in_error:
            self._note('the arm reports an error: a press of the stick clears it', logging.ERROR)
        self._in_error = error
        if silent:
            self._control_off('the headset sends no sample')
        elif hand is None:
            self._control_off('the controller is not tracked')
        elif state is None:
            self._control_off('the arm sends no state')
        elif state.status is not RobotStatus.AVAILABLE:
            self._control_off('the arm is not available')

    def _wait(
        self, answer: pimm.calls.Answer, should_stop: pimm.SignalReceiver, clock: pimm.Clock
    ) -> Iterator[pimm.Sleep]:
        """Yield until ``answer`` comes, and raise what it failed on. Return early when the world stops. The panel
        keeps the state of the arm while it waits."""
        next_report = clock.now() + self.REPORT_SEC
        while not answer.done() and not should_stop.value:
            if clock.now() >= next_report:
                self._send_report()
                next_report = clock.now() + self.REPORT_SEC
            yield pimm.Sleep(self.POLL_SEC)
        if answer.done():
            answer.result()

    def _settle(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        """Yield until the arm sends ``SETTLE_STATES`` new states. Raise TimeoutError when they do not come in
        ``SETTLE_SEC``."""
        deadline = clock.now() + self.SETTLE_SEC
        states = 0
        while states < self.SETTLE_STATES and not should_stop.value:
            if clock.now() >= deadline:
                raise TimeoutError(f'the arm sent {states} new states in {self.SETTLE_SEC} s')
            message = self.robot_state.read()
            if message is not None and message.updated:
                states += 1
                self._state_at = clock.now()
            yield pimm.Sleep(self.POLL_SEC)

    def _to_start(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        """Call the ready handlers of the arm, and put the arm at its start pose. Raise what a device failed on."""
        yield from self._wait(pimm.calls.all_of([ready(None) for ready in self.ready.values()]), should_stop, clock)
        yield from self._settle(should_stop, clock)
        if should_stop.value:
            return
        self._states_since_target = None  # the settle states show that the driver read the last target
        yield from self._wait(self.sync_move(command.JointPosition(self._start_joints)), should_stop, clock)
        if not should_stop.value:
            self._note('at the start pose')

    def _go_to_start(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self._turn_off()  # a stalled driver that comes back to answer ready would read the target next
        self._note('to the start pose')
        try:
            yield from self._to_start(should_stop, clock)
        # rules-allow: swallowed-error — the operator reads the failure on the panel and presses again
        except Exception as e:
            self._note(f'the start pose failed: {e}', logging.ERROR)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self._clock = clock
        next_report = clock.now()
        last_sample = clock.now()
        while not should_stop.value:
            # rules-allow: hidden-dependency — WebXR emits the poses of a sample before its buttons, so buttons read
            # first pair a press with the poses of its own sample or a later one; one signal would change each reader
            buttons = self.buttons.read()
            positions = self.controller_positions.read()
            if positions.updated:
                last_sample = clock.now()
            silent = clock.now() - last_sample > self._silence_sec
            hand = positions.data.get(self.side)
            presses, grip = self._read_buttons(buttons, clock.now_ns(), silent)
            self._check_availability(hand, silent)

            if Button.STICK in presses:  # the start move turns control off, so it wins over A in the same sample
                yield from self._go_to_start(should_stop, clock)
                self._read_buttons(self.buttons.read(), clock.now_ns(), silent)  # drop the presses made during the move
            elif Button.A in presses:
                for _ in range(ButtonMap.clutches(presses[Button.A])):
                    self._toggle_control(hand)  # a silent headset gives no press
            elif self._tracker.on and positions.updated and hand is not None:
                target = self._tracker.update(hand)
                self.robot_commands.emit(command.CartesianPosition(target))
                self._states_since_target = 0
                self._target = np.array(target.translation, dtype=np.float64)
            if self._tracker.on and grip is not None:
                self.target_grip.emit(grip)

            if clock.now() >= next_report:
                self._send_report()
                next_report = clock.now() + self.REPORT_SEC
            yield pimm.Sleep(self.POLL_SEC)


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


def debug(
    arms: Mapping[str, Arm],
    webxr: WebXR,
    operator_position: OperatorPosition,
    controller_axes: geom.Rotation | None = CONTROLLER_AXES,
    silence_sec: float = 0.5,
) -> None:
    """Run the teleoperation of ``arms`` until Ctrl-C. The controller on the side of each arm drives it.

    ``controller_axes`` puts the axes of the controller in the frame of the headset, as in ``positronic teleop``.
    Control of an arm goes off when no sample of the controllers comes for ``silence_sec``.
    """
    if not arms or not set(arms) <= {LEFT, RIGHT}:
        raise ValueError(f'the arms are on the {LEFT} side, the {RIGHT} side or both, got {list(arms)}')
    for side, arm in arms.items():
        if not len(arm.start_joints) or not np.all(np.isfinite(arm.start_joints)):
            raise ValueError(f'the {side} arm needs a finite angle for each joint of its start pose')
    check_finite_positive(silence_sec, 'silence_sec')
    with pimm.World() as world:
        panel = Panel(list(arms))
        world.connect(panel.frame, webxr.frame)
        controls: list[pimm.ControlSystem] = [panel]
        devices: list[pimm.ControlSystem] = [webxr]
        for side, arm in arms.items():
            control = ArmControl(side, arm.start_joints, operator_position, controller_axes, silence_sec)
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
        world.run([*controls], [*devices])


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.franka_droid,
    gripper=positronic.cfg.hardware.gripper.robotiq,
    start_joints=positronic.cfg.hardware.roboarm.FRANKA_NOMINAL_JOINTS,
)
def franka_arms(robot_arm, gripper, start_joints: Sequence[float]) -> dict[str, Arm]:
    """A Franka arm with a Robotiq gripper."""
    return {RIGHT: Arm(robot_arm, start_joints, gripper, (robot_arm.ready, gripper.ready))}


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.yam, start_joints=positronic.cfg.hardware.roboarm.YAM_NOMINAL_JOINTS
)
def yam_arms(robot_arm, start_joints: Sequence[float]) -> dict[str, Arm]:
    """One YAM arm. Its driver also drives its gripper, on the same CAN chain."""
    return {RIGHT: Arm(robot_arm, start_joints, robot_arm, (robot_arm.ready,))}


@cfn.config(
    station=positronic.cfg.embodiment.yam_bimanual_yambox.override(cameras={}),
    start_joints=positronic.cfg.hardware.roboarm.YAM_NOMINAL_JOINTS,
)
def yam_bimanual_arms(station: Embodiment, start_joints: Sequence[float]) -> dict[str, Arm]:
    """The two YAM arms of a bimanual station, as the embodiment of the station builds them. The driver of each arm
    also drives its gripper."""
    arms = {}
    for side, arm in ((LEFT, keys.LEFT_ARM), (RIGHT, keys.RIGHT_ARM)):
        ready = station.ready_handlers[f'{eval_keys.ARM}.{arm}']
        arms[side] = Arm(ready.owner, start_joints, ready.owner, (ready,))
    return arms


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.so101, start_joints=positronic.cfg.hardware.roboarm.SO101_NOMINAL_JOINTS
)
def so101_arms(robot_arm, start_joints: Sequence[float]) -> dict[str, Arm]:
    """One SO-101 arm. Its driver also drives its gripper, on the same motor bus."""
    return {RIGHT: Arm(robot_arm, start_joints, robot_arm)}


# TODO: Drive the DH gripper on each CRON arm, once the DH driver reaches it through the bridge project on the box.
# The speed of 0.7 is the one that an operator of the stand found right for teleoperation.
@cfn.config(
    left_arm=positronic.cfg.hardware.roboarm.cron.override(ip='192.168.68.67', relative_dynamics_factor=0.7),
    right_arm=positronic.cfg.hardware.roboarm.cron.override(ip='192.168.68.66', relative_dynamics_factor=0.7),
    left_start_joints=positronic.cfg.hardware.roboarm.CRON_LEFT_NOMINAL_JOINTS,
    right_start_joints=positronic.cfg.hardware.roboarm.CRON_RIGHT_NOMINAL_JOINTS,
)
def cron_bimanual_arms(
    left_arm, right_arm, left_start_joints: Sequence[float], right_start_joints: Sequence[float]
) -> dict[str, Arm]:
    """Two CRON arms. A move of a CRON arm clears its error before it starts."""
    return {LEFT: Arm(left_arm, left_start_joints), RIGHT: Arm(right_arm, right_start_joints)}


teleop_debug = cfn.Config(debug, webxr=positronic.cfg.webxr.oculus, operator_position=OperatorPosition.BACK)

# The subcommands of `positronic teleop-debug`: one for each robot
commands: CommandTree = {
    'franka': teleop_debug.override(arms=franka_arms),
    'yam': teleop_debug.override(arms=yam_arms),
    'yam_bimanual': teleop_debug.override(arms=yam_bimanual_arms),
    'so101': teleop_debug.override(arms=so101_arms),
    'cron_bimanual': teleop_debug.override(arms=cron_bimanual_arms, operator_position=OperatorPosition.FRONT),
}
