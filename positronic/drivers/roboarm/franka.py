import contextlib
import functools
import logging
import os
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Generator, Iterator, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np

import pimm
from positronic import geom
from positronic.drivers import vendor_import
from positronic.drivers.utils import (
    DriverRun,
    MoveAbandoned,
    MoveRefused,
    MoveStatus,
    RecoveryOutcome,
    RecoveryStep,
    log_failure,
)

from . import RobotStatus, State, command
from .models import bundled_franka_model

with vendor_import(
    'positronic_franka',
    'Franka support',
    hint='Re-run with the franka extra:\n  uv run --locked --extra franka ...\n',
    platforms=('linux',),
):
    import positronic_franka._franka as pf
    from positronic_franka.desk import ControlHeldElsewhere, Desk, SafetyControllerError

logger = logging.getLogger(__name__)


def _check_error(is_error, was_error):
    return is_error, is_error and not was_error


DESK_USER_ENV = 'FRANKA_DESK_USER'
DESK_PASSWORD_ENV = 'FRANKA_DESK_PASSWORD'


def _read_desk_credentials() -> tuple[str, str]:
    """Franka Desk login and password from the environment. Credentials stay out of the config so they never reach
    the command line, which is recorded verbatim in every run's metadata."""
    login, password = os.environ.get(DESK_USER_ENV), os.environ.get(DESK_PASSWORD_ENV)
    if not (login and password):
        missing = ' and '.join(
            name for name, value in ((DESK_USER_ENV, login), (DESK_PASSWORD_ENV, password)) if not value
        )
        raise RuntimeError(
            f'{missing} not set in the environment. The driver needs Desk credentials to open the brakes and '
            f'activate FCI. Export them, or pass manage_desk=False to open the brakes and activate FCI yourself '
            f'in Desk before starting.'
        )
    return login, password


class FrankaState(State, pimm.shared_memory.NumpySMAdapter):
    Q_OFFSET = 0
    DQ_OFFSET = Q_OFFSET + 7
    EE_POSE_OFFSET = DQ_OFFSET + 7
    EE_WRENCH_OFFSET = EE_POSE_OFFSET + 7
    STATUS_OFFSET = EE_WRENCH_OFFSET + 6
    TOTAL = STATUS_OFFSET + 1

    def __init__(self):
        super().__init__(shape=(FrankaState.TOTAL,), dtype=np.float32)

    def instantiation_params(self) -> tuple[Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[FrankaState.Q_OFFSET : FrankaState.Q_OFFSET + 7].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.array[FrankaState.DQ_OFFSET : FrankaState.DQ_OFFSET + 7].copy()

    @property
    def ee_pose(self) -> geom.Transform3D:
        ee_pose = self.array[FrankaState.EE_POSE_OFFSET : FrankaState.EE_POSE_OFFSET + 7].copy()
        return geom.Transform3D(ee_pose[:3], geom.Rotation.from_quat(ee_pose[3:7]))

    @property
    def ee_wrench(self) -> np.ndarray | None:
        return self.array[FrankaState.EE_WRENCH_OFFSET : FrankaState.EE_WRENCH_OFFSET + 6].copy()

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[FrankaState.STATUS_OFFSET]))

    def encode(self, state: pf.State, status: RobotStatus):
        self.array[FrankaState.Q_OFFSET : FrankaState.Q_OFFSET + 7] = state.q
        self.array[FrankaState.DQ_OFFSET : FrankaState.DQ_OFFSET + 7] = state.dq
        self.array[FrankaState.EE_POSE_OFFSET : FrankaState.EE_POSE_OFFSET + 7] = state.end_effector_pose
        self.array[FrankaState.EE_WRENCH_OFFSET : FrankaState.EE_WRENCH_OFFSET + 6] = state.ee_wrench
        self.array[FrankaState.STATUS_OFFSET] = status.value


# Where the driver leaves the arm: taking control it travels here, and handing it back it returns here.
_PARK_JOINTS = np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0])

# The field Desk answers the safe inputs in.
SAFE_INPUT_STATE = 'safeInputState'
# The safe input the emergency stop is wired to.
EMERGENCY_STOP_INPUT = 'x31'
# The error the arm's state carries, and a ready call answers, while the emergency stop is pressed.
EMERGENCY_STOP_PRESSED = 'Release the emergency stop button'
# How the error the arm's state carries starts, and a ready call answers, while the driver has no connection to the
# arm. The reason follows it. Only a recover call connects again.
NO_CONNECTION = 'The driver has no connection to the arm'
# The recoverable error that Desk reports for a lost safety bus. Desk refuses to open the brakes until a reboot.
FSOE_CONNECTION_ERROR = 'fsoeConnectionError'


class _SafeInputLevel(StrEnum):
    """The signal level Desk reports for one safe input."""

    ACTIVE = 'Active'
    INACTIVE = 'Inactive'
    ACKNOWLEDGE_REQUIRED = 'AcknowledgeRequired'
    INVALID = 'Invalid'


class _Reading(NamedTuple):
    """One reading of the safe inputs, published in one assignment so a reader never sees half of it."""

    sampled: bool
    triggered: frozenset[str]
    levels: Mapping[str, _SafeInputLevel | None] = {}


class _SafeInputs:
    """The control box's safe inputs, which libfranka does not report and Desk does.

    A thread reads them every ``_POLL_S``, over a Desk client this owns: the read needs no control
    token, and the session that drives the arm must stay on one thread.
    """

    # ACTIVE releases the emergency stop (x31) and presses an enabling device (x4, guidingEnableButton);
    # x32 and x33 carry no fixed meaning. A level or an input not listed here counts as triggered.
    _CLEAR_STATES: Mapping[str, frozenset[_SafeInputLevel]] = {
        EMERGENCY_STOP_INPUT: frozenset({_SafeInputLevel.ACTIVE}),
        'x4': frozenset({_SafeInputLevel.INACTIVE}),
        'guidingEnableButton': frozenset({_SafeInputLevel.INACTIVE}),
        'x32': frozenset({_SafeInputLevel.ACTIVE, _SafeInputLevel.INACTIVE}),
        'x33': frozenset({_SafeInputLevel.ACTIVE, _SafeInputLevel.INACTIVE}),
    }
    # How often the watch thread reads the safe inputs.
    _POLL_S = 0.5

    def __init__(self, ip: str, credentials: tuple[str, str] | None):
        self._ip = ip
        self._credentials = credentials
        self._desk: Desk | None = None
        self._reading = _Reading(False, frozenset())
        self._unreadable = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # `renew` swaps the client while the watch thread may read over the old one: unguarded, that thread's failure
        # handler wipes the fresh reading, or the two threads open two Desk logins.
        self._reading_lock = threading.Lock()

    @property
    def triggered(self) -> list[str]:
        """The safe inputs the last reading found triggered; empty where no reading is in hand."""
        reading = self._reading
        return sorted(reading.triggered) if reading.sampled else []

    @property
    def confirmed_clear(self) -> bool:
        """Whether a reading IS in hand and found every safe input clear.

        An unread box answers False, and so does one whose last sample failed: `triggered` flattens
        both to an empty list, and a caller that may not act on an unknown state cannot use it.
        """
        reading = self._reading
        return reading.sampled and not reading.triggered

    def level(self, name: str) -> _SafeInputLevel | None:
        """The level the last reading found for the safe input ``name``; None where no reading is in hand."""
        reading = self._reading
        return reading.levels.get(name) if reading.sampled else None

    @staticmethod
    def _level(reading: object) -> _SafeInputLevel | None:
        """The level Desk sent, or None where Desk sent a value that names no level."""
        try:
            return _SafeInputLevel(reading)
        except ValueError:
            return None

    @staticmethod
    def _triggered(name: str, level: _SafeInputLevel | None) -> bool:
        """Whether the safe input ``name`` is at a level that holds the driver back.

        An input or a level this does not recognise counts as triggered: the driver cannot read it as clear.
        """
        return level not in _SafeInputs._CLEAR_STATES.get(name, frozenset())

    def sample(self) -> None:
        """Take one reading, and log a safe input that changed."""
        with self._reading_lock:
            self._sample()

    def renew(self) -> None:
        """Take a reading over a new Desk client, as after a control box reboot ended the login of the old one."""
        with self._reading_lock:
            self._desk = None
            self._reading = self._reading._replace(sampled=False)
            self._sample()

    def _sample(self) -> None:
        credentials = self._credentials
        if credentials is None:
            return
        try:
            desk = self._desk
            if desk is None:
                desk = Desk(self._ip, *credentials)
                desk._authenticate()  # Desk publishes the read; the login behind it stays underscored
                self._desk = desk
            self._note(desk.safety_status()[SAFE_INPUT_STATE])
        # rules-allow: swallowed-error — a control box that stops answering must not end the run; the
        # reading goes stale instead, and a stale reading names no input.
        except Exception as exc:
            if not self._unreadable:
                logger.error(f'Cannot read the safe inputs: {exc}')
            self._desk, self._unreadable = None, True
            self._reading = self._reading._replace(sampled=False)

    def _note(self, state: Mapping[str, object]) -> None:
        """Record a reading, and log a safe input whose state changed."""
        sampled, was_triggered, _ = self._reading
        if not sampled:
            logger.info(f'The control box reports its safe inputs as {dict(state)}')
        self._unreadable = False
        levels = {name: self._level(reading) for name, reading in state.items()}
        triggered = frozenset(name for name, level in levels.items() if self._triggered(name, level))
        if triggered != was_triggered:
            if triggered:
                logger.warning(f'The control box prohibits motion: safe inputs {sorted(triggered)} are triggered')
            else:
                logger.info('The control box permits motion: every safe input is clear')
        self._reading = _Reading(True, triggered, levels)

    def __enter__(self) -> '_SafeInputs':
        """Take the first reading, then keep it fresh on a thread until the block ends."""
        if self._credentials is not None:
            self.sample()
            self._thread = threading.Thread(target=self._sample_until_stopped, name='franka-safe-inputs', daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._POLL_S * 4)
            self._thread = None

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self._POLL_S):
            self.sample()


class _Arm(DriverRun[command.CommandType]):
    """The arm the driver drives: the robot handle while the driver has a connection, and the state and moves that
    go with it."""

    # A move needs a deadline at all because the robot reports a goal it abandons, but a goal it never
    # converges on stays in flight for as long as the arm is pushed off course.
    # FR3 joint velocity limits in rad/s, from the bundled ``fr3.urdf``; ``relative_dynamics_factor`` scales them
    _MAX_JOINT_VELOCITY = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
    # On top of the travel itself: the robot's controller ramps in and out of its speed cap, and settles late
    _MOVE_GRACE_S = 5.0
    # How long the arm must accept moves again before the count of the moves it refused is logged.
    _REFUSAL_QUIET_S = 2.0
    # How long an error the recovery does not clear holds a waiting sync move before it is refused.
    _RECOVERY_GRACE_S = 2.0
    _TARGET_MARGIN_INSIDE_SOFT_LIMITS_RAD = 0.05
    # The FR3 model the joint names and the soft limits come from, with or without a connection to the arm.
    _FR3_URDF = (Path(__file__).resolve().parent / 'fr3.urdf').read_text()

    def __init__(
        self,
        sync_move: pimm.calls.ControlSystemHandler[command.CommandType, None],
        async_move: pimm.SignalReceiver[command.CommandType],
        out: pimm.SignalEmitter[FrankaState],
        dynamics_factor: float,
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
        safe_inputs: _SafeInputs,
    ):
        super().__init__(sync_move, async_move, should_stop, clock, hz=2000)
        self._robot: pf.Robot | None = None
        # Why the driver has no connection to the arm; None while ``robot`` drives the arm.
        self.disconnect_cause: Exception | None = RuntimeError('the driver has not connected yet')
        self.out = out
        self.state = FrankaState()
        self._dynamics_factor = dynamics_factor
        self.safe_inputs = safe_inputs
        self._refusals = 0
        self._refused = False
        self._quiet_at = 0.0
        # The error last emitted in place of the state, so that each one goes out once.
        self._emitted_error: str | None = None
        # A sync move that arrived while the arm errors, held until the error clears or the grace runs out.
        self._held: pimm.calls.Call[command.CommandType, None] | None = None
        self._error_started = 0.0
        self._q_at_error = np.zeros(len(_PARK_JOINTS))
        self._moved_in_error = 0.0
        model = ET.fromstring(self._FR3_URDF)
        self._joint_names = [joint.get('name') for joint in model.findall("joint[@type='revolute']")]
        soft_limits = model.findall("joint[@type='revolute']/safety_controller")
        margin = self._TARGET_MARGIN_INSIDE_SOFT_LIMITS_RAD
        self._lowest_target = np.array([float(c.attrib['soft_lower_limit']) for c in soft_limits]) + margin
        self._highest_target = np.array([float(c.attrib['soft_upper_limit']) for c in soft_limits]) - margin
        self._clamps = 0

    def __enter__(self) -> '_Arm':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Answer a move still held, then halt the control thread."""
        if self._held is not None:
            self._held.set_exception(MoveAbandoned())
            self._held = None
        # Before the Desk session deactivates FCI, or the thread dies mid-control with
        # "TCP connection got interrupted".
        self.halt()

    @property
    def robot(self) -> pf.Robot:
        """The configured handle that drives the arm."""
        assert self._robot is not None, f'{NO_CONNECTION}: {self.disconnect_cause}'
        return self._robot

    @property
    def connected(self) -> bool:
        return self._robot is not None

    def connect(self, robot: pf.Robot) -> None:
        """Drive the arm over ``robot``, a handle already configured."""
        self._robot, self.disconnect_cause = robot, None

    def disconnect(self, cause: Exception) -> None:
        """Drive the arm no more, for ``cause``, and stream the error that says so."""
        self._robot, self.disconnect_cause = None, cause
        self.emit_state()

    def halt(self) -> None:
        """Stop the control loop, where there is a connection that runs one."""
        if self._robot is not None:
            self._robot.stop()

    def publish(self, st: pf.State) -> None:
        """Ship the arm as it is reported, marked ERROR while it is not where the driver put it and no clear a
        caller asked for has worked since."""
        faulted = self.moves.errored or st.error != 0  # the robot reports its own faults; a stall is not one
        self.state.encode(st, RobotStatus.ERROR if faulted else RobotStatus.AVAILABLE)
        self.emit_state()

    @property
    def rejects_moves(self) -> bool:
        """Whether the arm refused the goal the last reading found, with no goal sent since."""
        return self._refused

    @property
    def stop_pressed(self) -> bool:
        """Whether the last reading found the emergency stop pressed."""
        return self.safe_inputs.level(EMERGENCY_STOP_INPUT) is _SafeInputLevel.INACTIVE

    @property
    def no_connection(self) -> str:
        """The error that says the driver has no connection to the arm, and why."""
        return f'{NO_CONNECTION}: {self.disconnect_cause}'

    def emit_state(self) -> None:
        """Emit ``self.state``, or an error once in its place: the emergency stop's while the stop is pressed, and
        the missing connection's while the driver has none.

        Once, because a reader keeps the last value and the link queues every error it carries.
        """
        error = EMERGENCY_STOP_PRESSED if self.stop_pressed else None if self.connected else self.no_connection
        if error is None:
            self._emitted_error = None
            self.out.emit(self.state)
        elif error != self._emitted_error:
            # The Message types do not name the SignalError that any message can carry.
            self.out.emit(cast(FrankaState, pimm.SignalError(error)))
            self._emitted_error = error

    def hold_through_error(self, st: pf.State, entered: bool, outcome: 'RecoveryOutcome') -> None:
        """Take one tick of an arm error for the sync move that waits on it.

        It drops a jog the arm cannot move on and holds one waiting sync move. ``next_request`` serves the
        held move once the error clears. ``MoveRefused`` answers it once the error outlasts the grace.
        """
        now = self.clock.now()
        if entered:
            self._error_started, self._q_at_error, self._moved_in_error = now, st.q, 0.0
        self._moved_in_error = max(self._moved_in_error, float(np.max(np.abs(st.q - self._q_at_error))))
        self.moves.drain_async()
        if self._held is None:
            asked = self.moves.next_request()
            self._held = asked if isinstance(asked, pimm.calls.Call) else None
        if outcome is RecoveryOutcome.CLEARED:
            logger.info(f'The arm error cleared after {now - self._error_started:.1f}s')
        elif self._held is not None and now - self._error_started >= self._RECOVERY_GRACE_S:
            logger.warning(
                f'The arm error persists after {self._RECOVERY_GRACE_S}s; refusing the waiting move: '
                f'{st.error_message}, moved {self._moved_in_error:.3f} rad'
            )
            self._held.set_exception(MoveRefused(st.error_message, self._moved_in_error))
            self._held = None

    def next_request(self) -> pimm.calls.Call[command.CommandType, None] | command.CommandType | None:
        """The sync move held through an error, before anything ``moves`` is asked for."""
        if self._held is not None:
            held, self._held = self._held, None
            return held
        return self.moves.next_request()

    def refuse_moves(self) -> None:
        """Answer every waiting sync move with the failure that ended the connection, and drop a streamed one."""
        while isinstance(asked := self.next_request(), pimm.calls.Call):
            asked.set_exception(RuntimeError(self.no_connection))

    @staticmethod
    def _to_pf_mode(mode: command.ControlModeType | None) -> pf.InternalImpedance | pf.SoftwareImpedance:
        """The pf mode carrying ``mode``'s gains; ``None`` — no pin — and a bare ``PositionControl`` take pf's own."""
        match mode:
            case None:
                return pf.InternalImpedance()
            case command.PositionControl(stiffness=stiffness):
                return pf.InternalImpedance() if stiffness is None else pf.InternalImpedance(k_theta=list(stiffness))
            case command.Impedance(kq=kq, kqd=kqd, kx=kx, kxd=kxd):
                return pf.SoftwareImpedance(kq=list(kq), kqd=list(kqd), kx=list(kx), kxd=list(kxd))

    def command_target(self, target: np.ndarray, mode: command.ControlModeType | None) -> None:
        """Put the arm under ``mode`` and publish ``target`` to it, in that order with nothing in between."""
        self.robot.set_control_mode(self._to_pf_mode(mode))  # the robot no-ops a mode already running
        self.robot.set_target_joints(target)
        self._refused = False  # the goal just dispatched is not the one the last reading found refused

    def await_goal(
        self, should_stop: Callable[[], bool], pace: Callable[[], pimm.Command]
    ) -> Generator[pimm.Command, None, MoveStatus]:
        """Poll the goal until the arm arrives, one poll per resume.

        ``pace`` is what to wait between polls, and so how often the goal is asked about.
        """
        while not should_stop():
            goal = self.robot.goal()
            if goal.status == pf.GoalStatus.REACHED:
                return MoveStatus.ARRIVED
            if goal.status != pf.GoalStatus.IN_FLIGHT:
                self.note_refusals(goal)
                raise RuntimeError(f'the arm stopped short of its target: {goal.reason or goal.status}')
            yield pace()
        return MoveStatus.GAVE_UP

    def _travel_s(self, q: np.ndarray, target: np.ndarray) -> float:
        """How long the arm may take to reach ``target``, from the speed its dynamics factor allows."""
        cap = self._MAX_JOINT_VELOCITY * self._dynamics_factor
        return self._MOVE_GRACE_S + float(np.max(np.abs(target - q) / cap))

    def note_refusals(self, goal: pf.Goal) -> None:
        """Log the moves the arm refuses: the first one as it happens, the rest as a count once they stop.

        libfranka prints the same rejection from the control thread, unstamped and outside Python.
        """
        refused = goal.status == pf.GoalStatus.ABORTED
        if refused and not self._refused:
            self._refusals += 1
            self._quiet_at = self.clock.now() + self._REFUSAL_QUIET_S
            if self._refusals == 1:
                triggered = self.safe_inputs.triggered
                cause = f'; safe inputs {triggered} are triggered' if triggered else ''
                logger.warning(f'The arm refused a move: {goal.reason or goal.status}{cause}')
        self._refused = refused
        # A goal the arm reached breaks the streak outright; any other unrefused one has to hold for the quiet time.
        reached = goal.status == pf.GoalStatus.REACHED
        if self._refusals and (reached or (not refused and self.clock.now() >= self._quiet_at)):
            if self._refusals > 1:  # the line above already reported a single one, with its reason
                logger.warning(f'The arm refused {self._refusals} moves in a row; it accepts them again')
            self._refusals = 0

    def recover(self) -> RecoveryOutcome:
        """Run the vendor's error recovery. NOT_CLEARED with no connection: only the Desk step reaches the arm then."""
        if not self.connected:
            return RecoveryOutcome.NOT_CLEARED
        return RecoveryOutcome.CLEARED if self.robot.recover_from_errors() else RecoveryOutcome.NOT_CLEARED

    def forget_missed_target(self) -> None:
        """Read AVAILABLE again after a move that missed: a cleared fault leaves the arm holding where it is."""
        self.moves.errored = False

    def clear_held_fault(self) -> None:
        """Clear a fault the arm holds, so the move about to go out is not rejected before it starts.

        A reflex latches libfranka into ``Reflex`` mode, where the arm rejects every move, and the latch
        outlives the error flags ``state().error`` reads — a refused goal is the only sign of it here.
        Recovery clears the latch and commands no motion; it runs only on a reading that found every safe
        input clear, because a person may hold the arm and only they release it. An unread control box is
        left alone too: a stale or missing reading names no input, which is not the same as a clear one.
        """
        if not self._refused or not self.safe_inputs.confirmed_clear:
            return
        logger.info('The arm is rejecting moves; clearing the fault it holds before it takes another')
        if not self.robot.recover_from_errors():
            raise RuntimeError(
                'the arm holds a fault that rejects every move, and the recovery did not clear it: '
                'clear the error in Desk, then start the run again'
            )

    def move_to(
        self, target: np.ndarray, mode: command.ControlModeType | None, *, at_teardown: bool = False
    ) -> Generator[pimm.Command, None, MoveStatus]:
        """Travel to ``target`` under ``mode``, yielding until it arrives.

        ``at_teardown`` is for the move the driver makes on its way out. The stop is already set by then, so
        heeding it would abandon the move before it began, and recovering from a fault cancels the goal.
        """
        # The first emit must not ship an unfilled state.
        self.state.encode(self.robot.state(), RobotStatus.BUSY)
        self.emit_state()

        deadline = self.clock.now() + self._travel_s(self.state.q, target)

        def expired() -> bool:
            return self.clock.now() >= deadline

        def abandoned() -> bool:
            return self.should_stop.value and not at_teardown

        def should_stop() -> bool:
            return abandoned() or expired()

        try:
            self.clear_held_fault()
            self.command_target(target, mode)
            for wait in self.await_goal(should_stop, self.limiter.wait):
                st = self.robot.state()
                self.state.encode(st, RobotStatus.BUSY)
                self.emit_state()
                if st.error != 0 and not at_teardown:
                    self.robot.recover_from_errors()
                yield wait
            # The loop exits before it polls again, so a goal that landed as the deadline passed is unseen.
            if expired() and (missed := self.robot.goal()).status != pf.GoalStatus.REACHED:
                self.note_refusals(missed)  # the hold target below replaces it, and any refusal it carries
                # The robot still tracks the goal it missed, and would resume the move once the arm comes free.
                self.robot.set_target_joints(self.robot.state().q)
                raise TimeoutError(f'the arm stopped short of {target}')
        except Exception:
            self.moves.errored = True
            raise

        if abandoned():
            return MoveStatus.GAVE_UP
        self.moves.errored = False
        # The poll that reports arrival ends the loop, so the sample before it was taken mid-travel
        self.publish(self.robot.state())
        return MoveStatus.ARRIVED

    def _clamp(self, target: np.ndarray) -> np.ndarray:
        """``target`` a margin inside the soft limits; logs the first of a run of clamped commands, and its length."""
        clamped = np.clip(target, self._lowest_target, self._highest_target)
        past = np.flatnonzero(clamped != target)
        if past.size:
            if not self._clamps:
                asked = ', '.join(f'{self._joint_names[i]} {target[i]:.4f} to {clamped[i]:.4f}' for i in past)
                margin = self._TARGET_MARGIN_INSIDE_SOFT_LIMITS_RAD
                logger.warning(
                    f'A command asks for joints less than {margin} rad inside their soft limits; clamping {asked}'
                )
            self._clamps += 1
        elif self._clamps:
            if self._clamps > 1:  # the first line already reported a single one
                logger.warning(f'Clamped {self._clamps} commands in a row; the next command needed no clamp')
            self._clamps = 0
        return clamped

    def _ik(self, pose: geom.Transform3D) -> np.ndarray:
        """The joints that put the end effector at ``pose``, within the arm's limits."""
        return self.robot.inverse_kinematics_with_limits(np.asarray([*pose.translation, *pose.rotation.as_quat]))

    def to_joints(self, cmd: command.CommandType) -> np.ndarray:
        """The joints ``cmd`` asks for, a margin inside the soft limits, not applied yet.

        Solved here so that a malformed command raises before anything changes; ``command_target``
        applies the result.
        """
        match cmd:
            case command.CartesianPosition(pose):
                target = self._ik(pose)
            case command.CartesianDelta() as delta_cmd:
                target = self._ik(delta_cmd.apply(self.state.ee_pose))
            case command.JointPosition(positions):
                target = np.asarray(positions, dtype=np.float64)
            case command.JointDelta(velocities=joint_delta):
                target = self.state.q + joint_delta
            case other:
                raise NotImplementedError(f'Unsupported command {other}')
        # The robot raises on a bad target too late to name the command; one velocity limit per joint sets the width.
        if np.shape(target) != self._MAX_JOINT_VELOCITY.shape or not np.all(np.isfinite(target)):
            raise ValueError(f'{cmd} does not name a joint target this arm can hold: {target}')
        return self._clamp(target)

    def sync_move(self, call: pimm.calls.Call[command.CommandType, None]) -> Iterator[pimm.Command]:
        """Put the arm where ``call`` asks and answer it once the state saying so is out.

        Answer with the release instruction, and send no target, while the emergency stop is pressed.
        """
        if self.stop_pressed:
            call.set_exception(pimm.SignalError(EMERGENCY_STOP_PRESSED))
            return
        cmd = call.request
        try:
            if (yield from self.move_to(self.to_joints(cmd), cmd.mode)) is MoveStatus.ARRIVED:
                call.set_result(None)
            else:
                call.set_exception(MoveAbandoned())
        except Exception as exc:
            try:
                self.publish(self.robot.state())
            finally:
                call.set_exception(exc)  # an arm the driver cannot read still leaves nobody waiting

    def park(self, *, at_teardown: bool = False) -> Iterator[pimm.Command]:
        """Move the arm to ``_PARK_JOINTS``, logging a failure rather than raising it."""
        try:
            logger.info('Moving the arm to the park pose')
            self.robot.recover_from_errors()
            # The park pose is a long way off, and only the native law shapes the reference on the way there.
            yield from self.move_to(_PARK_JOINTS, None, at_teardown=at_teardown)
        # rules-allow: swallowed-error — a failed park reads ERROR on the arm and ends neither run nor shutdown
        except Exception as exc:
            logger.error(f'The arm did not reach the park pose ({type(exc).__name__}: {exc}); it stays where it stands')


class _DeskSession:
    """The Desk session that holds robot control for the run, opened at first use.

    A control box reboot ends the session, and the next use opens a new one. On exit the open session turns FCI off,
    locks the brakes and releases control.
    """

    def __init__(self, ip: str, credentials: tuple[str, str] | None):
        self._ip = ip
        self._credentials = credentials
        self._stack = contextlib.ExitStack()
        self.desk: Desk | None = None

    @property
    def managed(self) -> bool:
        """Whether the driver runs the Desk session, rather than the operator."""
        return self._credentials is not None

    @property
    def login(self) -> str | None:
        return self._credentials[0] if self._credentials is not None else None

    def take_control(self) -> Desk:
        """The open session, with robot control. Raises ``ControlHeldElsewhere`` while another session holds it."""
        assert self._credentials is not None, 'the driver runs no Desk session'
        if self.desk is None:
            self.desk = self._stack.enter_context(Desk(self._ip, *self._credentials))
        else:
            self.desk.ensure_control()
        return self.desk

    def reboot(self) -> None:
        """Reboot the control box, wait until it settles, and end the session.

        The reboot resets FCI, the brakes and every control token, so it needs no control of its own.
        """
        assert self._credentials is not None, 'the driver runs no Desk session'
        desk = self.desk or Desk(self._ip, *self._credentials)
        try:
            desk.reboot(wait=True)
        finally:
            self.close()

    def close(self) -> None:
        self.desk = None
        self._stack.close()

    def __enter__(self) -> '_DeskSession':
        if self._credentials is None:
            logger.info('Desk is not managed by the driver; brakes must be open and FCI active before the run')
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _Brakes:
    """The arm's joint brakes, and the idle time the driver holds them open for.

    Both a Desk session and an idle time are needed to close them; without either the driver holds them
    open for the whole run.
    """

    def __init__(self, session: _DeskSession, arm: _Arm, clock: pimm.Clock, after_idle_s: float | None):
        self._session = session
        self._arm = arm
        self._clock = clock
        self._after_idle_s = after_idle_s
        self._closed = False
        self._idle_since = clock.now()

    def close(self) -> None:
        """Halt the control loop and lock the brakes."""
        desk = self._session.desk
        assert desk is not None, 'only a Desk session closes the brakes'
        self._arm.halt()  # the brakes cannot engage on an arm the control loop still drives
        desk.close_brakes()
        self._closed = True

    def record_opened(self) -> None:
        """Record brakes that ``Desk.prepare()`` opened, and count the idle time from now."""
        self._closed = False
        self._idle_since = self._clock.now()

    @contextlib.contextmanager
    def opened(self) -> Iterator[None]:
        """Open the brakes for the block, and count the idle time from the moment the block ends."""
        if self._closed:
            desk = self._session.desk
            assert desk is not None, 'only a Desk session closes the brakes'
            logger.info('A command arrived, opening the brakes')
            desk.open_brakes()
            self._closed = False
        try:
            yield
        finally:
            self._idle_since = self._clock.now()

    def close_if_idle(self, goal: pf.Goal) -> None:
        """Close the brakes once the idle time passes with the arm at rest.

        A streamed setpoint is published and done with, so ``goal`` is the only thing that says the arm is
        still travelling towards it.
        """
        if self._closed or self._session.desk is None or self._after_idle_s is None:
            return
        if goal.status == pf.GoalStatus.IN_FLIGHT:
            self._idle_since = self._clock.now()
            return
        if self._clock.now() - self._idle_since < self._after_idle_s:
            return
        logger.info(f'No command and no move for {self._after_idle_s}s, closing the brakes')
        self.close()


# The field Desk answers the state of each joint brake in.
BRAKE_STATE = 'brakeState'
# The field Desk answers the recoverable safety errors in, each by name.
RECOVERABLE_ERRORS = 'recoverableErrors'


class _BrakeState(StrEnum):
    """The state Desk reports for one joint brake."""

    LOCKED = 'Locked'
    UNLOCKED = 'Unlocked'


class Robot(pimm.ControlSystem):
    def __init__(
        self,
        ip: str,
        *,
        relative_dynamics_factor=0.2,
        load: tuple | None = None,
        collision_coeff: float = 2.0,
        manage_desk: bool = True,
        brake_after_idle_s: float | None = None,
    ) -> None:
        """
        The robot model a run sends is the bundled FR3 model, with or without a connection to the arm. An arm whose
        end-effector frame in Desk differs from that model's still records the bundled one.

        :param ip: IP address of the robot.
        :param relative_dynamics_factor: Relative dynamics factor in (0, 1]. Smaller values are more conservative.
        :param collision_coeff: Multiplier for collision thresholds. Higher = more tolerant.
        :param manage_desk: Run the Desk session from the driver: open the brakes and activate FCI on start, close
            them on stop. Requires ``FRANKA_DESK_USER`` and ``FRANKA_DESK_PASSWORD`` in the environment. A start that
            Desk or libfranka refuses leaves the driver running with no connection to the arm, and a recover call
            connects it. Set to False to leave brakes and FCI to the operator; the driver then never contacts Desk,
            expects FCI to be up already, and a failed connection ends the run.
        :param brake_after_idle_s: How long the arm may sit with no command and no move in flight before the driver
            closes the brakes; the next command opens them again. None holds the brakes open for the whole run.
            Needs ``manage_desk``.
        """
        assert 0 < relative_dynamics_factor <= 1, relative_dynamics_factor
        assert brake_after_idle_s is None or brake_after_idle_s > 0, brake_after_idle_s
        if brake_after_idle_s is not None and not manage_desk:
            raise ValueError(
                'brake_after_idle_s needs manage_desk=True: without a Desk session the driver cannot close the brakes'
            )
        self._ip = ip
        self._relative_dynamics_factor = relative_dynamics_factor
        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.ready = pimm.calls.ControlSystemHandler[None, None](self)
        self.state = pimm.ControlSystemEmitter[FrankaState](self)
        self.robot_meta = pimm.ControlSystemEmitter(self)
        # FOOTGUN: recovers whatever ``state().error`` reads, since a latched Reflex reads 0.
        self.recover = pimm.calls.ControlSystemHandler[None, RecoveryOutcome](self)
        # The step a recover call is at. A control box reboot holds one for minutes.
        self.recovery_step = pimm.ControlSystemEmitter[RecoveryStep](self)
        self._load = load
        self._collision_coeff = collision_coeff
        self._desk_credentials = _read_desk_credentials() if manage_desk else None
        self._brake_after_idle_s = brake_after_idle_s

    def _init_robot(self, robot):
        coeff = self._collision_coeff
        torque_threshold_acceleration = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
        torque_threshold_nominal = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        force_threshold_acceleration = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        force_threshold_nominal = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        robot.set_collision_behavior(
            lower_torque_threshold_acceleration=(coeff * torque_threshold_acceleration).tolist(),
            upper_torque_threshold_acceleration=(coeff * torque_threshold_acceleration).tolist(),
            lower_torque_threshold_nominal=(coeff * torque_threshold_nominal).tolist(),
            upper_torque_threshold_nominal=(coeff * torque_threshold_nominal * 2).tolist(),
            lower_force_threshold_acceleration=(coeff * force_threshold_acceleration).tolist(),
            upper_force_threshold_acceleration=(coeff * force_threshold_acceleration).tolist(),
            lower_force_threshold_nominal=(coeff * force_threshold_nominal).tolist(),
            upper_force_threshold_nominal=(coeff * force_threshold_nominal * 2).tolist(),
        )
        robot.set_control_mode(_Arm._to_pf_mode(None))
        if self._load is not None:
            logger.info(f'Setting load to {self._load}')
            robot.set_load(*self._load)
        else:
            robot.set_load(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    @functools.cached_property
    def _robot(self) -> pf.Robot:
        """The libfranka handle, connected on first use."""
        return pf.Robot(
            self._ip, realtime_config=pf.RealtimeConfig.Ignore, relative_dynamics_factor=self._relative_dynamics_factor
        )

    def _forget_connection(self) -> None:
        """Drop the libfranka handle, so that the next connection opens a new one."""
        with contextlib.suppress(AttributeError):  # no handle is cached until the first connection
            del self._robot

    def _arm(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock, safe_inputs: _SafeInputs) -> _Arm:
        """The arm this run drives, built from the driver's configuration, with no connection yet."""
        return _Arm(
            self.sync_move, self.commands, self.state, self._relative_dynamics_factor, should_stop, clock, safe_inputs
        )

    def _connect(self, arm: _Arm) -> None:
        """Configure the libfranka handle, and drive the arm over it. Only a configured handle drives the arm.

        A handle that fails its configuration is dropped, so that the next connection opens a new one.
        """
        robot = self._robot
        try:
            self._init_robot(robot)
        except Exception:
            try:
                robot.stop()  # the control loop that `set_control_mode` started
            finally:
                self._forget_connection()
            raise
        arm.connect(robot)

    def _start(self, session: _DeskSession, arm: _Arm, brakes: _Brakes) -> None:
        """Prepare the arm through Desk and connect to it, as a run starts.

        Where the driver runs the Desk session, a failure leaves the run with no connection. Without one, the
        failure ends the run.
        """
        if not session.managed:
            self._connect(arm)
            return
        try:
            session.take_control().prepare()
            brakes.record_opened()
            self._connect(arm)
        # rules-allow: swallowed-error — the arm streams the failure, and a recover call connects again
        except Exception as exc:
            # One line: a run log reads a traceback as a crash, and this run goes on.
            logger.error(
                f'The arm did not come up ({type(exc).__name__}: {exc}); '
                'the run goes on with no connection until a recover call'
            )
            arm.disconnect(exc)
            self._lock_open_brakes(session, brakes)

    @staticmethod
    def _lock_open_brakes(session: _DeskSession, brakes: _Brakes) -> None:
        """Lock the brakes a failed Desk step left open: with no connection nothing drives the arm."""
        if session.desk is None:
            return  # the brakes are the session's that holds control
        try:
            if _BrakeState.UNLOCKED in session.desk.safety_status()[BRAKE_STATE]:
                brakes.close()
        # rules-allow: swallowed-error — the caller gets the step's own failure, and the session teardown locks them
        except Exception as exc:
            logger.error(f'Cannot lock the brakes that the failed Desk step left open ({type(exc).__name__}: {exc})')

    def _prepare_and_connect(self, session: _DeskSession, arm: _Arm, brakes: _Brakes) -> RecoveryOutcome:
        """Take Desk control again, lock the brakes and turn FCI off, run ``Desk.prepare()`` as a run starts, connect
        to the arm again, and run the recovery on the new connection.

        A failure once FCI is off leaves the driver with no connection to the arm.
        """
        desk = session.take_control()
        arm.halt()  # no control loop may drive the arm when FCI goes down
        # A self-test that Desk.prepare() runs needs locked brakes. A safety stop can lock them before the driver does.
        if _BrakeState.UNLOCKED in desk.safety_status()[BRAKE_STATE]:
            brakes.close()
        # With no connection FCI is off, and a box in Recovery refuses requests until prepare() acknowledges the error.
        if arm.connected:
            desk.deactivate_fci()
            self._forget_connection()  # FCI off ended the connection this handle holds
        try:
            desk.prepare()
            brakes.record_opened()
            self._connect(arm)
        except Exception as exc:
            arm.disconnect(exc)
            self._lock_open_brakes(session, brakes)
            raise
        cleared = arm.robot.recover_from_errors()
        logger.info(f'After Desk prepared the arm, recover_from_errors returned {cleared}')
        return RecoveryOutcome.CLEARED if cleared else RecoveryOutcome.NOT_CLEARED

    @staticmethod
    def _safe_to_prepare(arm: _Arm) -> bool:
        """Whether a reading found every safe input clear: a person may hold the arm, and only they release it."""
        if not arm.safe_inputs.confirmed_clear:
            logger.warning('No reading shows every safe input clear, so Desk does not prepare the arm again')
        return arm.safe_inputs.confirmed_clear

    @staticmethod
    def _reboot_clears(session: _DeskSession, exc: Exception) -> str | None:
        """Why only a control box reboot clears ``exc``, the failure of a Desk step; None where a reboot does not.

        A control token of the driver's own Desk login that the session cannot take is stranded. Every actor that
        shares the login reads as that name, so the reboot also stops a run that one of them started by hand.
        """
        if isinstance(exc, SafetyControllerError):
            return 'the safety controller is in SafetyError'
        if isinstance(exc, ControlHeldElsewhere):
            # The checker runs without the Franka package, so it cannot narrow `exc` to its error.
            holder = exc.holder  # pyright: ignore[reportAttributeAccessIssue]
            return 'a control token of this Desk login is stranded' if holder == session.login else None
        if session.desk is not None and Robot._reports_fsoe_error(session.desk):
            return f'Desk reports {FSOE_CONNECTION_ERROR}'
        return None

    @staticmethod
    def _reports_fsoe_error(desk: Desk) -> bool:
        try:
            return bool(desk.safety_status()[RECOVERABLE_ERRORS].get(FSOE_CONNECTION_ERROR))
        # rules-allow: swallowed-error — an unread status names no error, and the caller raises the step's own failure
        except Exception:
            return False

    def _clear_with_desk_prepare(self, session: _DeskSession, arm: _Arm, brakes: _Brakes) -> RecoveryOutcome:
        """Run ``_prepare_and_connect`` where the driver runs the Desk session and every safe input reads clear.

        A failure that only a reboot clears reboots the control box once, and ``recovery_step`` says so. The step then
        reads the safe inputs again and runs once more.
        """
        if not session.managed or not self._safe_to_prepare(arm):
            return RecoveryOutcome.NOT_CLEARED
        logger.warning('The recovery did not clear the fault; preparing the arm through Desk as a run starts')
        self.recovery_step.emit(RecoveryStep.CONNECTING)
        try:
            return self._prepare_and_connect(session, arm, brakes)
        except Exception as exc:
            if (why := self._reboot_clears(session, exc)) is None:
                raise
            refused = exc
        logger.warning(f'Desk refused the step: {why}. Rebooting the control box, then preparing the arm again')
        self.recovery_step.emit(RecoveryStep.REBOOTING)
        arm.halt()
        arm.disconnect(refused)
        self._forget_connection()  # the reboot ends every connection
        session.reboot()
        arm.safe_inputs.renew()
        if not self._safe_to_prepare(arm):
            return RecoveryOutcome.NOT_CLEARED
        self.recovery_step.emit(RecoveryStep.CONNECTING)
        return self._prepare_and_connect(session, arm, brakes)

    def _recover(
        self, session: _DeskSession, arm: _Arm, brakes: _Brakes, asked: list[pimm.calls.Call[None, RecoveryOutcome]]
    ) -> RecoveryOutcome:
        """Run the arm's error recovery once, answer every caller that asked for it on this tick, and return
        what it did.

        When a caller asked and the recovery leaves the fault, ``_clear_with_desk_prepare`` runs. A throw reaches
        the callers that asked and counts as NOT_CLEARED; one nobody asked for reaches no caller, so it ends the run.
        """
        try:
            outcome = arm.recover()
            if asked:
                logger.info(f'A console asked to clear a fault; the recovery returned {outcome.name}')
            if asked and outcome is RecoveryOutcome.NOT_CLEARED:
                outcome = self._clear_with_desk_prepare(session, arm, brakes)
            if asked and outcome is RecoveryOutcome.CLEARED:
                arm.forget_missed_target()
        # rules-allow: swallowed-error — the throw is handed to every caller that asked
        except Exception as exc:
            if not asked:
                raise
            logger.error(f'The recovery a console asked for failed ({type(exc).__name__}: {exc})')
            for call in asked:
                call.set_exception(exc)
            return RecoveryOutcome.NOT_CLEARED
        finally:
            if asked:
                self.recovery_step.emit(RecoveryStep.IDLE)
        for call in asked:
            call.set_result(outcome)
        return outcome

    def _ready(self, arm: _Arm, call: pimm.calls.Call[None, None]) -> None:
        """Answer ``call`` once the arm takes moves, clearing an error or a fault it holds first.

        Answer with the error where the recovery leaves it, and with the missing connection where there is none. A
        pressed emergency stop gets the release instruction, and no recovery. A ready call takes no Desk control and
        opens no brakes: the harness asks again and again between episodes, with nobody at the rig.
        """
        with pimm.calls.raise_to(call):
            if arm.stop_pressed:
                raise pimm.SignalError(EMERGENCY_STOP_PRESSED)
            if not arm.connected:
                raise RuntimeError(arm.no_connection)
            if arm.robot.state().error != 0 and arm.recover() is RecoveryOutcome.NOT_CLEARED:
                remaining = arm.robot.state().error_message
                raise RuntimeError(f'the arm holds an error that the recovery did not clear: {remaining}')
            arm.clear_held_fault()
            if arm.rejects_moves and not arm.safe_inputs.confirmed_clear:
                triggered = arm.safe_inputs.triggered
                cause = (
                    f'safe inputs {triggered} are triggered' if triggered else 'no reading shows every safe input clear'
                )
                raise RuntimeError(f'the arm rejects every move, and its fault stays: {cause}')
            call.set_result(None)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        safe_inputs = _SafeInputs(self._ip, self._desk_credentials)
        session = _DeskSession(self._ip, self._desk_credentials)
        with session, safe_inputs, self._arm(should_stop, clock, safe_inputs) as arm:
            self.robot_meta.emit(bundled_franka_model())
            self.recovery_step.emit(RecoveryStep.IDLE)
            brakes = _Brakes(session, arm, clock, self._brake_after_idle_s)
            self._start(session, arm, brakes)

            if arm.connected:
                with brakes.opened():
                    yield from arm.park()

            in_error = False

            while not should_stop.value:
                for call in self.ready.incoming():
                    self._ready(arm, call)
                if not arm.connected:
                    arm.emit_state()  # the emergency stop's error comes and goes over the missing connection's
                    arm.refuse_moves()
                    if asked_to_recover := list(self.recover.incoming()):
                        self._recover(session, arm, brakes, asked_to_recover)
                    yield arm.limiter.wait()
                    continue
                st = arm.robot.state()
                arm.publish(st)
                goal = arm.robot.goal()
                arm.note_refusals(goal)

                in_error, entered_error = _check_error(st.error != 0, in_error)
                if entered_error:
                    logger.warning(f'Robot error: {st.error_message}')

                asked_to_recover = list(self.recover.incoming())
                if asked_to_recover or in_error:
                    outcome = self._recover(session, arm, brakes, asked_to_recover)
                    if in_error:
                        arm.hold_through_error(st, entered_error, outcome)
                    # This tick commands nothing; the next one reads the arm the recovery left behind.
                    yield arm.limiter.wait()
                    continue

                asked = arm.next_request()
                if isinstance(asked, pimm.calls.Call):
                    with brakes.opened():
                        yield from arm.sync_move(asked)
                elif asked is not None:
                    with brakes.opened(), log_failure(asked):
                        arm.command_target(arm.to_joints(asked), asked.mode)
                else:
                    brakes.close_if_idle(goal)

                yield arm.limiter.wait()

            if arm.connected:
                with brakes.opened():
                    yield from arm.park(at_teardown=True)


if __name__ == '__main__':
    with pimm.World() as world:
        robot = Robot('172.168.0.2', relative_dynamics_factor=0.2)
        commands = world.pair(robot.commands)
        state = world.pair(robot.state)
        world.start([], background=robot)

        trajectory = [
            ([0.03, 0.03, 0.03], 0.0),
            ([-0.03, 0.03, 0.03], 2.0),
            ([-0.03, -0.03, 0.03], 4.0),
            ([-0.03, -0.03, -0.03], 6.0),
            ([0.03, -0.03, -0.03], 8.0),
            ([0.03, 0.03, -0.03], 10.0),
            ([0.03, 0.03, 0.03], 12.0),
        ]

        while not world.should_stop and (state.read() is None or state.value.status == RobotStatus.BUSY):
            time.sleep(0.01)

        origin = state.value.ee_pose
        print(f'Origin: {origin}')

        alpha = 3.0
        start, i = time.monotonic(), 0
        while i < len(trajectory) and not world.should_stop:
            pos, duration = trajectory[i]
            pos = np.asarray(pos) * alpha
            if time.monotonic() > start + duration:
                print(f'Moving to {pos + origin.translation}')
                target = command.CartesianPosition(geom.Transform3D(pos + origin.translation, origin.rotation))
                commands.emit(target)
                i += 1
            else:
                time.sleep(0.01)

        print('Finishing')
