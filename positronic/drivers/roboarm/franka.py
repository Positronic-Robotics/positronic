import contextlib
import functools
import logging
import os
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Generator, Iterator, Mapping
from enum import Enum, auto
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

import pimm
from positronic import geom
from positronic.drivers import vendor_import
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.utils import DriverRun, MoveAbandoned, MoveStatus, log_failure

from . import RobotStatus, State, command
from .models import DEFAULT_FRAME, EE_LINK, add_default_frame, attach_robotiq_2f85

with vendor_import('positronic_franka', 'Franka support', platforms=('linux',)):
    import positronic_franka._franka as pf
    from positronic_franka.desk import Desk, SafetyControllerError

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


def _revolute_joint_names(urdf_xml):
    root = ET.fromstring(urdf_xml)
    return [j.get('name') for j in root.findall('joint') if j.get('type') == 'revolute']


_MESH_DIR = Path(__file__).resolve().parent.parent.parent / 'assets/fr3_collision'
# Where the driver leaves the arm: taking control it travels here, and handing it back it returns here.
_PARK_JOINTS = np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0])

# How often the watch thread reads the safe inputs, and how often a move waiting on one reads them.
_SAFE_INPUT_POLL_S = 0.5
# The field Desk answers the safe inputs in.
SAFE_INPUT_STATE = 'safeInputState'


class _Reading(NamedTuple):
    """One reading of the safe inputs, published in one assignment so a reader never sees half of it."""

    sampled: bool
    trips: int
    triggered: frozenset[str]


class _SafeInputs:
    """The control box's safe inputs, which libfranka does not report and Desk does.

    A thread reads them every ``_SAFE_INPUT_POLL_S``; a caller takes a reading of its own with
    ``sample``. Both go over a Desk client this owns, since the read needs no control token and the
    session that drives the arm must stay on one thread.
    """

    # Desk's own words for a safe input that permits motion. The control box answers a phrase, and its
    # safety log records the same two: 'Not triggered (Motion permitted)' and 'Triggered (Motion prohibited)'.
    _MOTION_PERMITTED = 'not triggered'

    def __init__(self, ip: str, credentials: tuple[str, str] | None):
        self._ip = ip
        self._credentials = credentials
        self._desk: Desk | None = None
        self._reading = _Reading(False, 0, frozenset())
        self._unreadable = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def mark(self) -> int:
        """What the command going out hands ``tripped_since`` when it fails.

        One below the count where a live reading finds an input triggered, so the trip under way counts
        as during the move it is about to refuse: it may clear again before anything samples next, and
        nothing would then attribute the refusal to it. A stale reading backdates nothing: the trip it
        holds may have cleared long before the move began.
        """
        reading = self._reading
        return reading.trips - 1 if reading.sampled and reading.triggered else reading.trips

    @property
    def motion_permitted(self) -> bool:
        """Whether the last reading found every safe input clear."""
        reading = self._reading
        return reading.sampled and not reading.triggered

    def tripped_since(self, since: int) -> bool:
        """Whether a safe input has been triggered since ``since`` trips, or is triggered now.

        False while no reading is in hand: without one there is nothing to attribute anything to.
        """
        reading = self._reading
        return reading.sampled and (reading.trips > since or bool(reading.triggered))

    @staticmethod
    def _triggered(reading: object) -> bool:
        """Whether Desk reports a safe input as triggered.

        A reading this does not recognise counts as triggered, which is the safe direction: it leaves
        every move the outcome it has where nothing reads the safe inputs at all.
        """
        return _SafeInputs._MOTION_PERMITTED not in str(reading).casefold()

    def sample(self) -> None:
        """Take one reading, and log a safe input that changed."""
        credentials = self._credentials
        if credentials is None:
            return
        with self._lock:
            try:
                desk = self._desk
                if desk is None:
                    desk = Desk(self._ip, *credentials)
                    desk._authenticate()  # Desk publishes the read; the login behind it stays underscored
                    self._desk = desk
                self._note(desk.safety_status()[SAFE_INPUT_STATE])
            # rules-allow: swallowed-error — a control box that stops answering must not end the run. The reading
            # goes stale instead, which leaves every move the outcome it has where Desk is unmanaged.
            except Exception as exc:
                if not self._unreadable:
                    logger.error(f'Cannot read the safe inputs: {exc}')
                self._desk, self._unreadable = None, True
                self._reading = self._reading._replace(sampled=False)

    def _note(self, state: Mapping[str, object]) -> None:
        """Record a reading, and log a safe input whose state changed."""
        sampled, trips, was_triggered = self._reading
        if not sampled:
            logger.info(f'The control box reports its safe inputs as {dict(state)}')
        self._unreadable = False
        triggered = frozenset(name for name, reading in state.items() if self._triggered(reading))
        if triggered != was_triggered:
            if triggered and not was_triggered:
                trips += 1
            if triggered:
                logger.warning(f'The control box prohibits motion: safe inputs {sorted(triggered)} are triggered')
            else:
                logger.info('The control box permits motion: every safe input is clear')
        self._reading = _Reading(True, trips, triggered)

    def __enter__(self) -> '_SafeInputs':
        if self._credentials is not None:
            self._thread = threading.Thread(target=self._sample_until_stopped, name='franka-safe-inputs', daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_SAFE_INPUT_POLL_S * 4)
            self._thread = None

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(_SAFE_INPUT_POLL_S):
            self.sample()


class _StoppedShort(RuntimeError):
    """The arm ended a goal without reaching it: how it reports a command it would not take.

    Its own type because it is the failure a triggered safe input produces: the control box refuses
    every command while one is triggered.
    """


class _SafeStopWait(Enum):
    """What a wait on the safe inputs leaves the move to do."""

    MAKE_THE_MOVE_AGAIN = auto()
    KEEP_THE_FAILURE = auto()


class _Arm(DriverRun[command.CommandType]):
    """The arm the driver drives: the robot handle, and the state and moves that go with it."""

    # A move needs a deadline at all because the robot reports a goal it abandons, but a goal it never
    # converges on stays in flight for as long as the arm is pushed off course.
    # FR3 joint velocity limits in rad/s, from the bundled ``fr3.urdf``; ``relative_dynamics_factor`` scales them
    _MAX_JOINT_VELOCITY = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
    # On top of the travel itself: the robot's controller ramps in and out of its speed cap, and settles late
    _MOVE_GRACE_S = 5.0
    # How long a move waits for a triggered safe input to clear before it fails. The longest interval a safe
    # input stayed triggered in this rig's safety log is 4 s.
    _SAFE_STOP_WAIT_S = 15.0
    # How many times one move may be made again after a safe input stopped it.
    _SAFE_STOP_RETRIES = 2
    # How long the arm must accept moves again before the count of the moves it refused is logged.
    _REFUSAL_QUIET_S = 2.0

    def __init__(
        self,
        robot: pf.Robot,
        sync_move: pimm.calls.ControlSystemHandler[command.CommandType, None],
        async_move: pimm.SignalReceiver[command.CommandType],
        out: pimm.SignalEmitter[FrankaState],
        dynamics_factor: float,
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
        safe_inputs: _SafeInputs,
    ):
        super().__init__(sync_move, async_move, should_stop, clock, hz=2000)
        self.robot = robot
        self.out = out
        self.state = FrankaState()
        self._dynamics_factor = dynamics_factor
        self.safe_inputs = safe_inputs
        self._refusals = 0
        self._refused = False
        self._quiet_at = 0.0
        self._mark = 0

    def __enter__(self) -> '_Arm':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Halt the control thread."""
        # Before ``_desk_session`` deactivates FCI, or the thread dies mid-control with
        # "TCP connection got interrupted".
        self.robot.stop()

    def publish(self, st: pf.State) -> None:
        """Ship the arm as it is reported, marked ERROR while it is not where the driver put it."""
        faulted = self.moves.errored or st.error != 0  # the robot reports its own faults; a stall is not one
        self.state.encode(st, RobotStatus.ERROR if faulted else RobotStatus.AVAILABLE)
        self.out.emit(self.state)

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
        # Taken as the command goes out, so only an input triggered then can be what refuses this one.
        self._mark = self.safe_inputs.mark
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
                raise _StoppedShort(f'the arm stopped short of its target: {goal.reason or goal.status}')
            yield pace()
        return MoveStatus.GAVE_UP

    def _travel_s(self, q: np.ndarray, target: np.ndarray) -> float:
        """How long the arm may take to reach ``target``, from the speed its dynamics factor allows."""
        cap = self._MAX_JOINT_VELOCITY * self._dynamics_factor
        return self._MOVE_GRACE_S + float(np.max(np.abs(target - q) / cap))

    def note_refusals(self) -> None:
        """Log the moves the arm refuses: the first one as it happens, the rest as a count once they stop.

        libfranka prints its own rejection from the control thread, unstamped and outside Python.
        ``Goal.reason`` carries the same words where the logger stamps them.
        """
        goal = self.robot.goal()
        refused = goal.status is pf.GoalStatus.ABORTED
        if refused and not self._refused:
            self._refusals += 1
            self._quiet_at = self.clock.now() + self._REFUSAL_QUIET_S
            if self._refusals == 1:
                logger.warning(f'The arm refused a move: {goal.reason or goal.status}')
        self._refused = refused
        # A goal that is not refused is what makes the summary true; without one the arm is still refusing.
        if self._refusals and not refused and self.clock.now() >= self._quiet_at:
            if self._refusals > 1:  # the line above already reported a single one, with its reason
                logger.warning(f'The arm refused {self._refusals} moves in a row; it accepts them again')
            self._refusals = 0

    def _await_safe_stop(self, since: int, *, at_teardown: bool) -> Generator[pimm.Command, None, _SafeStopWait]:
        """Read the safe inputs, then yield until one that tripped after ``since`` clears.

        ``KEEP_THE_FAILURE`` where nothing attributes the failure to a safe input, where the input
        stayed triggered for ``_SAFE_STOP_WAIT_S``, and where the world came down while it waited.
        """
        watch = self.safe_inputs
        watch.sample()
        if not watch.tripped_since(since):
            return _SafeStopWait.KEEP_THE_FAILURE
        wait_s = self._SAFE_STOP_WAIT_S
        logger.warning(f'A safe input stopped the move; waiting up to {wait_s:.0f}s for it to clear')
        deadline = self.clock.now() + wait_s
        while True:
            # The stop is read before the reading, so a world coming down as the input clears fails the
            # move rather than commanding a fresh one on its way out.
            if self.should_stop.value and not at_teardown:
                logger.warning('The world stopped before the move could be made again; the move fails')
                return _SafeStopWait.KEEP_THE_FAILURE
            if watch.motion_permitted:
                return _SafeStopWait.MAKE_THE_MOVE_AGAIN
            if self.clock.now() >= deadline:
                logger.error(f'A safe input stayed triggered for {wait_s:.0f}s; the move fails')
                return _SafeStopWait.KEEP_THE_FAILURE
            yield pimm.Sleep(_SAFE_INPUT_POLL_S)
            watch.sample()

    def move_to(
        self, target: np.ndarray, mode: command.ControlModeType | None, *, at_teardown: bool = False
    ) -> Generator[pimm.Command, None, MoveStatus]:
        """Travel to ``target`` under ``mode``, yielding until it arrives.

        A move the arm stopped short of its target is made again, up to ``_SAFE_STOP_RETRIES`` times, and
        only where a safe input was triggered during it and has since cleared. A real stop LATCHES the
        input until a person releases it, so one that clears on its own moved nothing. Every other
        failure — a deadline the arm ran past, a control box that stopped answering — reaches the caller,
        since one the driver cannot attribute may have moved the arm.

        ``at_teardown`` is for the move the driver makes on its way out. The stop is already set by then, so
        heeding it would abandon the move before it began, and recovering from a fault cancels the goal.
        """
        for _ in range(self._SAFE_STOP_RETRIES):
            try:
                return (yield from self._travel_to(target, mode, at_teardown=at_teardown))
            except _StoppedShort:
                # Read while the refused goal still carries its reason: the run loop does not tick inside a
                # move, and both the wait below and the retry after it leave nothing for it to find.
                self.note_refusals()
                outcome = yield from self._await_safe_stop(self._mark, at_teardown=at_teardown)
                if outcome is _SafeStopWait.KEEP_THE_FAILURE:
                    raise
                self.robot.recover_from_errors()  # the stop faulted the arm; the target is unchanged
                logger.warning('Every safe input is clear; making the move again')
        return (yield from self._travel_to(target, mode, at_teardown=at_teardown))

    def _travel_to(
        self, target: np.ndarray, mode: command.ControlModeType | None, *, at_teardown: bool = False
    ) -> Generator[pimm.Command, None, MoveStatus]:
        """One try at ``target``: command it, and yield until the arm arrives or stops short."""
        # The first emit must not ship an unfilled state.
        self.state.encode(self.robot.state(), RobotStatus.BUSY)
        self.out.emit(self.state)

        deadline = self.clock.now() + self._travel_s(self.state.q, target)

        def expired() -> bool:
            return self.clock.now() >= deadline

        def abandoned() -> bool:
            return self.should_stop.value and not at_teardown

        def should_stop() -> bool:
            return abandoned() or expired()

        try:
            self.command_target(target, mode)
            for wait in self.await_goal(should_stop, self.limiter.wait):
                st = self.robot.state()
                self.state.encode(st, RobotStatus.BUSY)
                self.out.emit(self.state)
                if st.error != 0 and not at_teardown:
                    self.robot.recover_from_errors()
                yield wait
            # The loop exits before it polls again, so a goal that landed as the deadline passed is unseen.
            if expired() and self.robot.goal().status != pf.GoalStatus.REACHED:
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

    def _ik(self, pose: geom.Transform3D) -> np.ndarray:
        """The joints that put the end effector at ``pose``, within the arm's limits."""
        return self.robot.inverse_kinematics_with_limits(np.asarray([*pose.translation, *pose.rotation.as_quat]))

    def to_joints(self, cmd: command.CommandType) -> np.ndarray:
        """The joints ``cmd`` asks for, not applied yet.

        Solved here so that a malformed command raises before anything changes; ``command_target``
        applies the result. ``_ik`` keeps a Cartesian command inside the joint limits; a joint-space
        one is unbounded.
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
        return target

    def sync_move(self, call: pimm.calls.Call[command.CommandType, None]) -> Iterator[pimm.Command]:
        """Put the arm where ``call`` asks and answer it once the state saying so is out."""
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
        except Exception:
            logger.exception('The arm did not reach the park pose, it stays where it stands')


class Robot(pimm.ControlSystem):
    def __init__(
        self,
        ip: str,
        *,
        relative_dynamics_factor=0.2,
        load: tuple | None = None,
        collision_coeff: float = 2.0,
        manage_desk: bool = True,
        reboot_on_safety_error: bool = False,
    ) -> None:
        """
        :param ip: IP address of the robot.
        :param relative_dynamics_factor: Relative dynamics factor in (0, 1]. Smaller values are more conservative.
        :param collision_coeff: Multiplier for collision thresholds. Higher = more tolerant.
        :param manage_desk: Run the Desk session from the driver: open the brakes and activate FCI on start, close
            them on stop. Requires ``FRANKA_DESK_USER`` and ``FRANKA_DESK_PASSWORD`` in the environment. Set to
            False to leave brakes and FCI to the operator; the driver then never contacts Desk and expects FCI to
            be up already.
        :param reboot_on_safety_error: When the control box is in an unrecoverable ``SafetyError`` on start, reboot
            it, wait for it to come back, and try once more before giving up. Only applies when ``manage_desk`` is
            set.
        """
        assert 0 < relative_dynamics_factor <= 1, relative_dynamics_factor
        self._ip = ip
        self._relative_dynamics_factor = relative_dynamics_factor
        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.state = pimm.ControlSystemEmitter[FrankaState](self)
        self.robot_meta = pimm.ControlSystemEmitter(self)
        self._load = load
        self._collision_coeff = collision_coeff
        self._desk_credentials = _read_desk_credentials() if manage_desk else None
        self._reboot_on_safety_error = reboot_on_safety_error

    @staticmethod
    def _build_robot_meta(robot) -> dict:
        urdf_xml = robot.get_robot_model()
        meshes = {f.name: f.read_bytes() for f in _MESH_DIR.iterdir() if f.suffix == '.stl'}
        root = ET.fromstring(urdf_xml)
        # Rewrite mesh references to bare filenames matching the bundled STL meshes.
        # The libfranka URDF uses package:// URIs (e.g. package://franka_description/.../link0.dae)
        # and references .dae visual meshes we don't bundle. Strip unresolvable elements and
        # normalise remaining refs so the URDF is self-contained — matching the style of
        # fr3.urdf used by cfg.ds.internal for offline dataset transforms.
        for link in root.findall('link'):
            for tag in ('visual', 'collision'):
                for vc_el in list(link.findall(tag)):
                    mesh_el = vc_el.find('.//mesh')
                    if mesh_el is None:
                        continue
                    basename = Path(mesh_el.get('filename', '')).name
                    if basename in meshes:
                        mesh_el.set('filename', basename)
                    else:
                        link.remove(vc_el)
            stl = link.get('name', '') + '.stl'
            if stl in meshes and link.find('visual') is None:
                ET.SubElement(ET.SubElement(link, 'visual'), 'geometry').append(ET.Element('mesh', filename=stl))
        # TODO: The arm driver should not own the gripper. Move the 2F-85 model to the gripper driver
        # (drivers/gripper) and compose the arm and gripper together at the embodiment level.
        gripper = attach_robotiq_2f85(root, meshes)
        add_default_frame(root, EE_LINK)
        return {
            roboarm_keys.URDF: ET.tostring(root, encoding='unicode'),
            roboarm_keys.JOINT_NAMES: _revolute_joint_names(urdf_xml),
            'meshes': meshes,
            roboarm_keys.CONTROL_FRAME: DEFAULT_FRAME,
            roboarm_keys.GRIPPER: gripper,
        }

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

    @contextlib.contextmanager
    def _desk_session(self):
        """Bracket the run with a Franka Desk session that opens brakes and activates FCI, and always releases
        control on exit. Hands both over to the operator when the driver does not manage Desk. When configured,
        recover a control box stuck in ``SafetyError`` by rebooting it once and retrying."""
        if self._desk_credentials is None:
            logger.info('Desk is not managed by the driver; brakes must be open and FCI active before the run')
            yield
            return
        rebooted = False
        while True:
            with Desk(self._ip, *self._desk_credentials) as desk:
                try:
                    desk.prepare()
                except SafetyControllerError:
                    if rebooted or not self._reboot_on_safety_error:
                        raise
                    logger.warning('Control box in SafetyError; rebooting it (unreachable ~40s) and retrying once')
                    desk.reboot(wait=True)
                    rebooted = True
                else:
                    if rebooted:
                        logger.info('Control box recovered after reboot')
                    yield desk
                    return

    @functools.cached_property
    def _robot(self) -> pf.Robot:
        """The libfranka handle, connected on first use."""
        return pf.Robot(
            self._ip, realtime_config=pf.RealtimeConfig.Ignore, relative_dynamics_factor=self._relative_dynamics_factor
        )

    def _arm(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock, safe_inputs: _SafeInputs) -> _Arm:
        """The arm this run drives, built from the driver's configuration."""
        return _Arm(
            self._robot,
            self.sync_move,
            self.commands,
            self.state,
            self._relative_dynamics_factor,
            should_stop,
            clock,
            safe_inputs,
        )

    def _safe_inputs(self) -> _SafeInputs:
        """The watch on the control box's safe inputs. It reads nothing where the driver does not manage Desk."""
        return _SafeInputs(self._ip, self._desk_credentials)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        safe_inputs = self._safe_inputs()
        with self._desk_session(), safe_inputs, self._arm(should_stop, clock, safe_inputs) as arm:
            robot = arm.robot
            self._init_robot(robot)
            self.robot_meta.emit(Robot._build_robot_meta(robot))

            yield from arm.park()

            in_error = False

            while not should_stop.value:
                st = robot.state()
                arm.publish(st)
                arm.note_refusals()

                in_error, entered_error = _check_error(st.error != 0, in_error)
                if entered_error:
                    logger.warning(f'Robot error: {st.error_message}')

                if in_error:
                    robot.recover_from_errors()
                    yield arm.limiter.wait()
                    continue

                asked = arm.moves.next_request()
                if isinstance(asked, pimm.calls.Call):
                    yield from arm.sync_move(asked)
                elif asked is not None:
                    with log_failure(asked):
                        arm.command_target(arm.to_joints(asked), asked.mode)

                yield arm.limiter.wait()

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
