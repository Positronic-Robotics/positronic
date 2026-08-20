import contextlib
import logging
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Any

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers import vendor_import
from positronic.drivers.utils import DriverRun, MoveStatus

from . import RobotStatus, State, command
from .models import DEFAULT_FRAME, EE_LINK, add_default_frame, attach_robotiq_2f85

with vendor_import('positronic_franka', 'Franka support', platforms=('linux',)):
    import positronic_franka._franka as pf
    from positronic_franka.desk import Desk, SafetyControllerError


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

    def _set_busy(self):
        self.array[FrankaState.STATUS_OFFSET] = RobotStatus.BUSY.value

    def _set_error(self):
        self.array[FrankaState.STATUS_OFFSET] = RobotStatus.ERROR.value

    def encode(self, state: pf.State):
        self.array[FrankaState.Q_OFFSET : FrankaState.Q_OFFSET + 7] = state.q
        self.array[FrankaState.DQ_OFFSET : FrankaState.DQ_OFFSET + 7] = state.dq
        self.array[FrankaState.EE_POSE_OFFSET : FrankaState.EE_POSE_OFFSET + 7] = state.end_effector_pose
        self.array[FrankaState.EE_WRENCH_OFFSET : FrankaState.EE_WRENCH_OFFSET + 6] = state.ee_wrench
        self.array[FrankaState.STATUS_OFFSET] = (
            RobotStatus.AVAILABLE.value if state.error == 0 else RobotStatus.ERROR.value
        )


def _revolute_joint_names(urdf_xml):
    root = ET.fromstring(urdf_xml)
    return [j.get('name') for j in root.findall('joint') if j.get('type') == 'revolute']


_MESH_DIR = Path(__file__).resolve().parent.parent.parent / 'assets/fr3_collision'


class _Arm(DriverRun):
    """The arm the driver drives: the libfranka handle, the state published from it, and the moves made with it."""

    # A move needs a deadline at all because the vendor reports a goal it abandons, but a goal it never
    # converges on stays in flight for as long as the arm is pushed off course.
    # FR3 joint velocity limits in rad/s, from the bundled ``fr3.urdf``; ``relative_dynamics_factor`` scales them
    _MAX_JOINT_VELOCITY = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
    # On top of the travel itself: the vendor controller ramps in and out of its speed cap, and settles late
    _MOVE_GRACE_S = 5.0
    # Parking publishes nothing, so it comes back only as often as it needs to ask again
    _PARK_POLL_S = 0.005

    def __init__(
        self,
        vendor: pf.Robot,
        out: pimm.SignalEmitter[FrankaState],
        home_joints: list[float],
        home_joints_variation: list[float],
        park_timeout_s: float,
        dynamics_factor: float,
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
    ):
        super().__init__(should_stop, clock, hz=2000)
        self.vendor = vendor
        self.out = out
        self.state = FrankaState()
        self._home_joints = home_joints
        self._home_joints_variation = home_joints_variation
        self._park_timeout_s = park_timeout_s
        self._dynamics_factor = dynamics_factor

    def publish(self, st: pf.State) -> None:
        """Ship the arm as the vendor reports it, marked ERROR while it is not where the driver put it."""
        self.state.encode(st)
        if self.errored:  # ``encode`` reads the vendor's fault flag, not where the arm was put
            self.state._set_error()
        self.out.emit(self.state)

    def home_target(self) -> np.ndarray:
        """The home joints this trip aims for, drawn afresh from the spread each time."""
        target = np.asarray(self._home_joints, dtype=np.float64)
        if any(v > 0 for v in self._home_joints_variation):
            variation = np.random.uniform(
                -np.asarray(self._home_joints_variation), np.asarray(self._home_joints_variation)
            )
            target = target + variation
        return target

    def await_goal(
        self, target: np.ndarray, should_stop: Callable[[], bool], pace: Callable[[], pimm.Command]
    ) -> Generator[pimm.Command, None, MoveStatus]:
        """Command ``target`` and poll the goal until the arm arrives, one poll per resume.

        ``pace`` is what to wait between polls, and so how often the goal is asked about. It is the caller's
        to choose: one with work of its own to do each pass comes back at the rate that work needs, one with
        none comes back only as often as it needs to ask.
        """
        self.vendor.set_target_joints(target)
        while not should_stop():
            goal = self.vendor.goal()
            if goal.status == pf.GoalStatus.REACHED:
                return MoveStatus.ARRIVED
            if goal.status != pf.GoalStatus.IN_FLIGHT:
                raise RuntimeError(f'the arm stopped short of its target: {goal.reason or goal.status}')
            yield pace()
        return MoveStatus.GAVE_UP

    def _travel_s(self, q: np.ndarray, target: np.ndarray) -> float:
        """How long the arm may take to reach ``target``, from the speed its dynamics factor allows.

        A conservative factor buys proportionally more time rather than failing a move the arm is tracking.
        """
        cap = self._MAX_JOINT_VELOCITY * self._dynamics_factor
        return self._MOVE_GRACE_S + float(np.max(np.abs(target - q) / cap))

    def move_to(self, target: np.ndarray) -> Generator[pimm.Command, None, MoveStatus]:
        """Travel to ``target``, yielding until it arrives."""
        # The first emit must not ship an unfilled state.
        self.state.encode(self.vendor.state())
        self.state._set_busy()
        self.out.emit(self.state)

        deadline = self.clock.now() + self._travel_s(self.state.q, target)
        try:
            # The arm's own rate: this loop publishes every pass, so the goal is asked about that often too
            for wait in self.await_goal(
                target, lambda: self.should_stop.value or self.clock.now() >= deadline, self.limiter.wait
            ):
                st = self.vendor.state()
                self.state.encode(st)
                self.state._set_busy()  # `encode` clears the status; the arm has not arrived
                self.out.emit(self.state)
                if st.error != 0:
                    self.vendor.recover_from_errors()
                yield wait
            # The deadline stops the poll loop before it asks, so a goal that landed as it expired has not
            # been seen yet; reading a timeout off the clock alone would fail a move the arm completed.
            if self.clock.now() >= deadline and self.vendor.goal().status != pf.GoalStatus.REACHED:
                # The vendor controller is still tracking the goal it did not reach; left alone it would
                # resume the move once whatever held the arm back goes away.
                self.vendor.set_target_joints(self.vendor.state().q)
                raise TimeoutError(f'the arm stopped short of {target}')
        except Exception:
            self.errored = True
            raise

        if self.should_stop.value:  # the travel gave up on the stop rather than on arrival
            return MoveStatus.GAVE_UP
        self.errored = False
        # The poll that reports arrival ends the loop, so the sample before it was taken mid-travel
        self.publish(self.vendor.state())
        return MoveStatus.ARRIVED

    def target_joints(self, cmd: command.CommandType) -> np.ndarray:
        """The joints ``cmd`` asks the arm to hold."""
        match cmd:
            case command.Reset():
                return self.home_target()
            case command.CartesianPosition(pose):
                return self.vendor.inverse_kinematics_with_limits(
                    np.asarray([*pose.translation, *pose.rotation.as_quat])
                )
            case command.CartesianDelta() as delta_cmd:
                target = delta_cmd.apply(self.state.ee_pose)
                return self.vendor.inverse_kinematics_with_limits(
                    np.asarray([*target.translation, *target.rotation.as_quat])
                )
            case command.JointPosition(positions):
                return np.asarray(positions, dtype=np.float64)
            case command.JointDelta(velocities=joint_delta):
                return self.state.q + joint_delta
            case other:
                raise NotImplementedError(f'Unsupported command {other}')

    def serve_sync_move(self, call: pimm.calls.Call[command.CommandType, None]) -> Iterator[pimm.Command]:
        """Put the arm where ``call`` asks and answer it once the state saying so is out.

        A stop cuts the move short, unanswered.
        """
        try:
            if (yield from self.move_to(self.target_joints(call.request))) is MoveStatus.ARRIVED:
                call.set_result(None)
        except Exception as exc:
            try:
                self.publish(self.vendor.state())
            finally:
                call.set_exception(exc)  # an arm the driver cannot read still leaves nobody waiting

    def park(self) -> Iterator[pimm.Command]:
        """Move the arm to the home pose, giving up after ``park_timeout_s``. Drive with ``yield from``.

        Runs after the control loop rather than in its ``finally``, so that only a stop request earns it.
        Answering a setup or control fault with a recovery and a fresh joint target would be autonomous
        motion in response to something going wrong.

        The motion itself is bounded: at the configured dynamics factor, to the configured ``home_joints``
        and nowhere else. An arm already there arrives immediately, so arrival is the controller's to
        report rather than something to pre-check. Every failure — including a goal that stops advancing —
        reaches the log and no further.
        """
        target = np.asarray(self._home_joints, dtype=np.float64)
        try:
            logging.info('Parking the arm at the home pose')
            self.vendor.recover_from_errors()  # once, before the move: a reflex during the move ends the park
            deadline = self.clock.now() + self._park_timeout_s
            outcome = yield from self.await_goal(
                target, lambda: self.clock.now() >= deadline, lambda: pimm.Sleep(self._PARK_POLL_S)
            )
            if outcome is MoveStatus.GAVE_UP:
                logging.error(f'Parking timed out after {self._park_timeout_s}s, the arm stays where it stands')
        # rules-allow: swallowed-error — parking is best-effort; brakes and control release must run regardless.
        except Exception:
            logging.exception('Parking failed, the arm stays where it stands')


class Robot(pimm.ControlSystem):
    def __init__(
        self,
        ip: str,
        *,
        relative_dynamics_factor=0.2,
        home_joints: list[float] | None = None,
        home_joints_variation: list[float] | None = None,
        load: tuple | None = None,
        collision_coeff: float = 2.0,
        manage_desk: bool = True,
        reboot_on_safety_error: bool = False,
        park_timeout_s: float = 10.0,
    ) -> None:
        """
        :param ip: IP address of the robot.
        :param relative_dynamics_factor: Relative dynamics factor in (0, 1]. Smaller values are more conservative.
        :param home_joints: Joints of "reset" position, and the pose the arm is parked at when the run ends.
        :param home_joints_variation: Max random deviation per joint in radians. Set to [0]*7 to disable.
        :param collision_coeff: Multiplier for collision thresholds. Higher = more tolerant.
        :param manage_desk: Run the Desk session from the driver: open the brakes and activate FCI on start, close
            them on stop. Requires ``FRANKA_DESK_USER`` and ``FRANKA_DESK_PASSWORD`` in the environment. Set to
            False to leave brakes and FCI to the operator; the driver then never contacts Desk and expects FCI to
            be up already.
        :param reboot_on_safety_error: When the control box is in an unrecoverable ``SafetyError`` on start, reboot
            it, wait for it to come back, and try once more before giving up. Only applies when ``manage_desk`` is
            set.
        :param park_timeout_s: How long the arm may travel back to ``home_joints`` when the run ends before the
            driver gives up and stops control where it stands. It is spent inside the world's teardown budget.
        """
        # A zero factor caps every joint at zero speed: the arm would never travel and no move could land
        assert 0 < relative_dynamics_factor <= 1, relative_dynamics_factor
        self._ip = ip
        self._relative_dynamics_factor = relative_dynamics_factor
        self._home_joints = home_joints if home_joints is not None else [0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0]
        self._home_joints_variation = (
            home_joints_variation if home_joints_variation is not None else [0.03, 0.05, 0.08, 0.08, 0.10, 0.10, 0.10]
        )
        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.state = pimm.ControlSystemEmitter[FrankaState](self)
        self.robot_meta = pimm.ControlSystemEmitter(self)
        self._load = load
        self._collision_coeff = collision_coeff
        self._robot: pf.Robot | None = None
        self._desk_credentials = _read_desk_credentials() if manage_desk else None
        self._reboot_on_safety_error = reboot_on_safety_error
        self._park_timeout_s = park_timeout_s

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
            keys.URDF: ET.tostring(root, encoding='unicode'),
            keys.JOINT_NAMES: _revolute_joint_names(urdf_xml),
            'meshes': meshes,
            keys.CONTROL_FRAME: DEFAULT_FRAME,
            'gripper': gripper,
        }

    @property
    def _vendor(self) -> pf.Robot:
        """The libfranka handle, connected on first use."""
        if self._robot is None:
            self._robot = pf.Robot(
                self._ip,
                realtime_config=pf.RealtimeConfig.Ignore,
                relative_dynamics_factor=self._relative_dynamics_factor,
            )
        return self._robot

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
        robot.set_control_mode(pf.InternalImpedance([3000, 3000, 3000, 2500, 2500, 2000, 2000]))
        if self._load is not None:
            logging.info(f'Setting load to {self._load}')
            robot.set_load(*self._load)
        else:
            robot.set_load(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    @contextlib.contextmanager
    def _desk_session(self):
        """Bracket the run with a Franka Desk session that opens brakes and activates FCI, and always releases
        control on exit. Hands both over to the operator when the driver does not manage Desk. When configured,
        recover a control box stuck in ``SafetyError`` by rebooting it once and retrying."""
        if self._desk_credentials is None:
            logging.info('Desk is not managed by the driver; brakes must be open and FCI active before the run')
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
                    logging.warning('Control box in SafetyError; rebooting it (unreachable ~40s) and retrying once')
                    desk.reboot(wait=True)
                    rebooted = True
                else:
                    if rebooted:
                        logging.info('Control box recovered after reboot')
                    yield desk
                    return

    def _arm(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> _Arm:
        """The arm this run drives, built from the driver's configuration.

        Built here, not in ``__init__``: a background control system is pickled before it runs, and a live
        libfranka handle does not survive the trip.
        """
        return _Arm(
            self._vendor,
            self.state,
            self._home_joints,
            self._home_joints_variation,
            self._park_timeout_s,
            self._relative_dynamics_factor,
            should_stop,
            clock,
        )

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        with self._desk_session():
            arm = self._arm(should_stop, clock)
            vendor = arm.vendor
            try:
                self._init_robot(vendor)
                self.robot_meta.emit(Robot._build_robot_meta(vendor))
                vendor.recover_from_errors()

                try:
                    yield from arm.move_to(arm.home_target())
                # rules-allow: swallowed-error — an arm that will not home reads ERROR; it does not end the run
                except Exception as exc:
                    logging.error(f'Homing failed, the arm is not where the driver put it: {exc}')

                in_error = False

                while not should_stop.value:
                    st = vendor.state()
                    arm.publish(st)

                    in_error, entered_error = _check_error(st.error != 0, in_error)
                    if entered_error:
                        logging.warning(f'Robot error: {st.error_message}')

                    if in_error:
                        vendor.recover_from_errors()
                        yield arm.limiter.wait()
                        continue

                    if (call := next(self.sync_move.incoming(), None)) is not None:
                        yield from arm.serve_sync_move(call)
                    elif (cmd := pimm.value_updated(self.commands)) is not None:
                        try:
                            if isinstance(cmd, command.Reset):
                                yield from arm.move_to(arm.home_target())
                            else:
                                vendor.set_target_joints(arm.target_joints(cmd))
                        # rules-allow: swallowed-error — a command stream cannot end the run; the next supersedes
                        except Exception as exc:
                            logging.warning(f'{cmd} not applied: {exc}')

                    yield arm.limiter.wait()

                yield from arm.park()
            finally:
                # Halt the driver's control thread before _desk_session deactivates FCI, or it dies mid-control
                # with "TCP connection got interrupted".
                vendor.stop()


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
