import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers.motors.feetech import MotorBus
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.kinematics import Kinematics
from positronic.drivers.roboarm.models import DEFAULT_FRAME, add_default_frame
from positronic.drivers.utils import DriverRun, MoveStatus, PendingMove


class SO101State(State, pimm.shared_memory.NumpySMAdapter):
    def __init__(self):
        super().__init__(shape=(5 + 5 + 7 + 1,), dtype=np.float32)

    def instantiation_params(self) -> tuple[geom.Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[:5]

    @property
    def dq(self) -> np.ndarray:
        return self.array[5:10]

    @property
    def ee_pose(self) -> geom.Transform3D:
        translation = self.array[10 : 10 + 3]
        quaternion = geom.Rotation.from_quat(self.array[10 + 3 : 10 + 7])
        return geom.Transform3D(translation, quaternion)

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[17]))

    def encode(self, q, dq, ee_pose, status: RobotStatus):
        self.array[:5] = q
        self.array[5:10] = dq
        self.array[10 : 10 + 3] = ee_pose.translation
        self.array[10 + 3 : 10 + 7] = ee_pose.rotation.as_quat
        self.array[17] = status.value


_SO101_URDF_PATH = 'positronic/drivers/roboarm/so101/so101.urdf'
_SO101_JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
_SO101_EE_LINK = 'gripper_frame_link'
_SO101_EE_JOINT = 'gripper_frame_joint'


class _Arm(DriverRun):
    """The arm the driver drives: the bus it reads and writes, the setpoint it holds the arm at, and the state
    published from what the bus reports.

    The bus reports position only while the loop reads it, so it cannot be held for a move.
    """

    # How close to its setpoint the arm counts as arrived, in the bus's normalized units: the bus reports
    # position but no goal, so arrival is judged from the reading
    _ARRIVED_TOL = 0.02
    # Seconds a synchronous move gets: the bus drives the whole range in well under this
    _MOVE_TIMEOUT_S = 5.0

    def __init__(
        self,
        bus: MotorBus,
        out: pimm.SignalEmitter[SO101State],
        grip_out: pimm.SignalEmitter[float],
        home_joints: list[float],
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
    ):
        super().__init__(should_stop, clock, hz=1000)
        self.bus = bus
        self.out = out
        self.grip_out = grip_out
        self.state = SO101State()
        self.kinematic = Kinematics(_SO101_URDF_PATH, _SO101_EE_JOINT)
        self._joint_limits = self.kinematic.joint_limits
        self._home_joints = home_joints
        self._move = PendingMove(self._ARRIVED_TOL)
        # Read here rather than left empty: every setpoint below is solved from where the arm is now, and the
        # arm holds where the bus finds it until something asks otherwise
        self.q_norm = bus.position
        self._qpos, self._grip = np.asarray(self.q_norm[:-1]), float(self.q_norm[-1])
        # Nothing has asked the arm to be anywhere, and the bus holds whatever it was left holding
        self._unsent = False

    def _norm_to_rad(self, qpos: np.ndarray) -> np.ndarray:
        """The bus's normalized 0..1 range mapped onto the joint's own travel, in radians."""
        return qpos * (self._joint_limits[:, 1] - self._joint_limits[:, 0]) + self._joint_limits[:, 0]

    def _rad_to_norm(self, qpos: np.ndarray) -> np.ndarray:
        """Radians mapped back onto the bus's normalized 0..1 range."""
        return (qpos - self._joint_limits[:, 0]) / (self._joint_limits[:, 1] - self._joint_limits[:, 0])

    def _forward_kinematics(self, q_norm: np.ndarray) -> tuple[geom.Transform3D, float]:
        return self.kinematic.forward(self._norm_to_rad(q_norm)), q_norm[-1]

    def _solve_ik(self, pose: geom.Transform3D) -> np.ndarray:
        q = self._norm_to_rad(self.q_norm).tolist()
        q[-1] = 0.0  # ignore gripper in ik
        return self._rad_to_norm(self.kinematic.inverse(q, pose, n_iter=10))[:-1]

    def _arm_rad_to_norm(self, q_rad: np.ndarray) -> np.ndarray:
        """Normalize the five arm joints. ``_rad_to_norm`` spans the bus, whose last entry is the gripper."""
        return self._rad_to_norm(np.append(q_rad, 0.0))[:-1]

    def _requested_qpos(self, cmd: roboarm_command.CommandType) -> np.ndarray:
        """The setpoint ``cmd`` asks for, in the bus's normalized units, whether or not the arm can hold it."""
        match cmd:
            case roboarm_command.Reset():
                return self._arm_rad_to_norm(np.asarray(self._home_joints, dtype=np.float32))
            case roboarm_command.CartesianPosition(pose):
                return self._solve_ik(pose)
            case roboarm_command.CartesianDelta() as delta_cmd:
                ee_pose, _ = self._forward_kinematics(self.q_norm)
                return self._solve_ik(delta_cmd.apply(ee_pose))
            case roboarm_command.JointPosition(qpos):
                return self._arm_rad_to_norm(np.asarray(qpos, dtype=np.float32))
            case other:
                raise ValueError(f'Unknown command: {other}')

    def _target_qpos(self, cmd: roboarm_command.CommandType) -> np.ndarray:
        """The setpoint ``cmd`` asks the arm to hold, clipped to the calibrated range the bus reports from.

        The bus clips the command anyway, so a target outside it is one the arm can reach but never read back.
        """
        return np.clip(self._requested_qpos(cmd), 0.0, 1.0)

    @property
    def takes_commands(self) -> bool:
        """Whether the arm will take a setpoint: a move owns it until it settles, and a settled one until it
        is answered."""
        return not (self._move.active or self._move.settled)

    def read(self) -> None:
        """Take the arm off the bus: one serial round-trip a tick, and every step below wants the same instant."""
        self.q_norm = self.bus.position

    def settle(self) -> None:
        """Judge a move in flight against what the bus reports, and stop holding a target the arm missed."""
        if not self._move.active:
            return
        if self._move.settle(self.q_norm[:-1], self.clock.now()) is MoveStatus.GAVE_UP:
            # Holding the target the arm stopped short of would resume the move once whatever blocked
            # it goes away, long after its asker was told it failed.
            self._qpos, self._unsent = np.asarray(self.q_norm[:-1]), True

    def hold_grip(self, grip: float) -> None:
        """Hold the fingers at ``grip``."""
        self._grip, self._unsent = grip, True

    def track(self, cmd: roboarm_command.CommandType) -> None:
        """Hold the arm at the setpoint ``cmd`` asks for, with nobody waiting on it getting there."""
        self._qpos, self._unsent = self._target_qpos(cmd), True

    def serve_sync_move(self, call: pimm.calls.Call[roboarm_command.CommandType, None]) -> None:
        """Hold the arm where ``call`` asks; ``settle`` answers it once the bus reads back there."""
        with pimm.calls.raise_to(call):
            target = self._target_qpos(call.request)
            self._qpos, self._unsent = target, True
            self._move.accept(call, target, self.clock.now(), self._MOVE_TIMEOUT_S)

    def write(self) -> None:
        """Put the setpoint on the bus, if anything has asked for one since it was last written.

        The arm and the gripper are one setpoint on a shared bus, but they arrive as two channels that need
        not carry a value in the same round, so either one changing rewrites the whole vector.
        """
        if not self._unsent:
            return
        self.bus.set_target_position(np.concatenate([self._qpos, [self._grip]]))
        self._unsent = False

    def publish(self) -> None:
        """Ship the arm as the bus reports it, arm and fingers."""
        ee_pose, gripper = self._forward_kinematics(self.q_norm)
        if self._move.active:  # the driver owns the arm until the move settles, and reads no command meanwhile
            status = RobotStatus.BUSY
        else:  # the bus reports position, not whether the arm is where the driver put it
            status = RobotStatus.ERROR if self._move.errored else RobotStatus.AVAILABLE
        self.state.encode(self._norm_to_rad(self.q_norm)[:-1], self.bus.velocity[:-1], ee_pose, status)
        self.out.emit(self.state)
        self.grip_out.emit(gripper)

    def answer(self) -> None:
        """Hand a move settled this tick its outcome, now that the state saying so is out."""
        self._move.answer()

    def fail(self, exc: BaseException) -> None:
        """Hand `exc` to a move the run died under, whose asker is blocked on an answer."""
        self._move.fail(exc)


class Robot(pimm.ControlSystem):
    def __init__(self, motor_bus: MotorBus, home_joints: list[float] | None = None):
        self.motor_bus = motor_bus
        self.mujoco_model_path = 'positronic/drivers/roboarm/so101/so101.xml'
        self.home_joints = home_joints if home_joints is not None else [0.0, 0.0, 0.0, 0.0, 0.0]
        self.commands = pimm.ControlSystemReceiver[roboarm_command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[roboarm_command.CommandType, None](self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)

        self.grip = pimm.ControlSystemEmitter[float](self)
        self.state = pimm.ControlSystemEmitter[SO101State](self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

        print('================================================================')
        print('Warning: Proper dq units is not implemented for SO101!')
        print('================================================================')

    def _arm(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> _Arm:
        """The arm this run drives, built from the driver's configuration.

        Built here, not in ``__init__``: a background control system is pickled before it runs, and an open
        serial port does not survive the trip.
        """
        self.motor_bus.connect()
        return _Arm(self.motor_bus, self.state, self.grip, self.home_joints, should_stop, clock)

    @staticmethod
    def _build_robot_meta() -> dict:
        urdf = ET.fromstring(Path(_SO101_URDF_PATH).read_text())
        add_default_frame(urdf, _SO101_EE_LINK)
        return {
            keys.URDF: ET.tostring(urdf, encoding='unicode'),
            keys.JOINT_NAMES: _SO101_JOINT_NAMES,
            keys.CONTROL_FRAME: DEFAULT_FRAME,
        }

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        arm = self._arm(should_stop, clock)
        self.robot_meta.emit(Robot._build_robot_meta())

        try:
            while not should_stop.value:
                arm.read()
                if (grip := pimm.value_updated(self.target_grip)) is not None:
                    arm.hold_grip(grip)
                arm.settle()
                if arm.takes_commands:
                    if (call := next(self.sync_move.incoming(), None)) is not None:
                        arm.serve_sync_move(call)
                    elif (cmd := pimm.value_updated(self.commands)) is not None:
                        try:
                            arm.track(cmd)
                        # rules-allow: swallowed-error — a command stream cannot end the run; the next supersedes
                        except Exception as exc:
                            logging.warning(f'{cmd} not applied: {exc}')

                arm.write()
                arm.publish()
                arm.answer()  # the state a settled move is answered with is out

                yield arm.limiter.wait()
        except Exception as exc:
            arm.fail(exc)  # a run that dies mid-move must not leave its asker waiting
            raise
