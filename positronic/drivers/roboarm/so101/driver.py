import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers.common import MoveStatus, PendingMove
from positronic.drivers.motors.feetech import MotorBus
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.kinematics import Kinematics
from positronic.drivers.roboarm.models import DEFAULT_FRAME, add_default_frame


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
# How close to its setpoint the arm counts as arrived, in the bus's normalized units: the bus reports
# position but no goal, so arrival is judged from the reading
_ARRIVED_TOL = 0.02


class Robot(pimm.ControlSystem):
    def __init__(self, motor_bus: MotorBus, home_joints: list[float] | None = None):
        self.motor_bus = motor_bus
        self.mujoco_model_path = 'positronic/drivers/roboarm/so101/so101.xml'
        self.kinematic = Kinematics(_SO101_URDF_PATH, _SO101_EE_JOINT)
        self.joint_limits = self.kinematic.joint_limits
        self.home_joints = home_joints if home_joints is not None else [0.0, 0.0, 0.0, 0.0, 0.0]
        self.commands = pimm.ControlSystemReceiver[roboarm_command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[roboarm_command.CommandType, None](self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self._last_grip: float = 0.0

        self.grip: pimm.SignalEmitter[float] = pimm.ControlSystemEmitter(self)
        self.state: pimm.SignalEmitter[SO101State] = pimm.ControlSystemEmitter(self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

        print('================================================================')
        print('Warning: Proper dq units is not implemented for SO101!')
        print('================================================================')

    def _arm_rad_to_norm(self, q_rad: np.ndarray) -> np.ndarray:
        """Normalize the five arm joints. ``rad_to_norm`` spans the bus, whose last entry is the gripper."""
        return self.rad_to_norm(np.append(q_rad, 0.0))[:-1]

    def _target_qpos(self, state: SO101State, cmd: roboarm_command.CommandType) -> np.ndarray:
        """The setpoint ``cmd`` asks the arm to hold, in the bus's normalized units."""
        match cmd:
            case roboarm_command.Reset():
                return self._arm_rad_to_norm(np.asarray(self.home_joints, dtype=np.float32))
            case roboarm_command.CartesianPosition(pose):
                return self._solve_ik(state, pose)
            case roboarm_command.CartesianDelta() as delta_cmd:
                ee_pose, _ = self._forward_kinematics(self.motor_bus.position)
                return self._solve_ik(state, delta_cmd.apply(ee_pose))
            case roboarm_command.JointPosition(qpos):
                return self._arm_rad_to_norm(np.asarray(qpos, dtype=np.float32))
            case other:
                raise ValueError(f'Unknown command: {other}')

    def _streamed_setpoint(self, state: SO101State) -> np.ndarray | None:
        """The setpoint the command stream asks for, if one arrived and the arm can be put there."""
        if (cmd := pimm.value_updated(self.commands)) is None:
            return None
        try:
            return self._target_qpos(state, cmd)
        # rules-allow: swallowed-error — a command stream cannot end the run; the next supersedes
        except Exception as exc:
            logging.warning(f'{cmd} not applied: {exc}')
            return None

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self.motor_bus.connect()
        urdf = ET.fromstring(Path(_SO101_URDF_PATH).read_text())
        add_default_frame(urdf, _SO101_EE_LINK)
        self.robot_meta.emit({
            keys.URDF: ET.tostring(urdf, encoding='unicode'),
            keys.JOINT_NAMES: _SO101_JOINT_NAMES,
            keys.CONTROL_FRAME: DEFAULT_FRAME,
        })

        rate_limit = pimm.RateLimiter(hz=1000, clock=clock)
        state = SO101State()
        # The arm half of the motor setpoint, in normalized units. Read here, not in ``__init__``: the arm
        # holds where the bus finds it until something asks otherwise, and there is no bus until now.
        self._last_qpos = np.asarray(self.motor_bus.position[:-1])
        # The bus reports position only while this loop reads it, so it cannot be held for a move
        move = PendingMove(_ARRIVED_TOL)

        while not should_stop.value:
            # The arm and the gripper are one setpoint on a shared bus, but they arrive as two channels that
            # need not carry a value in the same round, so either one changing rewrites the whole vector.
            setpoint_changed = False
            if (grip := pimm.value_updated(self.target_grip)) is not None:
                self._last_grip, setpoint_changed = grip, True

            if move.active and move.settle(self.motor_bus.position[:-1], clock.now()) is MoveStatus.GAVE_UP:
                # Holding the target the arm stopped short of would resume the move once whatever blocked
                # it goes away, long after its asker was told it failed.
                self._last_qpos, setpoint_changed = np.asarray(self.motor_bus.position[:-1]), True

            if not move.active:
                if (call := next(self.sync_move.incoming(), None)) is not None:
                    with pimm.calls.forward_failure(call):
                        target = self._target_qpos(state, call.request)
                        self._last_qpos, setpoint_changed = target, True
                        move.accept(call, target, clock.now())
                elif (target := self._streamed_setpoint(state)) is not None:
                    self._last_qpos, setpoint_changed = target, True

            if setpoint_changed:
                self.motor_bus.set_target_position(np.concatenate([self._last_qpos, [self._last_grip]]))

            q = self.motor_bus.position
            dq = self.motor_bus.velocity[:-1]
            ee_pose, gripper = self._forward_kinematics(q)
            position_rad = self.norm_to_rad(q)[:-1]
            if move.active:  # the driver owns the arm until the move answers, and reads no command meanwhile
                status = RobotStatus.RESETTING
            else:  # the bus reports position, not whether the arm is where the driver put it
                status = RobotStatus.ERROR if move.errored else RobotStatus.AVAILABLE
            state.encode(position_rad, dq, ee_pose, status)

            self.state.emit(state)
            self.grip.emit(gripper)
            yield rate_limit.wait()

    def _solve_ik(self, state, pose: geom.Transform3D) -> np.ndarray:
        q = np.array(state.q).tolist()
        q.append(0.0)  # ignore gripper in ik
        q_rad_new = self.kinematic.inverse(q, pose, n_iter=10)
        return self.rad_to_norm(q_rad_new)[:-1]

    def _forward_kinematics(self, motor_position) -> tuple[geom.Transform3D, float]:
        q_rad = self.norm_to_rad(motor_position)
        ee_pose = self.kinematic.forward(q_rad)
        gripper = motor_position[-1]
        return ee_pose, gripper

    def norm_to_rad(self, qpos: np.ndarray) -> np.ndarray:
        """Convert normalized position [0, 1] to radians.

        Args:
            qpos: Normalized position.

        Returns:
            Radians.
        """
        return qpos * (self.joint_limits[:, 1] - self.joint_limits[:, 0]) + self.joint_limits[:, 0]

    def rad_to_norm(self, qpos: np.ndarray) -> np.ndarray:
        """Convert radians to normalized position [0, 1]."""
        return (qpos - self.joint_limits[:, 0]) / (self.joint_limits[:, 1] - self.joint_limits[:, 0])
