import logging
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

try:
    import positronic_franka._franka as pf
except ImportError as e:
    raise ImportError(
        'Franka support is not installed. Install the hardware extra:\n'
        '  pip install "positronic[hardware]"\n'
        'or install the Franka core directly:\n'
        '  pip install positronic-franka\n'
    ) from e

import pimm
from positronic import geom

from . import RobotStatus, State, command


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

    def _start_reset(self):
        self.array[FrankaState.STATUS_OFFSET] = RobotStatus.RESETTING.value

    def _finish_reset(self):
        self.array[FrankaState.STATUS_OFFSET] = RobotStatus.AVAILABLE.value

    def encode(self, state: pf.State):
        self.array[FrankaState.Q_OFFSET : FrankaState.Q_OFFSET + 7] = state.q
        self.array[FrankaState.DQ_OFFSET : FrankaState.DQ_OFFSET + 7] = state.dq
        self.array[FrankaState.EE_POSE_OFFSET : FrankaState.EE_POSE_OFFSET + 7] = state.end_effector_pose
        self.array[FrankaState.EE_WRENCH_OFFSET : FrankaState.EE_WRENCH_OFFSET + 6] = state.ee_wrench
        self.array[FrankaState.STATUS_OFFSET] = (
            RobotStatus.AVAILABLE.value if state.error == 0 else RobotStatus.ERROR.value
        )


class Robot(pimm.ControlSystem):
    def __init__(
        self,
        ip: str,
        *,
        relative_dynamics_factor=0.2,
        home_joints: list[float] | None = None,
        home_joints_variation: list[float] | None = None,
        load: tuple | None = None,
    ) -> None:
        """
        :param ip: IP address of the robot.
        :param relative_dynamics_factor: Relative dynamics factor in [0, 1]. Smaller values are more conservative.
        :param home_joints: Joints of "reset" position.
        :param home_joints_variation: Max random deviation per joint in radians. Set to [0]*7 to disable.
        """
        self._ip = ip
        self._relative_dynamics_factor = relative_dynamics_factor
        self._home_joints = home_joints if home_joints is not None else [0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0]
        self._home_joints_variation = (
            home_joints_variation if home_joints_variation is not None else [0.03, 0.05, 0.08, 0.08, 0.10, 0.10, 0.10]
        )
        self.commands: pimm.SignalReceiver = pimm.ControlSystemReceiver(self, default=None)
        self.state: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)
        self._load = load

    @staticmethod
    def _init_robot(robot, load: tuple | None = None):
        coeff = 2.0
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
        robot.set_joint_impedance([3000, 3000, 3000, 2500, 2500, 2000, 2000])
        robot.set_cartesian_impedance([3000, 3000, 3000, 300, 300, 300])
        if load is not None:
            logging.info(f'Setting load to {load}')
            robot.set_load(*load)
        else:
            robot.set_load(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    def _reset(self, robot, robot_state: FrankaState):
        robot_state._start_reset()
        self.state.emit(robot_state)

        target = np.asarray(self._home_joints, dtype=np.float64)
        if any(v > 0 for v in self._home_joints_variation):
            variation = np.random.uniform(
                -np.asarray(self._home_joints_variation), np.asarray(self._home_joints_variation)
            )
            target = target + variation
        robot.set_target_joints(target, asynchronous=False)

        robot_state._finish_reset()
        self.state.emit(robot_state)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        robot = pf.Robot(
            self._ip, realtime_config=pf.RealtimeConfig.Ignore, relative_dynamics_factor=self._relative_dynamics_factor
        )
        Robot._init_robot(robot, self._load)
        robot.recover_from_errors()

        robot_state = FrankaState()
        rate_limiter = pimm.RateLimiter(clock, hz=2000)

        self._reset(robot, robot_state)

        in_error = False
        while not should_stop.value:
            st = robot.state()
            robot_state.encode(st)
            self.state.emit(robot_state)

            cmd_msg = self.commands.read()
            if st.error != 0:
                if not in_error:
                    logging.warning(f'Robot error: {st.error_message}')
                    in_error = True
                if cmd_msg.updated:
                    match cmd_msg.data:
                        case command.Recover():
                            robot.recover_from_errors()
                        case command.Reset():
                            robot.recover_from_errors()
                            self._reset(robot, robot_state)
                            in_error = False
                            continue
                yield pimm.Sleep(rate_limiter.wait_time())
                continue
            in_error = False

            if cmd_msg.updated:
                match cmd_msg.data:
                    case command.Reset():
                        self._reset(robot, robot_state)
                        continue
                    case command.CartesianPosition(pose):
                        target_pose_wxyz = np.asarray([*pose.translation, *pose.rotation.as_quat])
                        ik_solution = robot.inverse_kinematics_with_limits(target_pose_wxyz)
                        robot.set_target_joints(ik_solution)
                    case command.JointPosition(positions):
                        robot.set_target_joints(positions)
                    case command.JointDelta(velocities=joint_delta):
                        robot.set_target_joints(st.q + joint_delta)
                    case command.Recover():
                        pass
                    case _:
                        raise NotImplementedError(f'Unsupported command {cmd_msg.data}')

            yield pimm.Sleep(rate_limiter.wait_time())


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

        while not world.should_stop and (state.read() is None or state.value.status == RobotStatus.RESETTING):
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
                commands.emit(command.CartesianPosition(geom.Transform3D(pos + origin.translation, origin.rotation)))
                i += 1
            else:
                time.sleep(0.01)

        print('Finishing')
