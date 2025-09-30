import time
from collections.abc import Iterator
from typing import Any

import numpy as np

try:
    from . import _franka as pf
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
    def __init__(self):
        super().__init__(shape=(7 + 7 + 7 + 1,), dtype=np.float32)

    def instantiation_params(self) -> tuple[Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[:7]

    @property
    def dq(self) -> np.ndarray:
        return self.array[7:14]

    @property
    def ee_pose(self) -> geom.Transform3D:
        return geom.Transform3D(self.array[14 : 14 + 3], self.array[14 + 3 : 14 + 7])

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[14 + 7]))

    def _start_reset(self):
        self.array[14 + 7] = RobotStatus.RESETTING.value

    def _finish_reset(self):
        self.array[14 + 7] = RobotStatus.AVAILABLE.value

    def encode(self, q, dq, ee_pose: geom.Transform3D):
        self.array[:7] = q
        self.array[7:14] = dq
        self.array[14:17] = ee_pose.translation
        self.array[17:21] = ee_pose.rotation.as_quat
        self.array[21] = RobotStatus.AVAILABLE.value


class Robot(pimm.ControlSystem):
    def __init__(self, ip: str, relative_dynamics_factor=0.2, home_joints: list[float] | None = None) -> None:
        """
        :param ip: IP address of the robot.
        :param relative_dynamics_factor: Relative dynamics factor in [0, 1]. Smaller values are more conservative.
        :param home_joints: Joints of "reset" position.
        """
        self._ip = ip
        self._relative_dynamics_factor = relative_dynamics_factor
        self._home_joints = home_joints if home_joints is not None else [0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0]
        self.commands: pimm.SignalReceiver = pimm.ControlSystemReceiver(self)
        self.state: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)

    @staticmethod
    def _init_robot(robot):
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
        robot.set_load(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    def _reset(self, robot, robot_state: FrankaState):
        robot_state._start_reset()
        self.state.emit(robot_state)

        robot.set_target_joints(np.asarray(self._home_joints, dtype=np.float64), asynchronous=False)

        robot_state._finish_reset()
        self.state.emit(robot_state)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        robot = pf.Robot(
            self._ip, realtime_config=pf.RealtimeConfig.Ignore, relative_dynamics_factor=self._relative_dynamics_factor
        )
        Robot._init_robot(robot)
        robot.recover_from_errors()

        commands = pimm.DefaultReceiver(pimm.ValueUpdated(self.commands), (None, False))
        robot_state = FrankaState()
        rate_limiter = pimm.RateLimiter(clock, hz=2000)

        self._reset(robot, robot_state)

        while not should_stop.value:
            cmd, updated = commands.value
            if updated:
                match cmd:
                    case command.Reset():
                        self._reset(robot, robot_state)
                        continue
                    case command.CartesianMove(pose):
                        target_pose_wxyz = np.asarray([*pose.translation, *pose.rotation.as_quat])
                        ik_solution = robot.inverse_kinematics(target_pose_wxyz)
                        # TODO: implement MOVING state support
                        robot.set_target_joints(ik_solution)
                    case _:
                        raise NotImplementedError(f'Unsupported command {cmd}')

            st = robot.state()
            q = np.asarray(st.q, dtype=np.float32).reshape(-1)
            dq = np.asarray(st.dq, dtype=np.float32).reshape(-1)
            ee = geom.Transform3D(
                translation=np.asarray(st.end_effector_pose[:3], dtype=np.float32),
                rotation=geom.Rotation.from_quat([
                    float(st.end_effector_pose[3]),
                    float(st.end_effector_pose[4]),
                    float(st.end_effector_pose[5]),
                    float(st.end_effector_pose[6]),
                ]),
            )
            robot_state.encode(q, dq, ee)
            self.state.emit(robot_state)

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
                commands.emit(command.CartesianMove(geom.Transform3D(pos + origin.translation, origin.rotation)))
                i += 1
            else:
                time.sleep(0.01)

        print('Finishing')
