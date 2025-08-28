import time
from typing import Iterator

from mujoco import Any

import franky
import numpy as np

from positronic import geom
import pimm

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
        return geom.Transform3D(self.array[14:14 + 3], self.array[14 + 3:14 + 7])

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[14 + 7]))

    def _start_reset(self):
        self.array[14 + 7] = RobotStatus.RESETTING.value

    def _finish_reset(self):
        self.array[14 + 7] = RobotStatus.AVAILABLE.value

    def encode(self, q, dq, ee_pose):
        self.array[:7] = q
        self.array[7:14] = dq
        self.array[14:14 + 3] = ee_pose.translation
        q_wxyz = np.concatenate([ee_pose.quaternion[3:], ee_pose.quaternion[:3]])
        self.array[14 + 3:14 + 7] = q_wxyz
        self.array[14 + 7] = RobotStatus.AVAILABLE.value


class Robot:
    commands: pimm.SignalReader = pimm.NoOpReader()
    state: pimm.SignalEmitter = pimm.NoOpEmitter()

    def __init__(self,
                 ip: str,
                 relative_dynamics_factor=0.2,
                 home_joints: list[float] = [0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0]) -> None:
        """
        :param ip: IP address of the robot.
        :param relative_dynamics_factor: Relative dynamics factor in [0, 1]. Smaller values are more conservative.
        :param home_joints: Joints of "reset" position.
        """
        self._ip = ip
        self._relative_dynamics_factor = relative_dynamics_factor
        self._home_joints = home_joints

    @staticmethod
    def _init_robot(robot, rel_dynamics_factor: float):
        robot.set_joint_impedance([150, 150, 150, 125, 125, 250, 250])

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
        robot.relative_dynamics_factor = rel_dynamics_factor

    def _reset(self, robot: franky.Robot, robot_state: FrankaState):
        robot_state._start_reset()
        self.state.emit(robot_state)

        robot.join_motion(timeout=0.1)
        reset_motion = franky.JointWaypointMotion([franky.JointWaypoint(self._home_joints)])
        robot.move(reset_motion, asynchronous=False)

        robot_state._finish_reset()
        self.state.emit(robot_state)

    def run(self, should_stop: pimm.SignalReader, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        robot = franky.Robot(self._ip, realtime_config=franky.RealtimeConfig.Ignore)
        Robot._init_robot(robot, self._relative_dynamics_factor)
        robot.recover_from_errors()

        commands = pimm.DefaultReader(pimm.ValueUpdated(self.commands), (None, False))
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
                        pos = franky.Affine(translation=pose.translation, quaternion=pose.rotation.as_quat_xyzw)

                motion = franky.CartesianMotion(pos, franky.ReferenceType.Absolute)
                try:
                    # TODO: implement MOVING state support
                    robot.move(motion, asynchronous=True)
                except franky.ControlException as e:
                    robot.recover_from_errors()
                    print(f"Motion failed for {pos}: {e}")

            js = robot.current_joint_state
            robot_state.encode(js.position, js.velocity, robot.current_pose.end_effector_pose)
            self.state.emit(robot_state)

            yield pimm.Sleep(rate_limiter.wait_time())


if __name__ == "__main__":
    with pimm.World() as world:
        robot = Robot("172.168.0.2", relative_dynamics_factor=0.2)
        commands, robot.commands = world.mp_pipe()
        robot.state, state = world.mp_pipe()
        world.start_in_subprocess(robot.run)

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
        print(f"Origin: {origin}")

        start, i = time.monotonic(), 0
        while i < len(trajectory) and not world.should_stop:
            pos, duration = trajectory[i]
            if time.monotonic() > start + duration:
                print(f"Moving to {pos + origin.translation}")
                commands.emit(command.CartesianMove(geom.Transform3D(pos + origin.translation, origin.rotation)))
                i += 1
            else:
                time.sleep(0.01)

        print("Finishing")
