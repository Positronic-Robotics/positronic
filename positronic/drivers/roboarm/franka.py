import asyncio
from collections import deque
from enum import Enum
import threading
import time

import franky
import geom
import ironic as ir
from .status import RobotStatus
from geom import Rotation, Transform3D


class CartesianMode(Enum):
    LIBFRANKA = "libfranka"
    POSITRONIC = "positronic"


# TODO: Extract gripper into a separate control system
@ir.ironic_system(input_ports=['target_position', 'target_grip', 'reset'],
                  output_ports=['status'],
                  output_props=['position', 'grip', 'joint_positions', 'ext_force_base', 'ext_force_ee'])
class Franka(ir.ControlSystem):

    def __init__(self,
                 ip: str,
                 relative_dynamics_factor: float = 0.2,
                 gripper_speed: float = 0.02,
                 realtime_config: franky.RealtimeConfig = franky.RealtimeConfig.Ignore,
                 collision_behavior=None,
                 home_joints_config=None,
                 cartesian_mode: CartesianMode = CartesianMode.LIBFRANKA):
        super().__init__()
        self.robot = franky.Robot(ip, realtime_config=realtime_config)
        self._set_default_behavior()
        self.robot.relative_dynamics_factor = relative_dynamics_factor
        if collision_behavior is not None:
            self.robot.set_collision_behavior(*collision_behavior)
        self.robot.set_joint_impedance([150, 150, 150, 125, 125, 250, 250])
        # self.robot.set_cartesian_impedance([150, 150, 150, 300, 300, 300])

        self.home_joints_config = home_joints_config or [0.0, -0.31, 0.0, -1.53, 0.0, 1.522, 0.785]
        # self.home_joints_config = [-0.00485582, -0.64545132,  0.00633443, -1.91874464, -0.00588651,  2.31487911,  0.78129702]
        self.cartesian_mode = cartesian_mode
        self.last_q = None
        self.motion_counter = 0

        self._command_queue = deque()
        self._command_mutex = threading.Lock()
        self._motion_exit_event = threading.Event()
        self._motion_thread = threading.Thread(target=self._motion_thread, daemon=True)
        self._main_loop = asyncio.get_running_loop()

        try:
            self.gripper = franky.Gripper(ip)
            self.gripper_speed = gripper_speed
            self._gripper_grasped = None
        except Exception as e:
            print(f"Did not connect to gripper: {e}")
            self.gripper = None
            self.gripper_speed = 0.0
            self._gripper_grasped = None

    async def setup(self):
        self._motion_thread.start()
        await self.handle_reset(ir.Message(True))

    async def cleanup(self):
        print("Franka stopping")
        self._motion_exit_event.set()
        self._motion_thread.join()
        self.robot.stop()
        if self.gripper:
            self.gripper.open(self.gripper_speed)
        print("Franka stopped")

    def _motion_thread(self):
        fps = ir.utils.FPSCounter("Franka motion")
        motion_executed = 0

        while not self._motion_exit_event.is_set():
            motion = None
            with self._command_mutex:
                if self._command_queue:
                    motion, _ = self._command_queue.popleft()

            if motion:
                motion()
                motion_executed += 1
                print(f"Motion executed: {motion_executed}")
                fps.tick()
            else:
                time.sleep(0.01)  # Avoid busy-waiting
                continue

    def _submit_motion(self, motion: franky.Motion, asynchronous: bool = True):
        # There's only one async motion at a time in the queue, and any number of blocking motions
        # If motion is async, we delete existing async motion (keeping blocking motions),
        # and put the new in the end of the queue. If motion is not async, we just add it to the end
        with self._command_mutex:
            if asynchronous:
                self._command_queue = deque([(m, a) for m, a in self._command_queue if not a])
                self._command_queue.append((motion, asynchronous))
            else:
                self.motion_counter += 1
                print(f"Motion submitted: {self.motion_counter}")
                self._command_queue.append((motion, asynchronous))

    @ir.on_message('target_position')
    async def handle_target_position(self, message: ir.Message):
        if len(self._command_queue) > 0:
            return

        # pos = franky.Affine(translation=message.data.translation, quaternion=message.data.rotation.as_quat)
        pos = franky.Affine(translation=message.data.translation, quaternion=message.data.rotation.as_quat_xyzw)
        motion = None
        if self.cartesian_mode == CartesianMode.LIBFRANKA:
            motion = franky.CartesianMotion(pos, franky.ReferenceType.Absolute)
        else:
            if self.last_q is None:
                self.last_q = self.robot.current_joint_state.position
            self.last_q = self.robot.inverse_kinematics(pos, self.last_q)
            motion = franky.JointMotion(self.last_q, return_when_finished=True)

        def internal_motion():
            try:
                self.robot.move(motion, asynchronous=True)
            except franky.ControlException as e:
                self.robot.recover_from_errors()
                print(f"Motion failed for {message.data}: {e}")

        self._submit_motion(internal_motion, asynchronous=True)

    @ir.on_message('reset')
    async def handle_reset(self, message: ir.Message):
        """Commands the robot to start moving to the home position."""
        start_reset_future = self._main_loop.create_future()
        reset_done_future = self._main_loop.create_future()

        def _internal_reset():
            # Signal we're starting the reset
            self._main_loop.call_soon_threadsafe(lambda: start_reset_future.set_result(True))

            self.robot.join_motion(
                timeout=0.1
            )  # we need to set timeout here because with return_when_finished=False, the last motion never finishes
            motion = franky.JointWaypointMotion([franky.JointWaypoint(self.home_joints_config)])
            self.robot.move(motion, asynchronous=False)
            if self.gripper:
                self.gripper.homing()
                self._gripper_grasped = False

            self._main_loop.call_soon_threadsafe(lambda: reset_done_future.set_result(True))

        async def handle_status_transitions():
            await start_reset_future
            await self.outs.status.write(ir.Message(RobotStatus.RESETTING, ir.system_clock()))

            await reset_done_future
            await self.outs.status.write(ir.Message(RobotStatus.AVAILABLE, ir.system_clock()))

        self._main_loop.create_task(handle_status_transitions())
        self._submit_motion(_internal_reset, asynchronous=False)

    @ir.on_message('target_grip')
    async def handle_target_grip(self, message: ir.Message):
        if self.gripper is None:
            return

        if self._gripper_grasped:
            if message.data < 0.33:
                self.gripper.open(self.gripper_speed)
                self._gripper_grasped = False
        else:
            if message.data > 0.66:
                try:
                    self.gripper.grasp(0.0, self.gripper_speed, 20.0, epsilon_outer=1.0 * self.gripper.max_width)
                    self._gripper_grasped = True
                except franky.CommandException as e:
                    print(f"Grasping failed: {e}")

    @ir.out_property
    async def position(self):
        """End effector position in robot base coordinate frame."""
        pos = self.robot.current_pose.end_effector_pose
        return ir.Message(data=Transform3D(pos.translation, Rotation.from_quat_xyzw(pos.quaternion)))

    @ir.out_property
    async def joint_positions(self):
        return ir.Message(data=self.robot.current_joint_state.position)

    @ir.out_property
    async def grip(self):
        return ir.Message(data=self._gripper_grasped if self.gripper else 0.0, timestamp=ir.system_clock())

    @ir.out_property
    async def ext_force_base(self):
        return ir.Message(data=self.robot.state.O_F_ext_hat_K, timestamp=ir.system_clock())

    @ir.out_property
    async def ext_force_ee(self):
        return ir.Message(data=self.robot.state.K_F_ext_hat_K, timestamp=ir.system_clock())

    async def step(self) -> ir.State:
        return await super().step()

    def _set_default_behavior(self):
        self.robot.set_collision_behavior(
            lower_torque_threshold_acceleration=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            upper_torque_threshold_acceleration=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            lower_torque_threshold_nominal=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            upper_torque_threshold_nominal=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            lower_force_threshold_acceleration=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            upper_force_threshold_acceleration=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            lower_force_threshold_nominal=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            upper_force_threshold_nominal=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        )

        self.robot.set_joint_impedance([3000, 3000, 3000, 2500, 2500, 2000, 2000])
        self.robot.set_cartesian_impedance([3000, 3000, 3000, 300, 300, 300])
        self.robot.set_load(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.robot.relative_dynamics_factor = 1.0


if __name__ == "__main__":
    import numpy as np

    async def _main():
        franka = Franka("172.168.0.2")
        await franka.setup()

        target_port = ir.OutputPort('target')
        franka.bind(target_position=target_port)

        current_pos = (await franka.position()).data

        # Move in a small circle
        for t in np.linspace(0, 2 * np.pi, 50):
            tr = np.array(current_pos.translation)
            tr[0] += np.cos(t) * 0.1
            tr[1] += np.sin(t) * 0.1
            new_pos = Transform3D(tr, current_pos.rotation)

            await target_port.write(ir.Message(data=new_pos))
            await asyncio.sleep(0.1)

        await franka.cleanup()

    asyncio.run(_main())
