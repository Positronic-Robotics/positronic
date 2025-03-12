import asyncio
import numpy as np
from typing import List
import geom
import fire
from geom.trajectory import AbsoluteTrajectory, RelativeTrajectory

from scipy.linalg import orthogonal_procrustes
import ironic as ir
import rerun as rr

import positronic.cfg.ui
import positronic.cfg.hardware.roboarms
from positronic.policy_runner import PolicyRunnerSystem


def _polt_trajectory(trajectory: AbsoluteTrajectory, name: str, color: List[int] = [255, 0, 0, 255]):
    points = []

    for idx, pos in enumerate(trajectory):
        rr.set_time_sequence("trajectory", idx)
        points.append(pos.translation)

    rr.log(
        f"trajectory/{name}",
        rr.Points3D(
            positions=np.array(points),
            radii=np.array([0.005]),
            colors=np.array([color]),
        ),
    )


# Arbitrary trajectory for registration
WAYPOINTS = RelativeTrajectory([
    # Initial joint configuration is handled separately
    geom.Transform3D(translation=[0.0, 0.0, 0.2]),
    # YX plane triangle 0.2 side
    geom.Transform3D(translation=[0.0, 0.2, 0.0]),
    geom.Transform3D(translation=[-0.2, 0.0, 0.0]),
    geom.Transform3D(translation=[0.2, -0.2, 0.0]),

    # XY plane square 0.15 side
    geom.Transform3D(translation=[-0.15, 0.0, 0.0]),
    geom.Transform3D(translation=[0.0, -0.15, 0.0]),
    geom.Transform3D(translation=[0.15, 0.0, 0.0]),
    geom.Transform3D(translation=[0.0, 0.15, 0.0]),

    # XZ plane square 0.1 side
    geom.Transform3D(translation=[-0.1, 0.0, 0.0]),
    geom.Transform3D(translation=[0.0, 0.0, -0.1]),
    geom.Transform3D(translation=[0.1, 0.0, 0.0]),
    geom.Transform3D(translation=[0.0, 0.0, 0.1]),

    # Not parallel hourglass
    geom.Transform3D(translation=[-0.05, -0.05, -0.05]),
    geom.Transform3D(translation=[0.0, 0.0, 0.05]),
    geom.Transform3D(translation=[-0.05, -0.05, -0.05]),
    geom.Transform3D(translation=[0.0, 0.0, 0.05]),
    geom.Transform3D(translation=[0.1, 0.1, 0.0]),
    # return to start
    geom.Transform3D(translation=[0.0, 0.0, -0.2]),
])


def umi_relative(left_position: AbsoluteTrajectory, right_position: AbsoluteTrajectory):
    """
    Calculate the relative transformation between left and right positions.
    Similar to ee_position in UmiCS class.
    Args:
        left_position: Trajectory of left tracker positions
        right_position: Trajectory of right tracker positions
    Returns:
        AbsoluteTrajectory: Relative transformation trajectory
    """
    if len(left_position) == 0 or len(right_position) == 0:
        return AbsoluteTrajectory([])

    # Calculate initial relative gripper transform
    relative_gripper_transform = left_position[0].inv * right_position[0]

    result = []
    for i in range(1, len(right_position)):
        # Calculate relative transformation between consecutive right positions
        right_delta = right_position[i - 1].inv * right_position[i]

        # Apply the relative transformation to the gripper frame
        transform = relative_gripper_transform.inv * right_delta * relative_gripper_transform

        result.append(transform)

    return RelativeTrajectory(result)


@ir.ironic_system(
    input_ports=['start', 'webxr_position'],
    input_props=['robot_ee_position'],
    output_ports=['target_robot_position'],
)
class RegistrationSystem(ir.ControlSystem):
    def __init__(self, trajectory: RelativeTrajectory):
        super().__init__()
        self.relative_trajectory = trajectory
        self.absolute_trajectory = None
        self.index = 0
        self.data = []
        self.started = False
        self.step_throttler = ir.utils.Throttler(2)
        self.fps_counter = ir.utils.FPSCounter("Registration")

    @ir.on_message('start')
    async def start(self, message: ir.Message):
        self.started = True

    @ir.on_message('webxr_position')
    async def webxr_position(self, message: ir.Message):
        if not self.started:
            return

        assert message is not ir.NoValue
        assert isinstance(message.data['left'], geom.Transform3D)
        assert isinstance(message.data['right'], geom.Transform3D)

        robot_position = (await self.ins.robot_ee_position()).data
        assert robot_position is not ir.NoValue

        self.data.append({
            'left_gripper': message.data['left'],
            'right_gripper': message.data['right'],
            'robot_position': robot_position,
        })

        print(len(self.data))

    async def step(self):
        if not self.started:
            return ir.State.ALIVE

        self.fps_counter.tick()

        if self.absolute_trajectory is None:
            robot_position = (await self.ins.robot_ee_position()).data
            self.absolute_trajectory = self.relative_trajectory.to_absolute(robot_position)

        await self._next_robot_position()

        if self.index >= len(self.absolute_trajectory):
            self.started = False
            self.registration_transform = self._perform_umi_registration()
            return ir.State.FINISHED

        return ir.State.ALIVE

    async def _next_robot_position(self):
        if self.step_throttler():
            await self.outs.target_robot_position.write(ir.Message(data=self.absolute_trajectory[self.index]))
            self.index += 1
        else:
            await asyncio.sleep(0.01)

    def _perform_umi_registration(self):
        """
        Compute the optimal transformation P such that Ai ≈ P^-1 * Bi * P for all i.

        This uses a closed-form solution based on SVD to find the optimal transformation.

        Returns:
            geom.Transform3D: The optimal transformation P
        """
        if not self.data:
            raise ValueError("No data collected for registration")

        import pickle

        with open("data.pkl", "wb") as f:
            pickle.dump(self.data, f)

        rr.init("registration", spawn=True)

        robot_trajectory = AbsoluteTrajectory([d['robot_position'] for d in self.data]).to_relative().to_absolute()

        left_trajectory = AbsoluteTrajectory([d['left_gripper'] for d in self.data])
        right_trajectory = AbsoluteTrajectory([d['right_gripper'] for d in self.data])

        _polt_trajectory(robot_trajectory, "target", color=[255, 0, 0, 255])

        umi_relative_trajectory = umi_relative(left_trajectory, right_trajectory).to_absolute()

        translations = np.array([A.translation for A in umi_relative_trajectory])
        translations_target = np.array([A.translation for A in robot_trajectory])

        registration_mtx, _ = orthogonal_procrustes(translations, translations_target)
        registration_rotation = geom.Rotation.from_rotation_matrix(registration_mtx)

        if np.linalg.det(registration_mtx) < 0:
            print("Registration matrix is not a rotation matrix")

        transform = geom.Transform3D(rotation=registration_rotation)

        registered_trajectory = RelativeTrajectory([
            transform.inv * x * transform for x in umi_relative_trajectory.to_relative()
        ]).to_absolute()

        _polt_trajectory(registered_trajectory, "registered", color=[0, 255, 0, 255])

        print(f"Registration rotation QUAT: {transform.rotation.as_quat}")
        self._log_tracking_error(robot_trajectory, registered_trajectory)

        return transform

    def _log_tracking_error(
            self,
            robot_trajectory: AbsoluteTrajectory,
            registered_trajectory: AbsoluteTrajectory,
    ):
        robot_pos = np.array([x.translation for x in robot_trajectory])
        registered_pos = np.array([x.translation for x in registered_trajectory])

        error = np.linalg.norm(robot_pos - registered_pos, axis=1)
        print("=" * 100)
        print(f"Max Tracking error: {np.max(error)}")
        print(f"Mean Tracking error: {np.mean(error)}")
        for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            print(f"{percentile}th percentile Tracking error: {np.percentile(error, percentile)}")


@ir.config(webxr=positronic.cfg.ui.webxr_both, robot_arm=positronic.cfg.hardware.roboarms.franka_ik)
async def perform_registration(
    webxr: ir.ControlSystem,
    robot_arm: ir.ControlSystem,
):
    """
    This function performs registration procedure.

    Instructions:
    1. Start this function
    2. Connect to WebXR
    3. Install UMI gripper to the robot
    4. Press S button on keyboard and stay near the robot
    """

    policy_runner = PolicyRunnerSystem()
    registration = RegistrationSystem(WAYPOINTS)

    composed = ir.compose(
        webxr,
        policy_runner,
        robot_arm.bind(target_position=registration.outs.target_robot_position),
        registration.bind(
            webxr_position=webxr.outs.controller_positions,
            robot_ee_position=robot_arm.outs.position,
            start=policy_runner.outs.start_policy,
        ),
    )

    print("Press S after you attach the UMI gripper to the robot...")

    await ir.utils.run_gracefully(composed)


async def main():
    await perform_registration.override_and_instantiate()


def sync_main():
    asyncio.run(main())


if __name__ == "__main__":
    fire.Fire(sync_main)
