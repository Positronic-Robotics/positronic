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


def _plot_trajectory(trajectory: AbsoluteTrajectory, name: str, color: List[int] = [255, 0, 0, 255]):
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


def umi_relative(left_trajectory: AbsoluteTrajectory, right_trajectory: AbsoluteTrajectory):
    """
    Calculate the relative trajectory of the grippers in a frame that doesn't depend on the global reference frame.

    Args:
        left_trajectory: Trajectory of left tracker positions
        right_trajectory: Trajectory of right tracker positions
    Returns:
        AbsoluteTrajectory: Relative transformation trajectory
    """
    assert len(left_trajectory) == len(right_trajectory)
    assert len(left_trajectory) > 0

    # Calculate initial relative gripper transform
    relative_gripper_transform = left_trajectory[0].inv * right_trajectory[0]

    result = []
    for i in range(1, len(right_trajectory)):
        # Calculate relative transformation between consecutive right positions
        right_delta = right_trajectory[i - 1].inv * right_trajectory[i]

        # Apply the relative transformation to the gripper frame
        transform = relative_gripper_transform.inv * right_delta * relative_gripper_transform

        result.append(transform)

    return RelativeTrajectory(result)


@ir.ironic_system(
    input_ports=['webxr_position'],
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
        self.step_throttler = ir.utils.Throttler(2)
        self.fps_counter = ir.utils.FPSCounter("Registration")

    @ir.on_message('webxr_position')
    async def webxr_position(self, message: ir.Message):
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

    async def step(self):
        self.fps_counter.tick()

        if self.absolute_trajectory is None:
            robot_position = (await self.ins.robot_ee_position()).data
            self.absolute_trajectory = self.relative_trajectory.to_absolute(robot_position)

        await self._next_robot_position()

        if self.index >= len(self.absolute_trajectory):
            self.registration_transform = self._perform_umi_registration()
            return ir.State.FINISHED

        return ir.State.ALIVE

    async def _next_robot_position(self):
        # TODO: we need to add an ability for a robot to execute synchronous commands
        if self.step_throttler():
            await self.outs.target_robot_position.write(ir.Message(data=self.absolute_trajectory[self.index]))
            self.index += 1

    def _perform_umi_registration(self):
        """
        Compute the optimal transformation P such that Ai ≈ P^-1 * Bi * P for all i.

        This uses a closed-form solution based on SVD to find the optimal transformation.

        Returns:
            geom.Transform3D: The optimal transformation P
        """
        if not self.data:
            raise ValueError("No data collected for registration")

        rr.init("registration", spawn=True)

        robot_trajectory = AbsoluteTrajectory([d['robot_position'] for d in self.data])
        # make it start from the origin
        robot_trajectory = robot_trajectory.to_relative().to_absolute(geom.Transform3D.identity)

        left_trajectory = AbsoluteTrajectory([d['left_gripper'] for d in self.data])
        right_trajectory = AbsoluteTrajectory([d['right_gripper'] for d in self.data])

        _plot_trajectory(robot_trajectory, "target", color=[255, 0, 0, 255])

        umi_trajectory = umi_relative(left_trajectory, right_trajectory).to_absolute(geom.Transform3D.identity)

        translations = np.array([x.translation for x in umi_trajectory])
        translations_target = np.array([x.translation for x in robot_trajectory])

        registration_mtx, _ = orthogonal_procrustes(translations, translations_target)
        registration_rotation = geom.Rotation.from_rotation_matrix(registration_mtx)

        if np.linalg.det(registration_mtx) < 0:
            print("Registration matrix is not a rotation matrix")

        transform = geom.Transform3D(rotation=registration_rotation)

        registered_trajectory = RelativeTrajectory([
            transform.inv * x * transform for x in umi_trajectory.to_relative()
        ]).to_absolute(geom.Transform3D.identity)

        _plot_trajectory(registered_trajectory, "registered", color=[0, 255, 0, 255])

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

    registration = RegistrationSystem(WAYPOINTS)

    composed = ir.compose(
        webxr,
        robot_arm.bind(target_position=registration.outs.target_robot_position),
        registration.bind(
            webxr_position=webxr.outs.controller_positions,
            robot_ee_position=robot_arm.outs.position,
        ),
    )

    def _ask_input_to_start():
        print("Press Enter after you attach the UMI gripper to the robot...")
        input()

    await ir.utils.run_gracefully(composed, after_setup_fn=_ask_input_to_start)


async def main(**kwargs):
    await perform_registration.override_and_instantiate(**kwargs)


def sync_main(**kwargs):
    asyncio.run(main(**kwargs))


if __name__ == "__main__":
    fire.Fire(sync_main)
