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


def polt_trajectory(trajectory: AbsoluteTrajectory, name: str, color: List[int] = [255, 0, 0, 255]):
    points = []
    for idx, pos in enumerate(trajectory):
        rr.set_time_sequence("trajectory", idx)
        points.append(pos.translation)

    rr.log(f"trajectory/{name}", rr.Points3D(positions=np.array(points), radii=np.array([0.005]), colors=np.array([color])))


def interpol(xyz, k: int = 100):
    a = []

    xyz = np.array(xyz)

    for _ in range(k):
        a.append(geom.Transform3D(translation=xyz / k))

    return a


WAYPOINTS = RelativeTrajectory([
    # Initial joint configuration is handled separately
    *interpol([0.0, 0.0, 0.2]),
    # YX plane triangle 0.2 side
    *interpol([0.0, 0.2, 0.0]),
    *interpol([-0.2, 0.0, 0.0]),
    *interpol([0.2, -0.2, 0.0]),

    # XY plane square 0.15 side
    *interpol([-0.15, 0.0, 0.0]),
    *interpol([0.0, -0.15, 0.0]),
    *interpol([0.15, 0.0, 0.0]),
    *interpol([0.0, 0.15, 0.0]),

    # XZ plane square 0.1 side
    *interpol([-0.1, 0.0, 0.0]),
    *interpol([0.0, 0.0, -0.1]),
    *interpol([0.1, 0.0, 0.0]),
    *interpol([0.0, 0.0, 0.1]),

    # Not parallel hourglass
    *interpol([-0.05, -0.05, -0.05]),
    *interpol([0.0, 0.0, 0.05]),
    *interpol([-0.05, -0.05, -0.05]),
    *interpol([0.0, 0.0, 0.05]),
    *interpol([0.1, 0.1, 0.0]),
    # return to start
    *interpol([0.0, 0.0, -0.2]),
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
        right_delta = right_position[i-1].inv * right_position[i]

        # Apply the relative transformation to the gripper frame
        transform = relative_gripper_transform.inv * right_delta * relative_gripper_transform

        result.append(transform)

    return RelativeTrajectory(result)

@ir.ironic_system(
    input_ports=['start'],
    input_props=['webxr_position', 'robot_ee_position'],
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

    @ir.on_message('start')
    async def start(self, message: ir.Message):
        self.started = True

    async def step(self):
        if not self.started:
            return ir.State.ALIVE

        if self.absolute_trajectory is None:
            robot_position = (await self.ins.robot_ee_position()).data
            self.absolute_trajectory = self.relative_trajectory.to_absolute(robot_position)

        await self.outs.target_robot_position.write(ir.Message(data=self.absolute_trajectory[self.index]))
        await asyncio.sleep(0.01)

        umi_position = (await self.ins.webxr_position()).data
        robot_position = (await self.ins.robot_ee_position()).data

        assert umi_position is not ir.NoValue
        assert robot_position is not ir.NoValue
        assert isinstance(umi_position['left'], geom.Transform3D)
        assert isinstance(umi_position['right'], geom.Transform3D)

        self.data.append({
            'left_gripper': umi_position['left'],
            'right_gripper': umi_position['right'],
            'robot_position': robot_position,
        })

        self.index += 1

        if self.index >= len(self.absolute_trajectory):
            self.P = self._perform_umi_registration()
            return ir.State.FINISHED

        return ir.State.ALIVE

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

        polt_trajectory(robot_trajectory, "target", color=[255, 0, 0, 255])

        umi_relative_trajectory = umi_relative(left_trajectory, right_trajectory).to_absolute()

        translations = np.array([A.translation for A in umi_relative_trajectory])
        translations_target = np.array([A.translation for A in robot_trajectory])

        P, _ = orthogonal_procrustes(translations, translations_target)

        transform = geom.Transform3D(rotation=geom.Rotation.from_rotation_matrix(P))

        registered_trajectory = RelativeTrajectory([transform.inv * A * transform for A in umi_relative_trajectory.to_relative()]).to_absolute()
        polt_trajectory(registered_trajectory, "registered", color=[0, 255, 0, 255])

        return P



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
            webxr_position=ir.utils.last_value(webxr.outs.controller_positions),
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
