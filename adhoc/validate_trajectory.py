import glob
import os
from typing import List

import torch
import fire
import numpy as np
import rerun as rr

import geom
from positronic.tools.registration import AbsoluteTrajectory, RelativeTrajectory, batch_registration, grid_search_rotation, h_grid_search_rotation, pca_registration, register_trajectories


def split(xyz, k: int = 20):
    a = []

    xyz = np.array(xyz)

    for i in range(k):
        a.append(geom.Transform3D(translation=xyz / k))

    return a

WAYPOINTS = RelativeTrajectory([
    # Initial joint configuration is handled separately
    *split([0.0, 0.0, 0.2]),
    # YX plane triangle 0.2 side
    *split([0.0, 0.2, 0.0]),
    *split([-0.2, 0.0, 0.0]),
    *split([0.2, -0.2, 0.0]),

    # XY plane square 0.15 side
    *split([-0.15, 0.0, 0.0]),
    *split([0.0, -0.15, 0.0]),
    *split([0.15, 0.0, 0.0]),
    *split([0.0, 0.15, 0.0]),

    # XZ plane square 0.1 side
    *split([-0.1, 0.0, 0.0]),
    *split([0.0, 0.0, -0.1]),
    *split([0.1, 0.0, 0.0]),
    *split([0.0, 0.0, 0.1]),

    # Not parallel hourglass
    *split([-0.05, -0.05, -0.05]),
    *split([0.0, 0.0, 0.05]),
    *split([-0.05, -0.05, -0.05]),
    *split([0.0, 0.0, 0.05]),
    *split([0.1, 0.1, 0.0]),
    # return to start
    *split([0.0, 0.0, -0.2]),
]).to_absolute()


def polt_trajectory(trajectory: AbsoluteTrajectory, name: str, color: List[int] = [255, 0, 0, 255]):
    points = []
    for idx, pos in enumerate(trajectory):
        rr.set_time_sequence("trajectory", idx)
        points.append(pos.translation)

    rr.log(f"trajectory/{name}", rr.Points3D(positions=np.array(points), radii=np.array([0.005]), colors=np.array([color])))


def apply_trajectory(file, registration_transform: geom.Transform3D = None):
    data = torch.load(file)

    registration_transform = registration_transform or geom.Transform3D()
    res = [geom.Transform3D()]

    for idx, (trans, rot) in enumerate(zip(data['robot_position_translation'], data['robot_position_quaternion'])):
        diff = geom.Transform3D(translation=trans.numpy(), rotation=rot.numpy())

        res.append(res[-1] * (registration_transform.inv * diff * registration_transform))

    return res

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

def get_umi_trajectory(file: str):
    data = torch.load(file)

    right_position = AbsoluteTrajectory([
        geom.Transform3D(translation=t.numpy(), rotation=r.numpy())
        for t, r in zip(data['umi_right_translation'][50:], data['umi_right_quaternion'][50:])
    ])

    left_position = AbsoluteTrajectory([
        geom.Transform3D(translation=t.numpy(), rotation=r.numpy())
        for t, r in zip(data['umi_left_translation'][50:], data['umi_left_quaternion'][50:])
    ])

    umi_relative_trajectory = umi_relative(left_position, right_position)

    return umi_relative_trajectory


def plot_trajectories(trajectory_dir: str, registration_transform: geom.Transform3D = None):
    rr.init("trajectory", spawn=True)
    registration_transform = registration_transform or geom.Transform3D()
    waypoints_trajectory = WAYPOINTS
    polt_trajectory(waypoints_trajectory, "target", color=[255, 0, 0, 255])

    trajectories = []
    for i, file in enumerate(sorted(glob.glob(os.path.join(trajectory_dir, '*.pt')))):
        data = torch.load(file)

        right_position = AbsoluteTrajectory([
            geom.Transform3D(translation=t.numpy(), rotation=r.numpy())
            for t, r in zip(data['umi_right_translation'][50:], data['umi_right_quaternion'][50:])
        ])

        left_position = AbsoluteTrajectory([
            geom.Transform3D(translation=t.numpy(), rotation=r.numpy())
            for t, r in zip(data['umi_left_translation'][50:], data['umi_left_quaternion'][50:])
        ])

        umi_relative_trajectory = umi_relative(left_position, right_position)

        trajectories.append(umi_relative_trajectory)

    # registration_transform = h_grid_search_rotation(trajectories, waypoints_trajectory, traj_subsample=0.2)
    registration_transform = geom.Transform3D(rotation=geom.Rotation.from_euler([1.36897958, -0.73992762, 1.39720004]))

    print(registration_transform.rotation.as_euler)



    for i, umi_trajectory in enumerate(trajectories):
        # start = registration_transform.inv * umi_trajectory[0]
        # start = umi_trajectory[0]
        robot_relative = RelativeTrajectory([registration_transform.inv * pos * registration_transform for pos in umi_trajectory])
        registered_trajectory = robot_relative.to_absolute()

        polt_trajectory(registered_trajectory, f"{i}umi", color=[0, 255, 0, 255])

if __name__ == "__main__":
    fire.Fire(plot_trajectories)