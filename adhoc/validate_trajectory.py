import glob
import os
from typing import List

import torch
import fire
import numpy as np
import rerun as rr

import geom

WAYPOINTS = [
    # Y axis movement
    *[geom.Transform3D(translation=[0.0, y, 0.0])
      for y in np.arange(0.0, 0.2, 0.01)] +
    [geom.Transform3D(translation=[0.0, y, 0.0])
      for y in np.arange(0.2, -0.2, -0.01)] +
    [geom.Transform3D(translation=[0.0, y, 0.0])
      for y in np.arange(-0.2, 0.0, 0.01)] +
    # X axis movement
    [geom.Transform3D(translation=[x, 0.0, 0.0])
      for x in np.arange(0.0, -0.15, -0.01)] +
    [geom.Transform3D(translation=[x, 0.0, 0.0])
      for x in np.arange(-0.15, 0.15, 0.01)] +
    [geom.Transform3D(translation=[x, 0.0, 0.0])
      for x in np.arange(0.15, 0.0, -0.01)] +
    # Z axis movement
    [geom.Transform3D(translation=[0.0, 0.0, z])
      for z in np.arange(0.0, 0.05, 0.01)] +
    [geom.Transform3D(translation=[0.0, 0.0, z])
      for z in np.arange(0.05, -0.05, -0.01)] +
    [geom.Transform3D(translation=[0.0, 0.0, z])
      for z in np.arange(-0.05, 0.0, 0.01)],
]

def pca(points: List[geom.Transform3D]):
    # Convert points to numpy array of translations
    points_array = np.array([p.translation for p in points])

    # Center the points by subtracting mean
    mean = np.mean(points_array, axis=0)
    centered = points_array - mean

    # Perform SVD
    U, S, Vh = np.linalg.svd(centered)

    # Principal components are the right singular vectors (rows of Vh)
    principal_components = Vh

    # Explained variance ratio
    explained_variance = S**2 / np.sum(S**2)

    return principal_components, explained_variance, mean


def apply_trajectory(file, registration_transform: geom.Transform3D = None):
    data = torch.load(file)

    registration_transform = registration_transform or geom.Transform3D()
    res = [geom.Transform3D()]

    for idx, (trans, rot) in enumerate(zip(data['robot_position_translation'], data['robot_position_quaternion'])):
        diff = geom.Transform3D(translation=trans.numpy(), quaternion=rot.numpy())

        res.append(res[-1] * (registration_transform.inv * diff * registration_transform))

    return res


def register_trajectories(dir: str, waypoints: List[geom.Transform3D] = WAYPOINTS):
    target = [geom.Transform3D()]

    for w in waypoints:
        target.append(target[-1] * w)

    waypoints_pca = pca(target)

    trajs = []
    for file in glob.glob(os.path.join(dir, '*.pt')):
        trajs.extend(apply_trajectory(file))

    trajs_pca = pca(trajs)

    w_rotation = geom.Quaternion.from_rotation_matrix(waypoints_pca[0])
    t_rotation = geom.Quaternion.from_rotation_matrix(trajs_pca[0])

    reg_transform = geom.Transform3D(translation=np.zeros(3), quaternion=t_rotation * w_rotation.inv)

    return reg_transform

def plot_trajectories(trajectory_dir: str, registration_transform: geom.Transform3D = None):
    rr.init("trajectory", spawn=True)
    registration_transform = registration_transform or geom.Transform3D()

    for i, file in enumerate(glob.glob(os.path.join(trajectory_dir, '*.pt'))):
        data = torch.load(file)

        position = geom.Transform3D()

        for idx, (trans, rot) in enumerate(zip(data['robot_position_translation'], data['robot_position_quaternion'])):
            # Create transform from the relative transform data
            diff = geom.Transform3D(translation=trans.numpy(), quaternion=rot.numpy())

            position = position *  diff

            pout = registration_transform.inv * position * registration_transform
            rr.set_time_sequence("trajectory", idx)
            rr.log(f"trajectory/{i}", rr.Transform3D(translation=pout.translation, quaternion=pout.quaternion))


if __name__ == "__main__":
    fire.Fire(plot_trajectories)