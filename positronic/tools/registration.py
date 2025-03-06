from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

import geom


class AbsoluteTrajectory:
    def __init__(self, absolute_positions: List[geom.Transform3D]):
        self.absolute_positions = absolute_positions

    def __len__(self):
        return len(self.absolute_positions)

    def __iter__(self):
        return iter(self.absolute_positions)

    def __getitem__(self, index: int) -> geom.Transform3D:
        return self.absolute_positions[index]

    def __str__(self):
        return f"[{', '.join([str(p) for p in self.absolute_positions])}]"

    def to_relative(self) -> 'RelativeTrajectory':
        relative_positions = []

        for i in range(1, len(self.absolute_positions)):
            relative_positions.append(self.absolute_positions[i-1].inv * self.absolute_positions[i])

        return RelativeTrajectory(relative_positions)

    def distance(self, other: 'AbsoluteTrajectory') -> float:
        assert len(self) == len(other)

        matrix_l2 = np.linalg.norm(
            np.array([p.as_matrix for p in self.absolute_positions]) - np.array([p.as_matrix for p in other.absolute_positions]),
            axis=2,
        )

        return np.mean(matrix_l2)

    def distance_closest_positions(self, other: 'AbsoluteTrajectory') -> float:
        distances = np.zeros((len(self), len(other)))

        self_tr = np.array([p.translation for p in self.absolute_positions])
        other_tr = np.array([p.translation for p in other.absolute_positions])

        # Compute pairwise differences and norms in a vectorized way
        diff = self_tr[:, None, :] - other_tr[None, :, :]
        distances = np.linalg.norm(diff, axis=2)

        return distances

    def mean_translation(self) -> np.ndarray:
        return np.mean([p.translation for p in self.absolute_positions], axis=0)

    def mean_rotvec(self) -> np.ndarray:
        return np.mean([p.rotation.as_rotvec for p in self.absolute_positions], axis=0)

    def extend(self, other: 'AbsoluteTrajectory'):
        self.absolute_positions.extend(other.absolute_positions)

class RelativeTrajectory:
    def __init__(self, relative_positions: List[geom.Transform3D]):
        self.relative_positions = relative_positions

    def __len__(self):
        return len(self.relative_positions)

    def __iter__(self):
        return iter(self.relative_positions)

    def __getitem__(self, index: int) -> geom.Transform3D:
        return self.relative_positions[index]

    def __str__(self):
        return f"[{', '.join([str(p) for p in self.relative_positions])}]"

    def to_absolute(self, start_position: Optional[geom.Transform3D] = None) -> AbsoluteTrajectory:
        if start_position is None:
            start_position = geom.Transform3D()

        absolute_positions = [start_position]

        for pos in self.relative_positions:
            absolute_positions.append(absolute_positions[-1] * pos)

        return AbsoluteTrajectory(absolute_positions)

    def rotation_distances(self, other: 'RelativeTrajectory') -> float:
        distances = np.zeros((len(self), len(other)))

        for i in range(len(self)):
            for j in range(len(other)):
                distances[i, j] = self[i].rotation.angle_distance(other[j].rotation)

        return distances

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


def align_with_pca(trajectory: AbsoluteTrajectory):
    pca_transform = get_pca_transform(trajectory)

    return trajectory.to_relative().to_absolute(start_position=pca_transform * trajectory[0])

def get_pca_transform(trajectory: AbsoluteTrajectory):
    principal_components, explained_variance, mean = pca(trajectory)

    pca_transform = geom.Transform3D(translation=mean, rotation=geom.Rotation.from_rotation_matrix(principal_components).inv)

    return pca_transform.inv


def register_trajectories(trajectory: AbsoluteTrajectory, target: AbsoluteTrajectory):

    def objective(x):
        reg_transform = geom.Transform3D(translation=x[:3], rotation=geom.Rotation.from_quat(x[3:]))

        start_position = reg_transform * target[0]

        return trajectory.distance(target.to_relative().to_absolute(start_position=start_position)).mean()

    def objective_closest_positions(x):
        reg_transform = geom.Transform3D(translation=x[:3], rotation=geom.Rotation.from_quat(x[3:]))

        start_position = reg_transform * target[0]

        distances = trajectory.distance_closest_positions(target.to_relative().to_absolute(start_position=start_position))
        mins_1 = np.min(distances, axis=1)
        mins_2 = np.min(distances, axis=0)

        return np.sum(mins_1) + np.sum(mins_2)


    target_fn = objective if len(trajectory) == len(target) else objective_closest_positions

    init_translation = np.zeros(3)

    res = minimize(
        target_fn,
        np.array([*init_translation, 1.0, 0.0, 0.0, 0.0]),
        method='powell',
    )
    print(res)

    transform = geom.Transform3D(translation=res.x[:3], rotation=geom.Rotation.from_quat(res.x[3:]))

    return transform


def batch_registration(trajectories: List[AbsoluteTrajectory], target: AbsoluteTrajectory):

    repr = geom.Rotation.Representation.QUAT


    def objective_closest_positions(x):
        reg_transform = geom.Transform3D(translation=x[:3], rotation=geom.Rotation.create_from(x[3:], repr))

        start_position = reg_transform * target[0]
        target_registered = target.to_relative().to_absolute(start_position=start_position)

        dists = [trajectory.distance_closest_positions(target_registered) for trajectory in trajectories]

        sums = dists
        mins_1 = [np.min(sum, axis=1).mean() for sum in sums]
        mins_2 = [np.min(sum, axis=0).mean() for sum in sums]

        return np.mean(mins_1) + np.mean(mins_2)

    init_r = np.zeros(repr.size)

    if repr == geom.Rotation.Representation.QUAT:
        init_r = np.array([1.0, 0.0, 0.0, 0.0])

    #init_translation = trajectories[0].mean_translation() =
    init_translation = np.zeros(3)

    res = minimize(
        objective_closest_positions,
        np.array([*init_translation, *init_r]),
        method='powell',
    )
    print(res)

    return geom.Transform3D(translation=res.x[:3], rotation=geom.Rotation.create_from(res.x[3:], repr))


def pca_registration(trajectory: AbsoluteTrajectory, target: AbsoluteTrajectory):
    target_pca_transform = get_pca_transform(target)
    trajectory_pca_transform = get_pca_transform(trajectory)

    return trajectory_pca_transform.inv * target_pca_transform


def batch_pca_registration(trajectories: List[AbsoluteTrajectory], target: AbsoluteTrajectory):
    t = trajectories[0]
    for trajectory in trajectories[1:]:
        t.extend(trajectory)

    pca_transform = get_pca_transform(t)
    target_pca_transform = get_pca_transform(target)

    return pca_transform.inv * target_pca_transform

import tqdm

def grid_search_rotation(
        trajectories: List[AbsoluteTrajectory],
        target: AbsoluteTrajectory,
        roll_range=(-np.pi, np.pi),
        pitch_range=(-np.pi, np.pi),
        yaw_range=(-np.pi, np.pi),
        steps=8
):
    # Create separate ranges for each angle
    roll_angles = np.linspace(roll_range[0], roll_range[1], steps)
    pitch_angles = np.linspace(pitch_range[0], pitch_range[1], steps)
    yaw_angles = np.linspace(yaw_range[0], yaw_range[1], steps)

    # Create meshgrid of all angle combinations
    roll_grid, pitch_grid, yaw_grid = np.meshgrid(roll_angles, pitch_angles, yaw_angles)

    # Flatten the grids to iterate through all combinations
    roll_flat = roll_grid.flatten()
    pitch_flat = pitch_grid.flatten()
    yaw_flat = yaw_grid.flatten()

    best_error = float('inf')
    best_transform = None
    r_target = target.to_relative()

    # Create iterator over all combinations
    total_combinations = len(roll_flat)
    for i in tqdm.tqdm(range(total_combinations)):
        roll, pitch, yaw = roll_flat[i], pitch_flat[i], yaw_flat[i]
        transform = geom.Transform3D(rotation=geom.Rotation.from_euler([roll, pitch, yaw]))

        error = 0
        for trajectory in trajectories:
            dmtx = trajectory.distance_closest_positions(r_target.to_absolute(start_position=transform * target[0]))
            error += np.sum(np.min(dmtx, axis=1)) + np.sum(np.min(dmtx, axis=0))

        if error < best_error:
            best_error = error
            best_transform = transform
    return best_transform, best_error


def h_grid_search_rotation(trajectories: List[AbsoluteTrajectory], target: AbsoluteTrajectory, iters=20, traj_subsample=1.0):
    ranges = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

    radius = 2 * np.pi
    tr_sub = []

    for trajectory in trajectories:
        idxs = np.linspace(0, len(trajectory) - 1, int(len(trajectory) * traj_subsample), dtype=int)
        tr_sub.append(AbsoluteTrajectory([trajectory.absolute_positions[i] for i in idxs]))

    best_transform = None
    best_error = float('inf')
    for i in range(iters):
        t, error = grid_search_rotation(
            tr_sub,
            target,
            roll_range=ranges[0],
            pitch_range=ranges[1],
            yaw_range=ranges[2],
            steps=6
        )
        print(error)
        if error < best_error:
            best_error = error
            best_transform = t

        roll, pitch, yaw = best_transform.rotation.as_euler
        radius *= 0.5
        ranges[0] = (roll - radius, roll + radius)
        ranges[1] = (pitch - radius, pitch + radius)
        ranges[2] = (yaw - radius, yaw + radius)

    return best_transform
