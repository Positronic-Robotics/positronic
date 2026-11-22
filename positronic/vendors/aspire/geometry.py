"""Turn a calibrated camera view and a mask into points, a bounding box and a grasp choice.

Ported from NVlabs/ASPIRE ``src/cap_harness/geometry.py`` at commit ``fa00f0e``, under the Apache License 2.0 (see
``NOTICE``). A failure raises ``ValueError``, as in ASPIRE.
"""

import numpy as np

from positronic import geom

from .contracts import CameraObservation, GraspCandidate, GraspSet, ObjectGeometry, PointCloud, Pose, Segmentation

# ASPIRE's top-down grasp: the cosine between the approach axis, the gripper's z, and straight down is at least this.
TOP_DOWN_ALIGNMENT = 0.8


def mask_to_point_cloud(
    observation: CameraObservation, segmentation: Segmentation, *, target_frame: str | None = None
) -> PointCloud:
    """Back-project a mask and optionally transform it to the camera parent frame."""
    if segmentation.frame != observation.frame:
        raise ValueError(f'segmentation frame {segmentation.frame!r} does not match camera frame {observation.frame!r}')
    if observation.depth_m is None or observation.intrinsics is None:
        raise ValueError(f'camera {observation.frame!r} has no depth or no intrinsics')
    if segmentation.mask.shape != observation.depth_m.shape:
        raise ValueError('segmentation mask shape does not match camera depth')
    rows, columns = np.nonzero(segmentation.mask)
    depths = observation.depth_m[rows, columns].astype(np.float64)
    valid = np.isfinite(depths) & (depths > 0.0)
    rows, columns, depths = rows[valid], columns[valid], depths[valid]
    if depths.size == 0:
        raise ValueError('segmentation contains no pixels with valid positive depth')
    intrinsics = observation.intrinsics
    x = (columns - intrinsics[0, 2]) * depths / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) * depths / intrinsics[1, 1]
    points, colors = np.column_stack((x, y, depths)), observation.rgb[rows, columns]
    destination = observation.frame if target_frame is None else target_frame
    if destination == observation.frame:
        return PointCloud(points, observation.frame, colors)
    if observation.camera_pose is None or destination != observation.camera_pose.frame:
        raise ValueError(f'camera {observation.frame!r} has no pose in frame {destination!r}')
    transform = observation.camera_pose.as_matrix()
    return PointCloud(points @ transform[:3, :3].T + transform[:3, 3], destination, colors)


def crop_point_cloud(point_cloud: PointCloud, lower_xyz: object, upper_xyz: object) -> PointCloud:
    """Crop public XYZ observations to inclusive axis-aligned bounds."""
    lower, upper = np.asarray(lower_xyz, dtype=np.float64), np.asarray(upper_xyz, dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(lower < upper):
        raise ValueError('lower_xyz must be strictly less than upper_xyz on each of three axes')
    keep = np.all((point_cloud.points >= lower) & (point_cloud.points <= upper), axis=1)
    if not np.any(keep):
        raise ValueError('point cloud contains no points inside the requested bounds')
    colors = None if point_cloud.colors is None else point_cloud.colors[keep]
    return PointCloud(point_cloud.points[keep], point_cloud.frame, colors)


def estimate_oriented_bounding_box(point_cloud: PointCloud) -> ObjectGeometry:
    """Estimate a deterministic PCA-oriented bounding box."""
    points = point_cloud.points
    if len(points) < 4:
        raise ValueError('at least four points are required for a 3D oriented bounding box')
    mean = np.mean(points, axis=0)
    centered = points - mean
    eigenvalues, axes = np.linalg.eigh(centered.T @ centered / len(points))
    axes = axes[:, np.argsort(eigenvalues)[::-1]]
    # Eigenvector signs are arbitrary: make each dominant component positive, then restore a right-handed basis.
    for column in range(3):
        if axes[np.argmax(np.abs(axes[:, column])), column] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    local = centered @ axes
    lower, upper = np.min(local, axis=0), np.max(local, axis=0)
    extents = upper - lower
    if np.any(extents <= 1e-9):
        raise ValueError('point cloud is degenerate and has no finite 3D bounding box')
    center = mean + axes @ (0.5 * (lower + upper))
    pose = Pose(center, geom.Rotation.from_rotation_matrix(axes).as_quat, point_cloud.frame)
    return ObjectGeometry(pose, extents, len(points))


def select_top_down_grasp(grasps: GraspSet) -> GraspCandidate | None:
    """Select the highest-scoring grasp whose approach points down the z axis of the grasp frame."""
    down = [grasp for grasp in grasps.grasps if -grasp.transform[2, 2] >= TOP_DOWN_ALIGNMENT]
    return max(down, key=lambda grasp: grasp.score, default=None)
