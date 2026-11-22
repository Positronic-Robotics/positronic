"""Turn a calibrated camera view and a mask into points.

Ported from NVlabs/ASPIRE ``src/cap_harness/geometry.py`` at commit ``fa00f0e``, under the Apache License 2.0 (see
``NOTICE``). A failure raises ``ValueError``, as in ASPIRE.
"""

import numpy as np

from .contracts import CameraObservation, PointCloud, Segmentation


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
