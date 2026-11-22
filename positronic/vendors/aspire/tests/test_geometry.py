import numpy as np
import pytest

from positronic.vendors.aspire import geometry
from positronic.vendors.aspire.contracts import CameraObservation, Pose, Segmentation, SegmentationSet

CAMERA = 'wrist'
INTRINSICS = np.array([(10.0, 0.0, 3.0), (0.0, 10.0, 1.5), (0.0, 0.0, 1.0)])
# The camera is 0.6 m above the point (0.3, 0.2) of the base frame and looks down: its z is the base frame's -z, and
# its y is the base frame's -y.
LOOKS_DOWN = Pose([0.3, 0.2, 0.6], [0.0, 1.0, 0.0, 0.0], 'world')
# Depth in metres. A 2 x 3 object lies under the mask: its middle column is 0.1 m higher than its sides.
DEPTH_M = np.full((4, 6), 0.5, dtype=np.float32)
DEPTH_M[:, 3] = 0.4
DEPTH_M[0, 0] = 0.0
RGB = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
MASK = np.zeros((4, 6), dtype=np.bool_)
MASK[1:3, 2:5] = True
# The points of the mask in the base frame, row by row.
MASK_POINTS = [
    [0.25, 0.225, 0.1],
    [0.3, 0.22, 0.2],
    [0.35, 0.225, 0.1],
    [0.25, 0.175, 0.1],
    [0.3, 0.18, 0.2],
    [0.35, 0.175, 0.1],
]


def view(**fields) -> CameraObservation:
    return CameraObservation(
        **{'rgb': RGB, 'frame': CAMERA, 'depth_m': DEPTH_M, 'intrinsics': INTRINSICS, 'camera_pose': LOOKS_DOWN}
        | fields
    )


def segmentation(mask: np.ndarray = MASK, frame: str = CAMERA) -> Segmentation:
    return Segmentation(mask, 'cup', 0.9, CAMERA, frame)


def test_a_mask_becomes_points_in_the_base_frame_with_the_colors_of_its_pixels():
    cloud = geometry.mask_to_point_cloud(view(), segmentation(), target_frame='world')
    assert cloud.frame == 'world'
    np.testing.assert_allclose(cloud.points, MASK_POINTS, atol=1e-6)
    np.testing.assert_array_equal(cloud.colors, RGB[MASK])


@pytest.mark.parametrize('target_frame', [None, CAMERA])
def test_a_mask_becomes_points_in_the_camera_frame_with_no_target_frame_or_the_camera_frame(target_frame):
    cloud = geometry.mask_to_point_cloud(view(camera_pose=None), segmentation(), target_frame=target_frame)
    assert cloud.frame == CAMERA
    np.testing.assert_allclose(cloud.points[1], [0.0, -0.02, 0.4], atol=1e-6)


def test_a_pixel_with_no_depth_gives_no_point():
    mask = MASK.copy()
    mask[0, 0] = True
    cloud = geometry.mask_to_point_cloud(view(), segmentation(mask), target_frame='world')
    np.testing.assert_allclose(cloud.points, MASK_POINTS, atol=1e-6)


@pytest.mark.parametrize(
    ('observation', 'found', 'target_frame', 'message'),
    [
        (view(depth_m=None), segmentation(), 'world', 'no depth'),
        (view(intrinsics=None), segmentation(), 'world', 'no intrinsics'),
        (view(), segmentation(frame='exterior'), 'world', 'does not match camera frame'),
        (view(), segmentation(np.ones((2, 3), dtype=np.bool_)), 'world', 'shape'),
        (view(), segmentation(DEPTH_M == 0.0), 'world', 'no pixels with valid positive depth'),
        (view(camera_pose=None), segmentation(), 'world', "no pose in frame 'world'"),
        (view(), segmentation(), 'table', "no pose in frame 'table'"),
    ],
)
def test_points_need_a_view_with_depth_intrinsics_and_a_pose_in_the_target_frame(
    observation, found, target_frame, message
):
    with pytest.raises(ValueError, match=message):
        geometry.mask_to_point_cloud(observation, found, target_frame=target_frame)


def test_segmentations_are_binary_masks_that_share_one_view():
    with pytest.raises(ValueError, match='0/1'):
        segmentation(np.full((4, 6), 2))
    with pytest.raises(ValueError, match='share camera, frame, and mask shape'):
        SegmentationSet(ok=True, segmentations=[segmentation(), segmentation(frame='exterior')])
    with pytest.raises(ValueError, match='at least one segmentation'):
        SegmentationSet(ok=True)
