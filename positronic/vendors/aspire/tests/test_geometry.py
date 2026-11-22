import numpy as np
import pytest

from positronic.vendors.aspire import geometry
from positronic.vendors.aspire.contracts import (
    ApiError,
    CameraObservation,
    ErrorCode,
    PointCloud,
    Pose,
    Segmentation,
    SegmentationSet,
)

CAMERA = 'wrist'
# Focal lengths that differ, so a swap of x and y shows.
INTRINSICS = np.array([(10.0, 0.0, 3.0), (0.0, 20.0, 1.5), (0.0, 0.0, 1.0)])
# The camera is at (0.3, 0.2, 0.6) in the base frame, turned 90 degrees about x: its y is the base frame's z, and its
# z is the base frame's -y. The rotation is not its own inverse, so an inverse rotation shows.
TURNED = Pose([0.3, 0.2, 0.6], [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], 'world')
# Depth in metres. A 2 x 3 object is in the mask: its middle column is 0.1 m nearer than its sides.
DEPTH_M = np.full((4, 6), 0.5, dtype=np.float32)
DEPTH_M[:, 3] = 0.4
DEPTH_M[0, 0] = 0.0
DEPTH_M[0, 1] = np.inf
RGB = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
MASK = np.zeros((4, 6), dtype=np.bool_)
MASK[1:3, 2:5] = True
# The points of the mask in the base frame, row by row, computed by hand.
MASK_POINTS = [
    [0.25, -0.3, 0.5875],
    [0.3, -0.2, 0.59],
    [0.35, -0.3, 0.5875],
    [0.25, -0.3, 0.6125],
    [0.3, -0.2, 0.61],
    [0.35, -0.3, 0.6125],
]
# The same points in the camera frame.
MASK_POINTS_IN_CAMERA = [
    [-0.05, -0.0125, 0.5],
    [0.0, -0.01, 0.4],
    [0.05, -0.0125, 0.5],
    [-0.05, 0.0125, 0.5],
    [0.0, 0.01, 0.4],
    [0.05, 0.0125, 0.5],
]


def view(**fields) -> CameraObservation:
    return CameraObservation(
        **{'rgb': RGB, 'frame': CAMERA, 'depth_m': DEPTH_M, 'intrinsics': INTRINSICS, 'camera_pose': TURNED} | fields
    )


def segmentation(mask: np.ndarray = MASK, frame: str = CAMERA) -> Segmentation:
    return Segmentation(mask, 'cup', 0.9, CAMERA, frame)


def test_a_mask_becomes_points_in_the_base_frame_with_the_colors_of_its_pixels():
    cloud = geometry.mask_to_point_cloud(view(), segmentation(), target_frame='world')
    assert cloud.frame == 'world'
    np.testing.assert_allclose(cloud.points, MASK_POINTS, atol=1e-6)
    np.testing.assert_array_equal(cloud.colors, RGB[MASK])


@pytest.mark.parametrize(('camera_pose', 'target_frame'), [(TURNED, None), (None, None), (TURNED, CAMERA)])
def test_a_mask_becomes_points_in_the_camera_frame_with_no_target_frame_or_the_camera_frame(camera_pose, target_frame):
    cloud = geometry.mask_to_point_cloud(view(camera_pose=camera_pose), segmentation(), target_frame=target_frame)
    assert cloud.frame == CAMERA
    np.testing.assert_allclose(cloud.points, MASK_POINTS_IN_CAMERA, atol=1e-6)
    np.testing.assert_array_equal(cloud.colors, RGB[MASK])


def test_a_pixel_with_no_depth_or_an_infinite_depth_gives_no_point():
    mask = MASK.copy()
    mask[0, :2] = True
    cloud = geometry.mask_to_point_cloud(view(), segmentation(mask), target_frame='world')
    np.testing.assert_allclose(cloud.points, MASK_POINTS, atol=1e-6)


@pytest.mark.parametrize(
    ('observation', 'mask', 'target_frame', 'message'),
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
    observation, mask, target_frame, message
):
    with pytest.raises(ValueError, match=message):
        geometry.mask_to_point_cloud(observation, mask, target_frame=target_frame)


@pytest.mark.parametrize(
    ('make', 'message'),
    [
        (lambda: Segmentation(np.full((4, 6), 2), 'cup', 0.9, CAMERA, CAMERA), '0/1'),
        (lambda: Segmentation(np.zeros((0, 6)), 'cup', 0.9, CAMERA, CAMERA), 'non-empty shape'),
        (lambda: Segmentation(np.zeros(6), 'cup', 0.9, CAMERA, CAMERA), 'non-empty shape'),
        (lambda: Segmentation(MASK, 'cup', 1.5, CAMERA, CAMERA), 'score'),
        (lambda: Segmentation(MASK, 'cup', 0.9, CAMERA, CAMERA, [4, 1, 2, 2]), 'box_xyxy'),
        (lambda: Segmentation(MASK, ' ', 0.9, CAMERA, CAMERA), 'label'),
        (lambda: Segmentation(MASK, 'cup', 0.9, '', CAMERA), 'camera_name'),
        (lambda: Segmentation(MASK, 'cup', 0.9, CAMERA, ''), 'frame'),
        (lambda: SegmentationSet(ok=True), 'at least one segmentation'),
        (lambda: SegmentationSet(ok=True, segmentations=[segmentation(), segmentation(frame='exterior')]), 'share'),
        (
            lambda: SegmentationSet(ok=True, segmentations=[segmentation()], error=ApiError(ErrorCode.INTERNAL, 'x')),
            'error',
        ),
        (lambda: SegmentationSet(ok=False), 'error'),
        (
            lambda: SegmentationSet(ok=False, segmentations=[segmentation()], error=ApiError(ErrorCode.INTERNAL, 'x')),
            'cannot carry',
        ),
        (lambda: PointCloud(np.zeros((0, 3)), 'world'), 'at least one point'),
        (lambda: PointCloud(np.zeros((1, 3)), ''), 'frame'),
        (lambda: PointCloud(np.zeros((2, 3)), 'world', np.zeros((1, 3))), 'shape'),
        (lambda: PointCloud(np.zeros((1, 3)), 'world', [[300, 0, 0]]), r'\[0, 255\]'),
        (lambda: PointCloud(np.zeros((1, 3)), 'world', [[0.5, np.nan, 0.0]]), 'finite'),
        (lambda: PointCloud(np.zeros((1, 3)), 'world', [[1.5, 0.0, 0.0]]), r'\[0, 1.0\]'),
        (lambda: PointCloud(np.zeros((1, 3)), 'world', [[True, False, True]]), 'finite integers or floats'),
    ],
)
def test_segmentations_and_point_clouds_refuse_values_that_aspire_refuses(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_a_segmentation_set_gives_its_segmentations_only_when_it_succeeded():
    found = SegmentationSet(ok=True, segmentations=[segmentation()])
    assert found.value == (found.segmentations[0],) and not found.segmentations[0].mask.flags.writeable
    assert SegmentationSet(ok=False, error=ApiError(ErrorCode.NO_SEGMENTATION, 'none')).value is None
