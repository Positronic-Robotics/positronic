"""Observation keys of the roboarena wire protocol.

The codec writes them when it encodes an observation and the source rebuilds them when it warms a freshly
loaded checkpoint, so they are named once here rather than spelled out at each end.
"""

JOINT_POSITION = 'observation/joint_position'
GRIPPER_POSITION = 'observation/gripper_position'
WRIST_IMAGE = 'observation/wrist_image_left'
PROMPT = 'prompt'
SESSION_ID = 'session_id'


def exterior_image(index: int) -> str:
    """The key of the ``index``-th exterior camera, counted from 0 as the server numbers them."""
    return f'observation/exterior_image_{index}_left'


# Fields of the ``PolicyServerConfig`` the server sends on connect, stating which of the keys above it wants.
# ``RESOLUTION`` is ``(height, width)``, the way the frame it asks for is shaped.
RESOLUTION = 'image_resolution'
NEEDS_WRIST_CAMERA = 'needs_wrist_camera'
NEEDS_STEREO_CAMERA = 'needs_stereo_camera'
NUM_EXTERIOR_CAMERAS = 'n_external_cameras'
