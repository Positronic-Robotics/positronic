"""OpenPI codecs (observation encoder + action decoder pairs)."""

import configuronic as cfn

from positronic.cfg import codecs
from positronic.policy.observation import ObservationEncoder

# ===== OpenPI-specific Observation Configs =====

openpi_positronic_obs = codecs.eepose_grip.override(
    state_name='observation/state',
    image_mappings={'observation/wrist_image': 'image.wrist', 'observation/image': 'image.exterior'},
    task_field='prompt',  # OpenPI expects 'prompt' field
)

openpi_eeq_obs = codecs.eepose_grip_joints.override(
    state_name='observation/state',
    image_mappings={'observation/wrist_image': 'image.wrist', 'observation/image': 'image.exterior'},
    task_field='prompt',  # OpenPI expects 'prompt' field
)


@cfn.config(exterior_camera='image.exterior', wrist_camera='image.wrist', image_size=(224, 224))
def openpi_droid_obs(exterior_camera: str, wrist_camera: str, image_size: tuple[int, int]):
    """DROID observation encoder using joint positions."""
    return ObservationEncoder(
        state={'observation/joint_position': ['robot_state.q'], 'observation/gripper_position': ['grip']},
        images={
            'observation/wrist_image_left': (wrist_camera, image_size),
            'observation/exterior_image_1_left': (exterior_camera, image_size),
        },
        task_field='prompt',  # OpenPI expects 'prompt' field
    )


# ===== Combined Codec Configs (observation + action pairs) =====

eepose_absolute = codecs.codec.override(observation=codecs.eepose, action=codecs.absolute_position(rotation_rep=None))

openpi_positronic = codecs.codec.override(
    observation=openpi_positronic_obs, action=codecs.absolute_position(rotation_rep=None)
)

droid = codecs.codec.override(observation=openpi_droid_obs, action=codecs.joint_delta(num_joints=7))

eepose_q = codecs.codec.override(observation=codecs.eepose_q, action=codecs.absolute_position(rotation_rep=None))

joints = codecs.codec.override(observation=codecs.joints, action=codecs.absolute_position(rotation_rep=None))
