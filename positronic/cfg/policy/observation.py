import configuronic as cfn

from positronic import geom
from positronic.policy.observation import ObservationEncoder


@cfn.config()
def general(
    state_name: str, state_features: dict[str, int], image_mappings: dict[str, str], image_size: tuple[int, int]
):
    """General observation encoder for non-GR00T policies (OpenPI, ACT, etc.)."""
    state_dict = {state_name: list(state_features.keys())}
    images = {k: (v, image_size) for k, v in image_mappings.items()}
    result = ObservationEncoder(state=state_dict, images=images)

    state_dim = sum(state_features.values())
    lerobot_features = {state_name: {'shape': (state_dim,), 'names': list(state_features.keys()), 'dtype': 'float32'}}
    for out_name, (_input_key, (width, height)) in result._image_configs.items():
        lerobot_features[out_name] = {
            'shape': (height, width, 3),
            'names': ['height', 'width', 'channel'],
            'dtype': 'video',
        }
    result.meta['lerobot_features'] = lerobot_features

    return result


eepose_grip = general.override(
    state_name='observation.state', state_features={'robot_state.ee_pose': 7, 'grip': 1}, image_size=(224, 224)
)

joints_grip = general.override(
    state_name='observation.state', state_features={'robot_state.q': 7, 'grip': 1}, image_size=(224, 224)
)

eepose_grip_joints = general.override(
    state_name='observation.state',
    state_features={'robot_state.ee_pose': 7, 'grip': 1, 'robot_state.q': 7},
    image_size=(224, 224),
)

eepose = eepose_grip.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)
joints = joints_grip.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)
eepose_q = eepose_grip_joints.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)

openpi_positronic = eepose_grip.override(
    state_name='observation/state',
    image_mappings={'observation/wrist_image': 'image.wrist', 'observation/image': 'image.exterior'},
)

openpi_eeq = eepose_grip_joints.override(
    state_name='observation/state',
    image_mappings={'observation/wrist_image': 'image.wrist', 'observation/image': 'image.exterior'},
)


@cfn.config(exterior_camera='image.exterior', wrist_camera='image.wrist', image_size=(224, 224))
def openpi_droid(exterior_camera: str, wrist_camera: str, image_size: tuple[int, int]):
    """DROID observation encoder using joint positions."""
    return ObservationEncoder(
        state={'observation/joint_position': ['robot_state.q'], 'observation/gripper_position': ['grip']},
        images={
            'observation/wrist_image_left': (wrist_camera, image_size),
            'observation/exterior_image_1_left': (exterior_camera, image_size),
        },
    )


# =============================================================================
# Unified GR00T observation encoder (works for both training and inference)
# =============================================================================


@cfn.config(rotation_rep=None, include_joints=False)
def groot(rotation_rep: str | None, include_joints: bool):
    """Unified GR00T N1.6 observation encoder for training and inference.

    Always outputs:
        - ee_pose: 7D (xyz+quat) by default, or converted if rotation_rep specified
        - grip: 1D gripper state
        - joint_position: 7D (if include_joints=True)
        - wrist_image, exterior_image_1: (224, 224, 3) images
    """
    from positronic.policy.observation import GrootObservationEncoder

    rot_rep = geom.Rotation.Representation(rotation_rep) if rotation_rep else None
    ee_dim = rot_rep.size + 3 if rot_rep else 7  # 3 for xyz translation
    encoder = GrootObservationEncoder(rotation_rep=rot_rep, include_joints=include_joints)

    # Set metadata for dataset generation
    # Each state key is stored as a separate column, so original_key points to the column
    # and start/end are indices within that column (not a packed observation.state)
    state_meta = {
        'ee_pose': {'start': 0, 'end': ee_dim, 'original_key': 'ee_pose'},
        'grip': {'start': 0, 'end': 1, 'original_key': 'grip'},
    }
    if include_joints:
        state_meta['joint_position'] = {'start': 0, 'end': 7, 'original_key': 'joint_position'}

    encoder.meta['gr00t_modality'] = {
        'state': state_meta,
        'video': {
            'exterior_image_1': {'original_key': 'exterior_image_1'},
            'wrist_image': {'original_key': 'wrist_image'},
        },
    }
    encoder.meta['lerobot_features'] = {
        'ee_pose': {'shape': (ee_dim,), 'dtype': 'float32'},
        'grip': {'shape': (1,), 'dtype': 'float32'},
        'wrist_image': {'shape': (224, 224, 3), 'dtype': 'video'},
        'exterior_image_1': {'shape': (224, 224, 3), 'dtype': 'video'},
    }
    if include_joints:
        encoder.meta['lerobot_features']['joint_position'] = {'shape': (7,), 'dtype': 'float32'}

    return encoder


groot_ee_absolute = groot.copy()
groot_rot6d = groot.override(rotation_rep='rot6d')
groot_joints = groot.override(include_joints=True)
groot_rot6d_joints = groot_rot6d.override(include_joints=True)
