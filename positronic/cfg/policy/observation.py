from typing import Any

import configuronic as cfn

from positronic.policy.observation import ObservationEncoder


@cfn.config(image_size=(224, 224))
def eepose_grip(state_name: str, image_size: tuple[int, int], image_mappings: dict[str, str]):
    state = {state_name: ['robot_state.ee_pose', 'grip']}
    state_dim = 8
    images = {k: (v, image_size) for k, v in image_mappings.items()}
    result = ObservationEncoder(state=state, images=images)
    result.meta['gr00t_modality'] = {
        'state': {
            'robot_position_translation': {'start': 0, 'end': 3},
            'robot_position_quaternion': {'start': 3, 'end': 7, 'rotation_type': 'quaternion'},
            'grip': {'start': 7, 'end': 8},
        },
        'video': {  # TODO: Generalize this
            'ego_view': {'original_key': 'observation.images.left'},
            'side_image': {'original_key': 'observation.images.side'},
        },
    }
    lerobot_features = {k: {'shape': (state_dim,), 'names': v, 'dtype': 'float32'} for k, v in state.items()}
    for out_name, (_input_key, (width, height)) in result._image_configs.items():
        lerobot_features[out_name] = {
            'shape': (height, width, 3),
            'names': ['height', 'width', 'channel'],
            'dtype': 'video',
        }
    result.meta['lerobot_features'] = lerobot_features
    return result


@cfn.config(image_size=(224, 224))
def joints_grip(state_name: str, image_size: tuple[int, int], image_mappings: dict[str, str]):
    state = {state_name: ['robot_state.q', 'grip']}
    state_dim = 8
    images = {k: (v, image_size) for k, v in image_mappings.items()}
    result = ObservationEncoder(state=state, images=images)
    result.meta['gr00t_modality'] = {
        'state': {'robot_joints': {'start': 0, 'end': 7}, 'grip': {'start': 7, 'end': 8}},
        'video': {  # TODO: Generalize this
            'ego_view': {'original_key': 'observation.images.left'},
            'side_image': {'original_key': 'observation.images.side'},
        },
    }
    lerobot_features = {k: {'shape': (state_dim,), 'names': v, 'dtype': 'float32'} for k, v in state.items()}
    for out_name, (_input_key, (width, height)) in result._image_configs.items():
        lerobot_features[out_name] = {
            'shape': (height, width, 3),
            'names': ['height', 'width', 'channel'],
            'dtype': 'video',
        }
    result.meta['lerobot_features'] = lerobot_features
    return result


eepose_mujoco = eepose_grip.override(
    state_name='observation.state',
    image_mappings={'observation.images.left': 'image.handcam_left', 'observation.images.side': 'image.back_view'},
)
joints_mujoco = joints_grip.override(
    state_name='observation.state',
    image_mappings={'observation.images.left': 'image.handcam_left', 'observation.images.side': 'image.back_view'},
)


eepose_real = eepose_grip.override(
    state_name='observation.state',
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'},
)
openpi_positronic = eepose_grip.override(
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


@cfn.config()
def groot_ee_absolute():
    state_spec = {'observation.state': ['robot_state.ee_pose', 'grip']}
    result = ObservationEncoder(
        state=state_spec,
        images={
            'observation.images.exterior': ('image.exterior', (224, 224)),
            'observation.images.wrist': ('image.wrist', (224, 224)),
        },
    )
    result.meta['gr00t_modality'] = {
        'state': {
            'robot_position_translation': {'start': 0, 'end': 3},
            'robot_position_quaternion': {'start': 3, 'end': 7, 'rotation_type': 'quaternion'},
            'grip': {'start': 7, 'end': 8},
        },
        'video': {
            'exterior_image_1': {'original_key': 'observation.images.exterior'},
            'wrist_image': {'original_key': 'observation.images.wrist'},
        },
    }
    state_dim = 8
    lerobot_features = {k: {'shape': (state_dim,), 'names': v, 'dtype': 'float32'} for k, v in state_spec.items()}
    for out_name, (_input_key, (width, height)) in result._image_configs.items():
        lerobot_features[out_name] = {
            'shape': (height, width, 3),
            'names': ['height', 'width', 'channel'],
            'dtype': 'video',
        }
    result.meta['lerobot_features'] = lerobot_features
    return result


@cfn.config()
def groot_oxe_droid():
    state_spec = {'observation.state': ['robot_state.ee_pose', 'grip']}
    result = ObservationEncoder(
        state=state_spec,
        images={
            'observation.images.exterior': ('image.exterior', (224, 224)),
            'observation.images.wrist': ('image.wrist', (224, 224)),
        },
    )
    result.meta['gr00t_modality'] = {
        'state': {
            'eef_position': {'start': 0, 'end': 3},
            'eef_rotation': {'start': 3, 'end': 7, 'rotation_type': 'quaternion'},
            'gripper_position': {'start': 7, 'end': 8},
        },
        'video': {
            'exterior_image_1': {'original_key': 'observation.images.exterior'},
            'exterior_image_2': {'original_key': 'observation.images.exterior'},
            'wrist_image': {'original_key': 'observation.images.wrist'},
        },
    }
    state_dim = 8
    lerobot_features = {k: {'shape': (state_dim,), 'names': v, 'dtype': 'float32'} for k, v in state_spec.items()}
    for out_name, (_input_key, (width, height)) in result._image_configs.items():
        lerobot_features[out_name] = {
            'shape': (height, width, 3),
            'names': ['height', 'width', 'channel'],
            'dtype': 'video',
        }
    result.meta['lerobot_features'] = lerobot_features
    return result


@cfn.config()
def groot_infer():
    class GrootInferenceObservationEncoder(ObservationEncoder):
        def __init__(self):
            state = {'observation.state': ['robot_state.ee_pose', 'grip']}
            images = {
                'video.wrist_image': ('image.wrist', (224, 224)),
                'video.exterior_image_1': ('image.exterior', (224, 224)),
            }
            super().__init__(state, images)

        def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
            obs = super().encode(inputs)
            state = obs.pop('observation.state')
            obs['state.robot_position_translation'] = state[:3]
            obs['state.robot_position_quaternion'] = state[3:7]
            obs['state.grip'] = state[7:8]
            return obs

    return GrootInferenceObservationEncoder()
