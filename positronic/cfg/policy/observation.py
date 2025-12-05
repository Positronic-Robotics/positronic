import configuronic as cfn

from positronic.policy.observation import ObservationEncoder


@cfn.config()
def general(
    state_name: str,
    state_features: dict[str, int],
    image_mappings: dict[str, str],
    image_size: tuple[int, int],
    groot_state: dict[str, int | tuple[int, str]] | None,
    groot_video_mappings: dict[str, str] | None,
):
    groot_video_mappings = groot_video_mappings or {}
    state_dict = {state_name: list(state_features.keys())}

    images = {k: (v, image_size) for k, v in image_mappings.items()}
    result = ObservationEncoder(state=state_dict, images=images)
    result.meta.setdefault('gr00t_modality', {'state': {}, 'video': {}})

    state_dim = sum(state_features.values())
    groot_state = groot_state or state_features.copy()
    groot_dims = sum(v[0] if isinstance(v, tuple) else v for v in groot_state.values())
    assert groot_dims == state_dim, f'Groot state dimensions do not match state dimensions: {groot_dims} != {state_dim}'

    i = 0
    for k, v in groot_state.items():
        if isinstance(v, tuple):
            v, rotation_type = v
            result.meta['gr00t_modality']['state'][k] = {'start': i, 'end': i + v, 'rotation_type': rotation_type}
        else:
            result.meta['gr00t_modality']['state'][k] = {'start': i, 'end': i + v}
        i += v
    if groot_video_mappings:
        result.meta['gr00t_modality']['video'] = {k: {'original_key': v} for k, v in groot_video_mappings.items()}

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
    state_name='observation.state',
    state_features={'robot_state.ee_pose': 7, 'grip': 1},
    image_size=(224, 224),
    groot_state={'robot_position_translation': 3, 'robot_position_quaternion': (4, 'quaternion'), 'grip': 1},
    groot_video_mappings={'ego_view': 'observation.images.left', 'side_image': 'observation.images.side'},
)

joints_grip = general.override(
    state_name='observation.state',
    state_features={'robot_state.q': 7, 'grip': 1},
    image_size=(224, 224),
    groot_state={'robot_joints': 7, 'grip': 1},
    groot_video_mappings={'ego_view': 'observation.images.left', 'side_image': 'observation.images.side'},
)

eepose_grip_joints = general.override(
    state_name='observation.state',
    state_features={'robot_state.ee_pose': 7, 'grip': 1, 'robot_state.q': 7},
    image_size=(224, 224),
    groot_state={
        'robot_position_translation': 3,
        'robot_position_quaternion': (4, 'quaternion'),
        'grip': 1,
        'joint_position': 7,
    },
    groot_video_mappings={'ego_view': 'observation.images.left', 'side_image': 'observation.images.side'},
)


eepose_mujoco = eepose_grip.override(
    image_mappings={'observation.images.left': 'image.handcam_left', 'observation.images.side': 'image.back_view'}
)
joints_mujoco = joints_grip.override(
    image_mappings={'observation.images.left': 'image.handcam_left', 'observation.images.side': 'image.back_view'}
)
eepose_joints_mujoco = eepose_grip_joints.override(
    image_mappings={'observation.images.left': 'image.handcam_left', 'observation.images.side': 'image.back_view'}
)

eepose_real = eepose_grip.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)
joints_real = joints_grip.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)
eepose_q_real = eepose_grip_joints.override(
    image_mappings={'observation.images.left': 'image.wrist', 'observation.images.side': 'image.exterior'}
)

openpi_positronic = eepose_grip.override(
    state_name='observation/state',
    image_mappings={'observation/wrist_image': 'image.wrist', 'observation/image': 'image.exterior'},
    groot_state=None,
    groot_video_mappings=None,
)

openpi_eeq = eepose_grip_joints.override(
    state_name='observation/state',
    image_mappings={'observation/wrist_image': 'image.wrist', 'observation/image': 'image.exterior'},
    groot_state=None,
    groot_video_mappings=None,
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


groot_ee_absolute = eepose_grip.override(
    image_mappings={'observation.images.exterior': 'image.exterior', 'observation.images.wrist': 'image.wrist'},
    groot_video_mappings={'exterior_image_1': 'observation.images.exterior', 'wrist_image': 'observation.images.wrist'},
)
groot_ee_q = eepose_grip_joints.override(
    image_mappings={'observation.images.exterior': 'image.exterior', 'observation.images.wrist': 'image.wrist'},
    groot_video_mappings={'exterior_image_1': 'observation.images.exterior', 'wrist_image': 'observation.images.wrist'},
)
groot_oxe_droid = eepose_grip.override(
    image_mappings={'observation.images.exterior': 'image.exterior', 'observation.images.wrist': 'image.wrist'},
    groot_state={'eef_position': 3, 'eef_rotation': (4, 'quaternion'), 'gripper_position': 1},
    groot_video_mappings={
        'exterior_image_1': 'observation.images.exterior',
        'exterior_image_2': 'observation.images.exterior',
        'wrist_image': 'observation.images.wrist',
    },
)


@cfn.config()
def groot_infer():
    from positronic.policy.observation import GrootInferenceObservationEncoder

    return GrootInferenceObservationEncoder()


@cfn.config()
def groot_ee_q_infer():
    from positronic.policy.observation import GrootEE_QObservationEncoder

    return GrootEE_QObservationEncoder()
