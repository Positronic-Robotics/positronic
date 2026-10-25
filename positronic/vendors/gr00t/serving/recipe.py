"""GR00T client descriptions, importable without Positronic."""

from copy import deepcopy
from typing import Any

from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY
from positronic_model_server.spec import component, parallel, sequence

from . import EE_POSE, GRIP, JOINT_POSITION, LANGUAGE, STATE, STATE_DIMS, TASK, VIDEO
from . import keys as obs_keys
from . import settings as model_settings


def inference(settings: dict[str, Any]) -> dict[str, Any]:
    """Describe the client codec using the checkpoint's image and pose conventions."""
    image_mappings = settings[model_settings.IMAGE_MAPPINGS]
    observation_keys = settings[model_settings.OBSERVATION_KEYS]
    state = {observation_keys[name]: {observation_keys[name]: size} for name, size in STATE_DIMS.items()}
    return sequence(
        component(
            'metadata',
            values={
                model_settings.ACTION_FPS: settings[model_settings.ACTION_FPS],
                model_settings.MODEL_SETTINGS: settings,
            },
        ),
        component('binarize_grip_inference'),
        component('change_ee_frame', transform=settings[model_settings.EE_FRAME], keys=[observation_keys[EE_POSE]]),
        component(
            'convert_pose',
            output_rotation='rot6d',
            rotation_offset=settings[model_settings.ROTATION_OFFSET],
            keys=[observation_keys[EE_POSE]],
        ),
        parallel(
            sequence(
                component('rename_observation_fields', mapping={observation_keys[TASK]: obs_keys.TASK}),
                parallel(
                    component('state_observation', state=state, task_field=observation_keys[TASK]),
                    sequence(
                        component(
                            'select_observation_fields', mapping={source: source for source in image_mappings.values()}
                        ),
                        component(
                            'transform_images',
                            width=settings[model_settings.IMAGE_SIZE][0],
                            height=settings[model_settings.IMAGE_SIZE][1],
                            mode='pad',
                            paths=[[source] for source in dict.fromkeys(image_mappings.values())],
                        ),
                        component(
                            'metadata',
                            values={'image_sizes': settings[model_settings.IMAGE_SIZE] if image_mappings else {}},
                        ),
                    ),
                ),
            ),
            sequence(
                component('set_control_mode', mode=settings[model_settings.CONTROL_MODE]),
                component(
                    'joint_position_action',
                    joints_key=JOINT_POSITION,
                    grip_key=GRIP,
                    num_joints=STATE_DIMS[JOINT_POSITION],
                ),
            ),
        ),
        component(
            'pack_observation_fields',
            layout={
                VIDEO: image_mappings,
                STATE: {name: observation_keys[name] for name in STATE_DIMS},
                LANGUAGE: {TASK: observation_keys[TASK]},
            },
            unsqueeze_dims=2,
        ),
        component('unpack_action_chunk', fields={name: [0, name] for name in (JOINT_POSITION, GRIP)}, squeeze_dims=1),
    )


def client_stack(
    settings: dict[str, Any],
    *,
    fps: float,
    horizon_sec: float | None,
    image_mappings: dict[str, str],
    compress_images: bool,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> dict[str, Any]:
    """Describe scheduling, observation preparation and native-result decoding for one session."""
    settings = deepcopy(settings)
    settings[model_settings.IMAGE_MAPPINGS] = image_mappings
    parts = [
        component('chunked_schedule', fps=fps, horizon_sec=horizon_sec),
        inference(settings),
        component('metadata', values={model_settings.ACTION_FPS: fps}),
    ]
    if compress_images:
        parts.append(component('encode_images', quality=jpeg_quality))
    return sequence(*parts)
