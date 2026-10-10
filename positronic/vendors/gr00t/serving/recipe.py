"""GR00T client descriptions, importable without Positronic."""

from typing import Any

from positronic_model_server.spec import component, parallel, sequence

from . import EE_POSE, GRIP, JOINT_POSITION, LANGUAGE, STATE, STATE_DIMS, TASK, VIDEO
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
            component(
                'observation_codec',
                state=state,
                images={source: [source, settings[model_settings.IMAGE_SIZE]] for source in image_mappings.values()},
                task_field=observation_keys[TASK],
                task_source=observation_keys[TASK],
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
