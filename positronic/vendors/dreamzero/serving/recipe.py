"""DreamZero's client-side sampling, scheduling and codecs."""

from enum import Enum

from positronic_model_server.spec import component, parallel, sequence

from .keys import (
    ACTION_GRIP,
    ACTION_JOINTS,
    CAMERAS,
    EXTERIOR,
    EXTERIOR_2,
    EXTERIOR_IMAGE,
    EXTERIOR_IMAGE_2,
    GRIP,
    JOINTS,
    OBS_GRIP,
    OBS_JOINTS,
    OBS_TIME_NS,
    PROMPT,
    TASK,
    WRIST,
    WRIST_IMAGE,
)


class Mode(Enum):
    CHUNKED = 'chunked'
    RTC = 'rtc'


# rules-allow: hardcoded-keys — standalone wire form, pinned to DROID_IMPEDANCE by test_droid_control_mode.
DROID_CONTROL_MODE = {
    'type': 'impedance',
    'kq': [40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0],
    'kqd': [4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0],
    'kx': [400.0, 400.0, 400.0, 15.0, 15.0, 15.0],
    'kxd': [37.0, 37.0, 37.0, 2.0, 2.0, 2.0],
    'frame': [0.0, 0.0, -0.155, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
}


def inference(image_mappings: dict[str, str], width: int, height: int, jpeg_quality: int) -> dict:
    """Encode one observation; decode the native time-major action arrays."""
    return sequence(
        parallel(
            sequence(
                component('state_observation', state={JOINTS: {OBS_JOINTS: 7}, GRIP: {OBS_GRIP: 1}}),
                component('pack_fields', encode_layout={JOINTS: [JOINTS], GRIP: [GRIP]}, unsqueeze_dims=1),
            ),
            component('select_observation_fields', mapping={**image_mappings, PROMPT: TASK, OBS_TIME_NS: OBS_TIME_NS}),
            sequence(
                component('set_control_mode', mode=DROID_CONTROL_MODE),
                component('joint_position_action', joints_key=ACTION_JOINTS, grip_key=ACTION_GRIP),
            ),
        ),
        component('split_action_chunk'),
        component('transform_images', width=width, height=height, mode='pad', paths=[[name] for name in CAMERAS]),
        component('encode_images', quality=jpeg_quality),
    )


def client_stack(
    mode: Mode,
    *,
    fps: float,
    sampling_period_sec: float,
    prediction_period_sec: float,
    image_mappings: dict[str, str],
    width: int,
    height: int,
    jpeg_quality: int,
) -> dict:
    """Both modes sample periodically; RTC-like playback starts each continuation at its observation."""
    return sequence(
        component('trajectory_player', replan_on_error=False),
        component('periodic', fps=1 / sampling_period_sec),
        component(
            'inference_schedule',
            fps=fps,
            request_timing={'reference': 'every', 'seconds': sampling_period_sec},
            merge={
                'start': 'after_previous' if mode is Mode.CHUNKED else 'observation',
                'horizon_sec': prediction_period_sec if mode is Mode.CHUNKED else None,
            },
            latency_estimate=0.0,
        ),
        inference(image_mappings, width, height, jpeg_quality),
    )


def droid_cameras() -> dict[str, str]:
    return {EXTERIOR: EXTERIOR_IMAGE, EXTERIOR_2: EXTERIOR_IMAGE_2, WRIST: WRIST_IMAGE}
