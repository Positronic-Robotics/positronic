"""FLUX 3 Action's DROID codecs and scheduling, described for the client."""

from positronic_model_server.spec import component, parallel, sequence

from . import keys

# rules-allow: hardcoded-keys — standalone wire form, pinned to DROID_IMPEDANCE by test_droid_control_mode.
DROID_CONTROL_MODE = {
    'type': 'impedance',
    'kq': [40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0],
    'kqd': [4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0],
    'kx': [400.0, 400.0, 400.0, 15.0, 15.0, 15.0],
    'kxd': [37.0, 37.0, 37.0, 2.0, 2.0, 2.0],
    'frame': [0.0, 0.0, -0.155, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
}


# The flange seen from the `gripper` site, as ChangeEEFrame's wire transform [tx, ty, tz, qw, qx, qy, qz].
# EE-50 poses are DROID flange poses; the arms report the `gripper` site.
DROID_FLANGE_FROM_GRIPPER = [0.0, 0.0, -0.155, 1.0, 0.0, 0.0, 0.0]
GRIPPER_SITE = 'gripper'


def droid_cameras(*, three_cameras: bool = False) -> dict[str, str]:
    return {
        keys.WRIST: keys.WRIST_IMAGE,
        keys.EXTERIOR: keys.EXTERIOR_IMAGE,
        keys.EXTERIOR_2: keys.EXTERIOR_IMAGE_2 if three_cameras else keys.EXTERIOR_IMAGE,
    }


def _ee_frame() -> dict:
    return component(
        'change_ee_frame',
        site=GRIPPER_SITE,
        transform=DROID_FLANGE_FROM_GRIPPER,
        keys=[keys.EE_POSE, keys.TARGET_EE_POSE, keys.ROBOT_COMMAND],
    )


def inference(image_mappings: dict[str, str], *, rtc: bool = False, ee: bool = False) -> dict:
    """The client codecs. ``ee``: an EE-50 checkpoint reads the flange pose and returns absolute flange targets."""
    if ee and rtc:
        raise ValueError('EE-50 checkpoints serve without RTC')
    action_args = {'commands_path': [keys.RTC.ACTION_PREFIX, keys.RTC.PREFIX_ACTIONS]} if rtc else {}
    context = [component('select_observation_fields', mapping={keys.CALL_TIMINGS: keys.CALL_TIMINGS})] if rtc else []
    pose = {keys.EE_POSE_FIELD: {keys.EE_POSE: 7}} if ee else {keys.JOINT_POSITION: {keys.JOINTS: 7}}
    action = (
        component('absolute_position_action', tgt_ee_pose_key=keys.TARGET_EE_POSE, tgt_grip_key=keys.GRIP)
        if ee
        else component('absolute_joints_action', tgt_joints_key=keys.JOINTS, tgt_grip_key=keys.GRIP, **action_args)
    )
    encoded = parallel(
        component('state_observation', state={**pose, keys.GRIPPER_POSITION: {keys.GRIP: 1}}),
        sequence(
            component('select_observation_fields', mapping={**image_mappings, keys.PROMPT: keys.TASK}),
            component(
                'transform_images',
                width=keys.VIEW_SIZE[0],
                height=keys.VIEW_SIZE[1],
                mode='pad',
                paths=[[name] for name in keys.CAMERAS],
            ),
        ),
        sequence(component('set_control_mode', mode=DROID_CONTROL_MODE), action),
        *context,
    )
    if ee:
        return sequence(_ee_frame(), encoded, component('split_action_chunk'))
    if not rtc:
        return sequence(encoded, component('split_action_chunk'))
    return sequence(
        encoded,
        component('split_action_chunk'),
        component(
            'pack_fields',
            encode_layout={
                **{name: [name] for name in (*keys.CAMERAS, keys.JOINT_POSITION, keys.GRIPPER_POSITION, keys.PROMPT)},
                keys.RTC.FIELD: {
                    keys.RTC.ACTION_PREFIX: [keys.RTC.ACTION_PREFIX],
                    keys.CALL_TIMINGS: [keys.CALL_TIMINGS],
                },
            },
        ),
    )


def client_stack(
    *,
    fps: float,
    horizon_sec: float | None,
    image_mappings: dict[str, str],
    compress_images: bool,
    jpeg_quality: int,
    rtc: dict[str, float] | None,
    ee: bool = False,
) -> dict:
    player = {'replan_on_error': True}
    request = {'reference': 'before_end'}
    merge = {'start': 'after_previous', 'horizon_sec': horizon_sec}
    if rtc is not None:
        player = {'plan_key': keys.PLAN, 'record_stats': False}
        request = {'reference': 'after_start', 'seconds': rtc[keys.RTC.CALL_AFTER_SEC]}
        merge = {'start': 'observation'}
    parts = [
        component('trajectory_player', **player),
        component('inference_schedule', fps=fps, request_timing=request, merge=merge, latency_estimate=0.0),
        inference(image_mappings, rtc=rtc is not None, ee=ee),
        component('metadata', values={keys.ACTION_FPS: fps}),
    ]
    if compress_images:
        parts.append(component('encode_images', quality=jpeg_quality))
    return sequence(*parts)
