import configuronic as cfn

import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.video_encoder
from positronic import keys
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm.settle import SettleTuning
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Observation
from positronic.eval import keys as eval_keys


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.franka_droid,
    gripper=positronic.cfg.hardware.gripper.robotiq,
    cameras=positronic.cfg.hardware.camera.droid,
)
def droid(robot_arm, gripper, cameras):
    """Real single-arm Franka (DROID) + Robotiq gripper + ZED cameras."""
    observations = {
        keys.ROBOT_STATE: Observation(robot_arm.state, Serializers.robot_state),
        keys.GRIP: Observation(gripper.grip, None),
        **{name: Observation(cam.frame, Serializers.camera_images) for name, cam in cameras.items()},
    }
    commands = {
        keys.ROBOT_COMMAND: Command(robot_arm.commands, Serializers.robot_command),
        keys.TARGET_GRIP: Command(gripper.target_grip, None),
    }
    return Embodiment(
        descriptor='',
        observations=observations,
        commands=commands,
        prepare_handlers={eval_keys.ARM: robot_arm.sync_move, eval_keys.GRIPPER: gripper.sync_move},
        static_meta=dict(ROBOT_STATIC_META),
        meta_source=robot_arm.robot_meta,
        control_systems=(*cameras.values(), robot_arm, gripper),
        simulated=False,
    )


droid_3cam = droid.override(cameras=positronic.cfg.hardware.camera.droid_3cam)


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.yam, cameras={}, video_encoder=positronic.cfg.video_encoder.jetson_h264
)
def yam(robot_arm, cameras, video_encoder):
    """Real single-arm i2rt YAM: the arm driver carries the gripper (they share one CAN chain)."""
    observations = {
        keys.ROBOT_STATE: Observation(robot_arm.state, Serializers.robot_state),
        keys.GRIP: Observation(robot_arm.grip, None),
        **{name: Observation(cam.frame, Serializers.camera_images) for name, cam in cameras.items()},
    }
    commands = {
        keys.ROBOT_COMMAND: Command(robot_arm.commands, Serializers.robot_command),
        keys.TARGET_GRIP: Command(robot_arm.target_grip, None),
    }
    return Embodiment(
        descriptor='yam',
        observations=observations,
        commands=commands,
        # One driver, one handler: the YAM chain carries its own fingers
        prepare_handlers={eval_keys.ARM: robot_arm.sync_move},
        static_meta=dict(ROBOT_STATIC_META),
        meta_source=robot_arm.robot_meta,
        control_systems=(*cameras.values(), robot_arm),
        simulated=False,
        video_encoder=video_encoder,
    )


@cfn.config(
    left_channel='can0',
    right_channel='can1',
    park_after_idle_s=60.0,
    park_tuning={
        'left': positronic.cfg.hardware.roboarm.yam_park_tuning,
        'right': positronic.cfg.hardware.roboarm.yam_park_tuning,
    },
    move_tuning={
        'left': positronic.cfg.hardware.roboarm.yam_move_tuning,
        'right': positronic.cfg.hardware.roboarm.yam_move_tuning,
    },
    # World-frame arm-base mount positions of the sim scene the training data uses: tabletop z=0.30 plus the
    # 0.011 base plate, arms at (0.30, ±0.305) facing +x.
    mounts={'left': [0.30, 0.305, 0.311], 'right': [0.30, -0.305, 0.311]},
    gravity_comp_factor=None,
    cameras={
        keys.EXTERIOR_IMAGE: positronic.cfg.hardware.camera.zed_x_top.override(resolution='svga', fps=30),
        'image.wrist_left': positronic.cfg.hardware.camera.zed_x_one_left.override(resolution='svga', fps=30),
        'image.wrist_right': positronic.cfg.hardware.camera.zed_x_one_right.override(resolution='svga', fps=30),
    },
    video_encoder=positronic.cfg.video_encoder.jetson_h264,
)
def yam_bimanual(
    left_channel: str,
    right_channel: str,
    mounts: dict[str, list[float]],
    gravity_comp_factor: list[float] | None,
    cameras,
    park_after_idle_s: float | None,
    park_tuning: dict[str, SettleTuning],
    move_tuning: dict[str, SettleTuning],
    video_encoder,
):
    """Real bimanual i2rt YAM on two CAN chains.

    Per-arm channels are the flat names the whole stack shares: ``robot_state.{side}`` expands into
    ``robot_state.{side}.q/.dq/.ee_pose`` on record and commands are ``robot_command.{side}`` +
    ``target_grip.{side}``. Each arm is mounted at ``mounts[side]``, so real ``ee_pose`` lands in the world
    frame the training data uses; static_meta records the mount of every arm built, keyed by the joint
    signal that drives it.
    """
    from positronic import geom
    from positronic.drivers.roboarm import yam as yam_driver

    arms = {
        side: yam_driver.Robot(
            channel,
            base_pose=geom.Transform3D(mounts[side]),
            park_after_idle_s=park_after_idle_s,
            park_tuning=park_tuning[side],
            move_tuning=move_tuning[side],
            gravity_comp_factor=gravity_comp_factor,
        )
        for side, channel in (('left', left_channel), ('right', right_channel))
    }
    observations = {
        **{f'{keys.ROBOT_STATE}.{s}': Observation(arm.state, Serializers.robot_state) for s, arm in arms.items()},
        **{f'{keys.GRIP}.{s}': Observation(arm.grip, None) for s, arm in arms.items()},
        **{name: Observation(cam.frame, Serializers.camera_images) for name, cam in cameras.items()},
    }
    commands = {
        **{f'{keys.ROBOT_COMMAND}.{s}': Command(arm.commands, Serializers.robot_command) for s, arm in arms.items()},
        **{f'{keys.TARGET_GRIP}.{s}': Command(arm.target_grip, None) for s, arm in arms.items()},
    }
    joint_signals = {side: f'{keys.ROBOT_STATE}.{side}.q' for side in arms}
    static_meta = {
        eval_keys.JOINT_SIGNALS: list(joint_signals.values()),
        eval_keys.POSE_SIGNALS: [f'{keys.ROBOT_STATE}.{s}.ee_pose' for s in arms]
        + [f'{keys.ROBOT_COMMAND}.{s}.pose' for s in arms],
        eval_keys.MOUNTS: {sig: mounts[side] for side, sig in joint_signals.items()},
    }
    return Embodiment(
        descriptor='yam_bimanual',
        observations=observations,
        commands=commands,
        prepare_handlers={f'{eval_keys.ARM}.{s}': arm.sync_move for s, arm in arms.items()},
        static_meta=static_meta,
        # Both drivers emit the identical per-arm meta; record one copy.
        meta_source=arms['left'].robot_meta,
        control_systems=(*cameras.values(), *arms.values()),
        simulated=False,
        video_encoder=video_encoder,
    )


# The yambox station. Its CAN chains carry the names udev gives the two adapters, because `can0` and `can1`
# there are the onboard controllers and reach no arm.
# TODO: `mounts` still holds the sim-scene values the default carries. Survey where this station's arm bases
# sit and put the measured pose here, or `ee_pose` reaches the world frame wrong.
yam_bimanual_yambox = yam_bimanual.override(
    left_channel='can_follower_l',
    right_channel='can_follower_r',
    # Measured on this station: under i2rt's own factors joints 3 and 4 hold 29 and 32 mrad below where they
    # are sent, which is past the driver's 20 mrad arrival tolerance, so every park reports ERROR. These park
    # both arms with about 10 mrad to spare. Joint 4 is the sensitive one — its zero sits near 1.37.
    gravity_comp_factor=[1.0, 1.1, 1.4, 1.4, 1.0, 1.0],
    cameras={
        keys.EXTERIOR_IMAGE: positronic.cfg.hardware.camera.yambox_zed_x_top.override(resolution='svga', fps=30),
        'image.wrist_left': positronic.cfg.hardware.camera.yambox_zed_x_one_left.override(resolution='svga', fps=30),
        'image.wrist_right': positronic.cfg.hardware.camera.yambox_zed_x_one_right.override(resolution='svga', fps=30),
    },
)


def mujoco_franka(sim, camera_dict):
    """Mujoco single-arm Franka + gripper over a given sim.

    Maps the sim's arm, gripper and camera ports into an embodiment, and its scene draw into the prepare a
    trial opens on. Which scene it draws (the loaders) and the privileged ground-truth are the eval's
    concern. 3 cameras because Mujoco does not render the second image when using only 2 cameras.
    """
    observations = {
        keys.ROBOT_STATE: Observation(sim.state, Serializers.robot_state),
        keys.GRIP: Observation(sim.grip, None),
        **{name: Observation(sim.cameras[orig], Serializers.camera_images) for name, orig in camera_dict.items()},
    }
    commands = {
        keys.ROBOT_COMMAND: Command(sim.commands, Serializers.robot_command),
        keys.TARGET_GRIP: Command(sim.target_grip, None),
    }
    return Embodiment(
        descriptor='mujoco.franka',
        observations=observations,
        commands=commands,
        # Two handlers on one sim: the draw places the objects and leaves the arm at the pose the scene
        # itself starts from; the move is what a trial names to put the arm somewhere else.
        prepare_handlers={eval_keys.SCENE: sim.env_reset, eval_keys.ARM: sim.sync_move},
        static_meta={**ROBOT_STATIC_META, 'simulation.mujoco_model_path': sim.mujoco_model_path},
        meta_source=sim.robot_meta,
        control_systems=(sim,),
        simulated=True,
    )
