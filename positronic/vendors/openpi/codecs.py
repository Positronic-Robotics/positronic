"""OpenPI codecs (observation encoder + action decoder pairs).

OpenPI has different key format expectations for training vs inference:

- **Training (LeRobot format)**: Dot-separated keys like `observation.state`, `observation.images.left`.
  This is what LeRobot datasets use and what OpenPI training code consumes.

- **Inference (OpenPI format)**: Slash-separated keys like `observation/state`, `observation/image`.
  This is what OpenPI's policy classes (e.g., `positronic_policy.py`) expect at inference time.

The `ObservationCodec` class handles both cases:
- `encode()` (inference): Produces OpenPI-compatible format with slash-separated keys
- `training_encoder` (training): Produces LeRobot-compatible format with dot-separated keys

Note: `droid` codec is inference-only, designed to work with pretrained DROID models.
"""

from functools import partial
from typing import Any

import configuronic as cfn
import numpy as np
from PIL import Image as PilImage

from positronic import geom, keys
from positronic.cfg import codecs
from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, Get
from positronic.drivers.roboarm import command
from positronic.policy.codec import Codec, lerobot_image, lerobot_state
from positronic.policy.observation import ObservationCodec as GenericObservationCodec


class ObservationCodec(Codec):
    """Observation encoder that outputs LeRobot keys for training, OpenPI keys for inference."""

    def __init__(
        self,
        state_features: dict[str, int],
        exterior_camera: str = keys.EXTERIOR_IMAGE,
        wrist_camera: str = keys.WRIST_IMAGE,
        image_size: tuple[int, int] = (224, 224),
    ):
        self._state_features = state_features
        self._exterior_camera = exterior_camera
        self._wrist_camera = wrist_camera
        self._image_size = image_size

        self._derive_transforms = {
            'observation.state': self._derive_state,
            'observation.images.left': partial(self._derive_image, wrist_camera),
            'observation.images.side': partial(self._derive_image, exterior_camera),
            'task': Get(keys.TASK, ''),
        }

        state_dim = sum(state_features.values())
        w, h = image_size
        self._training_meta: dict[str, Any] = {
            'lerobot_features': {
                'observation.state': lerobot_state(state_dim, list(state_features.keys())),
                'observation.images.left': lerobot_image(w, h),
                'observation.images.side': lerobot_image(w, h),
            }
        }

    def _derive_state(self, episode: Episode) -> Signal[Any]:
        return transforms.concat(*[episode[key] for key in self._state_features], dtype=np.float32)

    def _derive_image(self, input_key: str, episode: Episode) -> Signal[Any]:
        w, h = self._image_size
        return image.resize_with_pad(w, h, signal=episode[input_key])

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        return {}

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        state_parts: list[np.ndarray] = []
        for feature_key in self._state_features:
            if feature_key not in inputs:
                raise KeyError(f"Missing state input '{feature_key}', available keys: {list(inputs.keys())}")
            state_parts.append(np.asarray(inputs[feature_key], dtype=np.float32).reshape(-1))

        obs: dict[str, Any] = {
            'observation/state': np.concatenate(state_parts) if state_parts else np.empty((0,), dtype=np.float32),
            'observation/wrist_image': self._encode_image(self._wrist_camera, inputs),
            'observation/image': self._encode_image(self._exterior_camera, inputs),
        }
        if keys.TASK in inputs:
            obs['prompt'] = inputs[keys.TASK]
        return obs

    def _encode_image(self, input_key: str, inputs: dict[str, Any]) -> np.ndarray:
        if input_key not in inputs:
            raise KeyError(f"Missing image input '{input_key}', available keys: {list(inputs.keys())}")
        frame = inputs[input_key]
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        w, h = self._image_size
        return image.resize_with_pad_per_frame(w, h, PilImage.Resampling.BILINEAR, frame)

    def dummy_encoded(self, data=None) -> dict[str, Any]:
        """Return a zero-filled encoded observation in OpenPI's slash-separated format."""
        state_dim = sum(self._state_features.values())
        w, h = self._image_size
        return {
            'observation/state': np.zeros(state_dim, dtype=np.float32),
            'observation/wrist_image': np.zeros((h, w, 3), dtype=np.uint8),
            'observation/image': np.zeros((h, w, 3), dtype=np.uint8),
            'prompt': 'warmup',
        }

    @property
    def meta(self):
        return {'image_sizes': self._image_size}

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, **self._derive_transforms)


@cfn.config(
    state_features={keys.EE_POSE: 7, keys.GRIP: 1},
    exterior_camera=keys.EXTERIOR_IMAGE,
    wrist_camera=keys.WRIST_IMAGE,
    image_size=(224, 224),
)
def observation(state_features: dict[str, int], exterior_camera: str, wrist_camera: str, image_size: tuple[int, int]):
    """General OpenPI observation encoder with configurable state features."""
    return ObservationCodec(
        state_features=state_features, exterior_camera=exterior_camera, wrist_camera=wrist_camera, image_size=image_size
    )


ee_obs = observation
ee_joints_obs = observation.override(state_features={keys.EE_POSE: 7, keys.GRIP: 1, keys.JOINTS: 7})


# Pretrained DROID models read joints and gripper as separate observation keys and the language
# prompt under `prompt` (see openpi `droid_policy.DroidInputs`).
droid_obs = cfn.Config(
    GenericObservationCodec,
    state={'observation/joint_position': {keys.JOINTS: 7}, 'observation/gripper_position': {keys.GRIP: 1}},
    images={
        'observation/wrist_image_left': (keys.WRIST_IMAGE, (224, 224)),
        'observation/exterior_image_1_left': (keys.EXTERIOR_IMAGE, (224, 224)),
    },
    task_field='prompt',
)

ee = codecs.compose.override(obs=ee_obs, action=codecs.absolute_pos_action)
ee_joints = ee.override(obs=ee_joints_obs)

ee_traj = ee.override(action=codecs.traj_ee_action, binarize_grip=(keys.GRIP,))
ee_joints_traj = ee_joints.override(action=codecs.traj_ee_action, binarize_grip=(keys.GRIP,))

# Pure joint-based trajectory variant (no commanded joint targets in recordings)
joints_obs = observation.override(state_features={keys.JOINTS: 7, keys.GRIP: 1})
joints_traj = codecs.compose.override(
    obs=joints_obs,
    action=codecs.absolute_joints_action.override(tgt_joints_key=keys.JOINTS, tgt_grip_key=keys.GRIP),
    binarize_grip=(keys.GRIP,),
)

# IK variants: reconstruct joint targets from recorded EE targets via IK
joints_ik = codecs.compose.override(obs=joints_obs, action=codecs.ik_joints_action)
joints_ik_sim = joints_ik.override(**{'action.solver': 'lm'})

# DROID re-queries after its 8-step open-loop horizon; truncate the served chunk to match (8/15 s
# at 15 fps) so the client re-queries every 8 steps instead of playing the full chunk open-loop.
droid = codecs.compose.override(obs=droid_obs, action=codecs.joint_delta_action, horizon=8 / 15)

# The DROID jointpos models (openpi `*_droid_jointpos` configs — the RoboLab leaderboard policies): the
# server returns absolute joint-position chunks ``(action_horizon, 8)`` and RoboLab's client
# (``policies/pi0_family/client.py``) executes the whole chunk before re-querying, gripper binarized at
# 0.5 — its ``open_loop_horizon`` defaults equal each variant's ``action_horizon`` (pi05 = 15, pi0 = 10).
# No ``horizon`` here: the timestamp codec's validity sentinel closes the chunk, so re-inference lands
# after the full chunk executes, whatever each variant's length.
droid_jointpos = codecs.compose.override(
    obs=droid_obs,
    action=codecs.absolute_joints_action.override(tgt_joints_key=keys.JOINTS, tgt_grip_key=keys.GRIP),
    binarize_grip=(keys.GRIP,),
)


class PoseDeltaAction(Codec):
    """Decodes pi05_libero's OSC pose-delta chunk into per-step end-effector ``CartesianDelta`` (inference only).

    The policy emits a chunk of per-step actions ``[Δpos(3), Δrotvec(3), grip(1)]`` normalized to ``[-1, 1]`` in
    robosuite OSC_POSE space. Each step's pose delta is scaled by the controller's per-step output range
    (``OUTPUT_MAX``) and forwarded as a world-frame ``CartesianDelta`` the driver composes onto its live measured
    pose, reproducing robosuite's feed-forward OSC control (``goal = live_eef ∘ Δ`` every step). Integrating the
    chunk into absolute waypoints instead chases the open-loop trajectory and folds the OSC tracking residual back
    into each command — dynamics the policy was never trained against. The grip channel maps to the absolute
    ``[0, 1]`` closure the gripper command uses.
    """

    OUTPUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])

    def _decode_single(self, data, context=None):
        action = np.asarray(data['action']).clip(-1.0, 1.0)  # robosuite clips both pose and gripper to [-1, 1]
        physical = action[:6] * self.OUTPUT_MAX
        delta = geom.Transform3D(translation=physical[:3], rotation=geom.Rotation.from_rotvec(physical[3:6]))
        grip = (float(action[6]) + 1.0) / 2.0
        return {'robot_command': command.CartesianDelta(delta=delta), 'target_grip': grip}


# Constants pi05_libero's training distribution requires that the env's wire observation does not carry
# directly, measured once on a LIBERO box (`validate.py` drives the arm/gripper and prints these to bake):
#   - the env reports the eef orientation in the grip-site frame; openpi trained on robosuite's hand-body
#     `robot0_eef_quat`, a fixed 90°-about-z rotation away;
#   - the env collapses the two Panda finger qpos to a single closure scalar, reconstructed by interpolating
#     between the open (grip=0) and closed (grip=1) finger endpoints.
_GRIP_SITE_TO_HAND = geom.Rotation.from_quat([0.7071068, 0.0, 0.0, 0.7071068])
_GRIPPER_QPOS_OPEN = np.array([0.039683, -0.039682])
_GRIPPER_QPOS_CLOSED = np.array([0.0005, -0.0005])


class LiberoObservationCodec(Codec):
    """pi05_libero observation: 8-dim ``[eef_pos(3), eef_axisangle(3), gripper_qpos(2)]`` + 180°-rotated images.

    Reproduces openpi's LIBERO training preprocessing (openpi ``examples/libero/main.py``): the state packs the
    end-effector position, its orientation as an axis-angle vector in robosuite's hand-body frame, and the two
    Panda finger positions; both cameras are rotated 180°. The env reports the orientation in the grip-site frame
    and the gripper as a single closure scalar, so the codec rotates into the hand frame (``_GRIP_SITE_TO_HAND``)
    and reconstructs the finger positions. The adapter already flips images vertically, so the codec adds the
    horizontal flip to complete the 180° rotation. Inference only.
    """

    def __init__(
        self,
        exterior_camera: str = 'image.agentview',
        wrist_camera: str = keys.WRIST_IMAGE,
        image_size: tuple[int, int] = (224, 224),
    ):
        self._exterior_camera = exterior_camera
        self._wrist_camera = wrist_camera
        self._image_size = image_size

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        obs = {
            'observation/state': self._libero_state(inputs),
            'observation/wrist_image': self._encode_image(self._wrist_camera, inputs),
            'observation/image': self._encode_image(self._exterior_camera, inputs),
        }
        if keys.TASK in inputs:
            obs['prompt'] = inputs[keys.TASK]
        return obs

    def _libero_state(self, inputs: dict[str, Any]) -> np.ndarray:
        ee_pose = np.asarray(inputs[keys.EE_POSE], dtype=float)
        hand_rot = geom.Rotation.from_quat(ee_pose[3:7]) * _GRIP_SITE_TO_HAND
        # Reproduce robosuite's axis-angle branch. Its `robot0_eef_quat` is MuJoCo's `body_xquat`, FK-continuous
        # from the tool-down home pose and thus consistently in the w<=0 hemisphere (angle >= pi) across the
        # manipulation workspace — not the canonical w>=0 a fresh `mat2quat` would pick. Canonicalize to w<=0 to
        # match (`validate.py` asserts the per-pose axis-angle equals robosuite's; rerun it per suite to confirm).
        quat = np.asarray(hand_rot.to(geom.Rotation.Representation.QUAT))
        canonical = geom.Rotation.from_quat(quat if quat[0] <= 0 else -quat)
        axisangle = np.asarray(canonical.to(geom.Rotation.Representation.ROTVEC)).reshape(3)
        closure = 1.0 - float(inputs[keys.GRIP])
        gripper_qpos = _GRIPPER_QPOS_CLOSED + closure * (_GRIPPER_QPOS_OPEN - _GRIPPER_QPOS_CLOSED)
        return np.concatenate([ee_pose[:3], axisangle, gripper_qpos]).astype(np.float32)

    def _encode_image(self, input_key: str, inputs: dict[str, Any]) -> np.ndarray:
        frame = np.ascontiguousarray(np.asarray(inputs[input_key])[:, ::-1])
        w, h = self._image_size
        return image.resize_with_pad_per_frame(w, h, PilImage.Resampling.BILINEAR, frame)

    def dummy_encoded(self, data: dict | None = None) -> dict[str, Any]:
        w, h = self._image_size
        return {
            'observation/state': np.zeros(8, dtype=np.float32),
            'observation/wrist_image': np.zeros((h, w, 3), dtype=np.uint8),
            'observation/image': np.zeros((h, w, 3), dtype=np.uint8),
            'prompt': 'warmup',
        }

    @property
    def meta(self) -> dict[str, Any]:
        return {'image_sizes': self._image_size}


libero_obs = cfn.Config(LiberoObservationCodec)
libero_action = cfn.Config(PoseDeltaAction)

# pi05_libero emits a 10-step chunk; openpi's official LIBERO eval (`replan_steps=5`) executes the first 5 before
# re-querying. LIBERO's OSC runs at 20 Hz, so the chunk is stamped at 20 fps (one step per 0.05 s). Truncating at
# 0.25 s keeps the first five steps (timestamps < 0.25 s); the horizon sentinel marks 0.25 s as the chunk's
# validity end, so the fifth step runs its full period and re-inference lands right after it — five steps per
# query, matching replan_steps=5.
libero = codecs.compose.override(obs=libero_obs, action=libero_action, fps=20.0, horizon=0.25)
