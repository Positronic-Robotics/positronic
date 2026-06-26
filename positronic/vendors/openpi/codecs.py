"""OpenPI codecs (observation encoder + action decoder pairs).

OpenPI has different key format expectations for training vs inference:

- **Training (LeRobot format)**: Dot-separated keys like `observation.state`, `observation.images.left`.
  This is what LeRobot datasets use and what OpenPI training code consumes.

- **Inference (OpenPI format)**: Slash-separated keys like `observation/state`, `observation/image`.
  This is what OpenPI's policy classes (e.g., `positronic_policy.py`) expect at inference time.

The `OpenpiObservationCodec` class handles both cases:
- `encode()` (inference): Produces OpenPI-compatible format with slash-separated keys
- `training_encoder` (training): Produces LeRobot-compatible format with dot-separated keys

Note: `droid` codec is inference-only, designed to work with pretrained DROID models.
"""

from functools import partial
from typing import Any

import configuronic as cfn
import numpy as np
from PIL import Image as PilImage

from positronic import geom
from positronic.cfg import codecs
from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, Get
from positronic.drivers.roboarm import command
from positronic.policy.codec import Codec, lerobot_image, lerobot_state
from positronic.policy.observation import ObservationCodec


class OpenpiObservationCodec(Codec):
    """Observation encoder that outputs LeRobot keys for training, OpenPI keys for inference."""

    def __init__(
        self,
        state_features: dict[str, int],
        exterior_camera: str = 'image.exterior',
        wrist_camera: str = 'image.wrist',
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
            'task': Get('task', ''),
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
        if 'task' in inputs:
            obs['prompt'] = inputs['task']
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
    state_features={'robot_state.ee_pose': 7, 'grip': 1},
    exterior_camera='image.exterior',
    wrist_camera='image.wrist',
    image_size=(224, 224),
)
def observation(state_features: dict[str, int], exterior_camera: str, wrist_camera: str, image_size: tuple[int, int]):
    """General OpenPI observation encoder with configurable state features."""
    return OpenpiObservationCodec(
        state_features=state_features, exterior_camera=exterior_camera, wrist_camera=wrist_camera, image_size=image_size
    )


ee_obs = observation
ee_joints_obs = observation.override(state_features={'robot_state.ee_pose': 7, 'grip': 1, 'robot_state.q': 7})


# Pretrained DROID models read joints and gripper as separate observation keys and the language
# prompt under `prompt` (see openpi `droid_policy.DroidInputs`).
droid_obs = cfn.Config(
    ObservationCodec,
    state={'observation/joint_position': {'robot_state.q': 7}, 'observation/gripper_position': {'grip': 1}},
    images={
        'observation/wrist_image_left': ('image.wrist', (224, 224)),
        'observation/exterior_image_1_left': ('image.exterior', (224, 224)),
    },
    task_field='prompt',
)

ee = codecs.compose.override(obs=ee_obs, action=codecs.absolute_pos_action)
ee_joints = ee.override(obs=ee_joints_obs)

ee_traj = ee.override(action=codecs.traj_ee_action, binarize_grip=('grip',))
ee_joints_traj = ee_joints.override(action=codecs.traj_ee_action, binarize_grip=('grip',))

# Pure joint-based trajectory variant (no commanded joint targets in recordings)
joints_obs = observation.override(state_features={'robot_state.q': 7, 'grip': 1})
joints_traj = codecs.compose.override(
    obs=joints_obs,
    action=codecs.absolute_joints_action.override(tgt_joints_key='robot_state.q', tgt_grip_key='grip'),
    binarize_grip=('grip',),
)

# IK variants: reconstruct joint targets from recorded EE targets via IK
joints_ik = codecs.compose.override(obs=joints_obs, action=codecs.ik_joints_action)
joints_ik_sim = joints_ik.override(**{'action.solver': 'lm'})

# DROID re-queries after its 8-step open-loop horizon; truncate the served chunk to match (8/15 s
# at 15 fps) so the client re-queries every 8 steps instead of playing the full chunk open-loop.
droid = codecs.compose.override(obs=droid_obs, action=codecs.joint_delta_action, horizon=8 / 15)


class CumulativePoseDeltaAction(Codec):
    """Decodes pi05_libero's OSC pose-delta action chunk into absolute Cartesian waypoints (inference only).

    The policy emits a chunk of per-step actions ``[Δpos(3), Δrotvec(3), grip(1)]`` normalized to ``[-1, 1]``
    in robosuite OSC_POSE space. Each step's pose delta is scaled by the controller's per-step output range
    (``OUTPUT_MAX``) and integrated cumulatively onto the previous waypoint, anchored once on the observed
    end-effector pose, so the chunk plays open-loop as a trajectory of absolute poses — the LIBERO env
    recovers each per-step OSC delta from its own controller pose. The rotation delta left-multiplies the
    running orientation and the position delta adds in the world frame, matching robosuite ``set_goal``. The
    open-loop chunk assumes OSC tracking error stays within one step's output range; the short replan horizon
    bounds the accumulated divergence from the per-step control law.
    """

    OUTPUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])

    def __init__(self, robot_pose_key: str = 'robot_state.ee_pose'):
        self.robot_pose_key = robot_pose_key

    def decode(self, data, *, context=None):
        if not isinstance(data, list):
            raise ValueError('CumulativePoseDeltaAction decodes whole action chunks, not single actions')
        anchor = np.asarray(context[self.robot_pose_key], dtype=float)
        pos = anchor[:3].copy()
        rot = geom.Rotation.from_quat(anchor[3:7])
        decoded = []
        for step in data:
            action = np.asarray(step['action']).clip(-1.0, 1.0)  # robosuite clips both pose and gripper to [-1, 1]
            physical = action[:6] * self.OUTPUT_MAX
            pos = pos + physical[:3]
            rot = geom.Rotation.from_rotvec(physical[3:6]) * rot
            grip = (float(action[6]) + 1.0) / 2.0
            pose = geom.Transform3D(translation=pos, rotation=rot)
            decoded.append({'robot_command': command.CartesianPosition(pose=pose), 'target_grip': grip})
        return decoded


# Constants pi05_libero's training distribution requires that the env's wire observation does not carry
# directly, measured once on a LIBERO box (`validate.py` drives the arm/gripper and prints these to bake):
#   - the env reports the eef orientation in the grip-site frame; openpi trained on robosuite's hand-body
#     `robot0_eef_quat`, a fixed 90°-about-z rotation away;
#   - the env collapses the two Panda finger qpos to a single closure scalar, reconstructed by interpolating
#     between the open (grip=0) and closed (grip=1) finger endpoints.
_GRIP_SITE_TO_HAND = geom.Rotation.from_quat([0.7071068, 0.0, 0.0, 0.7071068])
_GRIPPER_QPOS_OPEN = np.array([0.039683, -0.039682])
_GRIPPER_QPOS_CLOSED = np.array([0.0005, -0.0005])


class OpenpiLiberoObservationCodec(Codec):
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
        wrist_camera: str = 'image.wrist',
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
        if 'task' in inputs:
            obs['prompt'] = inputs['task']
        return obs

    def _libero_state(self, inputs: dict[str, Any]) -> np.ndarray:
        ee_pose = np.asarray(inputs['robot_state.ee_pose'], dtype=float)
        hand_rot = geom.Rotation.from_quat(ee_pose[3:7]) * _GRIP_SITE_TO_HAND
        # Reproduce robosuite's axis-angle branch. Its `robot0_eef_quat` is MuJoCo's `body_xquat`, FK-continuous
        # from the tool-down home pose and thus consistently in the w<=0 hemisphere (angle >= pi) across the
        # manipulation workspace — not the canonical w>=0 a fresh `mat2quat` would pick. Canonicalize to w<=0 to
        # match (`validate.py` asserts the per-pose axis-angle equals robosuite's; rerun it per suite to confirm).
        quat = np.asarray(hand_rot.to(geom.Rotation.Representation.QUAT))
        canonical = geom.Rotation.from_quat(quat if quat[0] <= 0 else -quat)
        axisangle = np.asarray(canonical.to(geom.Rotation.Representation.ROTVEC)).reshape(3)
        closure = 1.0 - float(inputs['grip'])
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


libero_obs = cfn.Config(OpenpiLiberoObservationCodec)
libero_action = cfn.Config(CumulativePoseDeltaAction)

# pi05_libero emits a 10-step chunk, but openpi's official LIBERO eval (`replan_steps=5`) executes only the
# first 5 before re-querying; truncate to match. LIBERO's OSC runs at 20 Hz, so the chunk is stamped at
# 20 fps and the horizon keeps the first 5 steps (timestamps < 0.25 s).
libero = codecs.compose.override(obs=libero_obs, action=libero_action, fps=20.0, horizon=0.25)
