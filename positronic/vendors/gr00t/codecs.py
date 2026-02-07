"""GR00T codecs: implementation classes and configuronic configs in one file.

This module contains:
1. GrootObservationEncoder and GrootActionDecoder implementation classes
2. Configuronic configs for observation encoders and action decoders
3. Combined codec configs (observation + action pairs) for different GR00T modalities
"""

from functools import partial
from typing import Any

import configuronic as cfn
import numpy as np
from PIL import Image as PilImage

from positronic import geom
from positronic.cfg import codecs
from positronic.dataset import transforms
from positronic.dataset import transforms as tf
from positronic.dataset.episode import Episode
from positronic.dataset.signal import Signal
from positronic.dataset.transforms import image
from positronic.drivers.roboarm import command
from positronic.policy.action import ActionDecoder
from positronic.policy.observation import ObservationEncoder

RotRep = geom.Rotation.Representation
ROT_REP = geom.Rotation.Representation


# ===== Implementation Classes =====


class GrootObservationEncoder(ObservationEncoder):
    """Unified observation encoder for GR00T N1.6 training and inference.

    For training (__call__): outputs flat dict with separate keys for each state component.
    For inference (encode): outputs nested GR00T format for PolicyServer.

    State keys output:
        - ee_pose: 7D (xyz+quat) by default, or 9D (xyz+rot6d) if rotation_rep=ROT6D
        - grip: 1D gripper state
        - joint_position: 7D joint positions (if include_joints=True)

    Video keys output:
        - wrist_image: (H, W, 3) uint8
        - exterior_image_1: (H, W, 3) uint8

    Args:
        rotation_rep: Target rotation representation. None keeps original (QUAT).
        include_joints: If True, include joint_position in output.
        image_size: Output image size as (width, height).
        exterior_camera: Episode key for exterior camera.
        wrist_camera: Episode key for wrist camera.
    """

    def __init__(
        self,
        rotation_rep: ROT_REP | None = None,
        include_joints: bool = False,
        image_size: tuple[int, int] = (224, 224),
        exterior_camera: str = 'image.exterior',
        wrist_camera: str = 'image.wrist',
    ):
        self._rotation_rep = rotation_rep
        self._include_joints = include_joints
        self._image_size = image_size
        self._exterior_camera = exterior_camera
        self._wrist_camera = wrist_camera

        # Define transforms for training (derive from Episode)
        derive_transforms = {
            'ee_pose': self._derive_ee_pose,
            'grip': self._derive_grip,
            'wrist_image': partial(self._derive_image, wrist_camera),
            'exterior_image_1': partial(self._derive_image, exterior_camera),
        }
        if include_joints:
            derive_transforms['joint_position'] = self._derive_joints

        super().__init__(**derive_transforms)
        self._metadata: dict[str, Any] = {}

    @property
    def meta(self) -> dict[str, Any]:
        return self._metadata

    @meta.setter
    def meta(self, value: dict[str, Any]):
        self._metadata = value

    # --- Training transforms (Episode -> Signal) ---

    def _derive_ee_pose(self, episode: Episode) -> Signal[Any]:
        pose = episode['robot_state.ee_pose']  # 7D xyz+quat
        if self._rotation_rep is not None:
            pose = tf.recode_transform(ROT_REP.QUAT, self._rotation_rep, pose)
        return tf.astype(pose, np.float32)

    def _derive_grip(self, episode: Episode) -> Signal[Any]:
        # Reshape scalar grip values to (1,) arrays for LeRobot compatibility
        def reshape_to_1d(values):
            arr = np.asarray(values, dtype=np.float32)
            return arr.reshape(-1, 1)

        return transforms.Elementwise(episode['grip'], reshape_to_1d)

    def _derive_joints(self, episode: Episode) -> Signal[Any]:
        return tf.astype(episode['robot_state.q'], np.float32)

    def _derive_image(self, input_key: str, episode: Episode) -> Signal[Any]:
        w, h = self._image_size
        return image.resize_with_pad(w, h, signal=episode[input_key])

    # --- Inference encoding (raw inputs -> GR00T nested format) ---

    def _encode_ee_pose(self, inputs: dict[str, Any]) -> np.ndarray:
        """Encode ee_pose from raw robot inputs."""
        pose = np.asarray(inputs['robot_state.ee_pose'], dtype=np.float32).reshape(-1)
        if self._rotation_rep is not None:
            pose = geom.Transform3D.from_vector(pose, ROT_REP.QUAT).as_vector(self._rotation_rep).astype(np.float32)
        return pose

    def _encode_image(self, input_key: str, inputs: dict[str, Any]) -> np.ndarray:
        """Encode image from raw inputs."""
        frame = inputs[input_key]
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        w, h = self._image_size
        return image.resize_with_pad_per_frame(w, h, PilImage.Resampling.BILINEAR, frame)

    def dummy_input(self) -> dict[str, Any]:
        dummy: dict[str, Any] = {}
        dummy['robot_state.ee_pose'] = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)  # identity QUAT
        dummy['grip'] = np.zeros(1, dtype=np.float32)
        if self._include_joints:
            dummy['robot_state.q'] = np.zeros(7, dtype=np.float32)
        w, h = self._image_size
        dummy[self._wrist_camera] = np.zeros((h, w, 3), dtype=np.uint8)
        dummy[self._exterior_camera] = np.zeros((h, w, 3), dtype=np.uint8)
        return dummy

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Encode raw robot inputs into GR00T N1.6 nested format for inference.

        Args:
            inputs: Dict with keys: robot_state.ee_pose, grip, robot_state.q (optional),
                   image.wrist, image.exterior, task (optional).

        Returns:
            Nested dict with structure:
            {
                'video': {key: (1, 1, H, W, 3) uint8},
                'state': {key: (1, 1, D) float32},
                'language': {'annotation.language.language_instruction': [[str]]}
            }
        """
        # Encode state
        ee_pose = self._encode_ee_pose(inputs)
        grip = np.asarray(inputs['grip'], dtype=np.float32).reshape(-1)

        state_dict = {'ee_pose': ee_pose[np.newaxis, np.newaxis, ...], 'grip': grip[np.newaxis, np.newaxis, ...]}

        if self._include_joints:
            joints = np.asarray(inputs['robot_state.q'], dtype=np.float32).reshape(-1)
            state_dict['joint_position'] = joints[np.newaxis, np.newaxis, ...]

        return {
            'video': {
                'wrist_image': self._encode_image(self._wrist_camera, inputs)[np.newaxis, np.newaxis, ...],
                'exterior_image_1': self._encode_image(self._exterior_camera, inputs)[np.newaxis, np.newaxis, ...],
            },
            'state': state_dict,
            'language': {'annotation.language.language_instruction': [[inputs.get('task', '')]]},
        }


class GrootActionDecoder(ActionDecoder):
    """Unified GR00T action decoder for training and inference.

    For training (encode_episode): reads from episode and outputs action vector.
    For inference (decode): converts GR00T output to robot commands.

    Uses unified keys: ee_pose (7D or 9D) and grip (1D).

    Args:
        rotation_rep: Target rotation representation. None keeps QUAT (7D ee_pose).
        tgt_ee_pose_key: Episode key for target end-effector pose.
        tgt_grip_key: Episode key for target gripper position.
    """

    def __init__(
        self,
        rotation_rep: RotRep | None = None,
        tgt_ee_pose_key: str = 'robot_commands.pose',
        tgt_grip_key: str = 'target_grip',
    ):
        super().__init__()
        self._rotation_rep = rotation_rep if rotation_rep else RotRep.QUAT
        self._tgt_ee_pose_key = tgt_ee_pose_key
        self._tgt_grip_key = tgt_grip_key

    # --- Training: Episode -> action vector ---
    def encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        pose = episode[self._tgt_ee_pose_key]
        pose = transforms.recode_transform(RotRep.QUAT, self._rotation_rep, pose)
        return transforms.concat(pose, episode[self._tgt_grip_key], dtype=np.float32)

    # --- Inference: GR00T output -> robot command ---
    def decode(self, action: dict[str, Any], _inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        ee_pose = action['ee_pose']
        target_pose = geom.Transform3D.from_vector(ee_pose, self._rotation_rep)
        target_grip = action['grip'].item()
        return {
            'robot_command': command.to_wire(command.CartesianPosition(pose=target_pose)),
            'target_grip': target_grip,
        }


# ===== Configuronic Configs =====


@cfn.config(rotation_rep=None, include_joints=False)
def observation(rotation_rep: str | None, include_joints: bool):
    """GR00T N1.6 observation encoder.

    Outputs: ee_pose, grip, [joint_position], wrist_image, exterior_image_1
    Sets metadata: lerobot_features, gr00t_modality

    Args:
        rotation_rep: Rotation representation ('rot6d' or None for quaternion)
        include_joints: Whether to include joint_position feedback
    """
    rot_rep = geom.Rotation.Representation(rotation_rep) if rotation_rep else None
    ee_dim = rot_rep.size + 3 if rot_rep else 7
    encoder = GrootObservationEncoder(rotation_rep=rot_rep, include_joints=include_joints)

    # Set metadata for dataset generation
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


@cfn.config(rotation_rep=None, tgt_ee_pose_key='robot_commands.pose', tgt_grip_key='target_grip')
def action(rotation_rep: str | None, tgt_ee_pose_key: str, tgt_grip_key: str):
    """GR00T action decoder.

    Decodes from {'ee_pose': ..., 'grip': ...} format.
    Sets metadata: lerobot_features, gr00t_modality

    Args:
        rotation_rep: Rotation representation ('rot6d' or None for quaternion)
        tgt_ee_pose_key: Episode key for target pose
        tgt_grip_key: Episode key for target gripper position
    """
    rot_rep = RotRep(rotation_rep) if rotation_rep else None
    ee_dim = (rot_rep.size if rot_rep else 4) + 3

    result = GrootActionDecoder(rotation_rep=rot_rep, tgt_ee_pose_key=tgt_ee_pose_key, tgt_grip_key=tgt_grip_key)
    result.meta['gr00t_modality'] = {
        'action': {'ee_pose': {'start': 0, 'end': ee_dim}, 'grip': {'start': ee_dim, 'end': ee_dim + 1}}
    }
    result.meta['lerobot_features'] = {'action': {'shape': (ee_dim + 1,), 'names': ['actions'], 'dtype': 'float32'}}
    return result


# ===== Combined Codec Configs (observation + action pairs) =====


# GR00T codec variants using base codec config
ee_absolute = codecs.codec.override(
    observation=observation(rotation_rep=None, include_joints=False), action=action(rotation_rep=None)
)

ee_rot6d = codecs.codec.override(
    observation=observation(rotation_rep='rot6d', include_joints=False), action=action(rotation_rep='rot6d')
)

ee_joints = codecs.codec.override(
    observation=observation(rotation_rep=None, include_joints=True), action=action(rotation_rep=None)
)

ee_rot6d_joints = codecs.codec.override(
    observation=observation(rotation_rep='rot6d', include_joints=True), action=action(rotation_rep='rot6d')
)
