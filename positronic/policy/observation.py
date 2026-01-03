from functools import partial
from typing import Any

import numpy as np
from PIL import Image as PilImage

from positronic import geom
from positronic.dataset import Signal, transforms
from positronic.dataset import transforms as tf
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive

ROT_REP = geom.Rotation.Representation


class ObservationEncoder(Derive):
    def __init__(self, state: dict[str, list[str]], images: dict[str, tuple[str, tuple[int, int]]]):
        """
        Build an observation encoder.

        Args:
            state: mapping from output state key to an ordered list of episode keys to concatenate.
            images: mapping from output image name to tuple (input_key, (width, height)).
        """
        transforms = {k: partial(self.encode_state, k) for k in state.keys()}
        transforms.update({k: partial(self.encode_image, k) for k in images.keys()})
        super().__init__(**transforms)
        self._state = state
        self._image_configs = images
        self._metadata = {}

    @property
    def meta(self) -> dict[str, Any]:
        return self._metadata

    @meta.setter
    def meta(self, value: dict[str, Any]):
        self._metadata = value

    def encode_state(self, out_name: str, episode: Episode) -> Signal[Any]:
        state_features = self._state[out_name]
        return transforms.concat(*[episode[k] for k in state_features], dtype=np.float32)

    def encode_image(self, out_name: str, episode: Episode) -> Signal[Any]:
        input_key, (width, height) = self._image_configs[out_name]
        return image.resize_with_pad(width, height, signal=episode[input_key])

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Encode a single inference observation from raw images and input dict."""

        obs: dict[str, Any] = {}

        if 'task' in inputs:
            obs['task'] = inputs['task']

        # Encode images
        for out_name, (input_key, (width, height)) in self._image_configs.items():
            if input_key not in inputs:
                raise KeyError(f"Missing image input '{input_key}' for '{out_name}', available keys: {inputs.keys()}")
            frame = inputs[input_key]
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Image '{input_key}' must be HWC with 3 channels, got {frame.shape}")
            resized = image.resize_with_pad_per_frame(width, height, PilImage.Resampling.BILINEAR, frame)
            obs[out_name] = resized

        # Encode state vector
        for out_name, feature_names in self._state.items():
            parts: list[np.ndarray] = []
            for feature in feature_names:
                if feature not in inputs:
                    raise KeyError(f"Missing state input '{feature}' for '{out_name}', available keys: {inputs.keys()}")
                v = inputs[feature]
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                parts.append(arr)
            if parts:
                state_vec = np.concatenate(parts, axis=0)
            else:
                state_vec = np.empty((0,), dtype=np.float32)
            obs[out_name] = state_vec
        return obs


class GrootObservationEncoder(Derive):
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
            pose = geom.Transform3D.from_vector(pose, ROT_REP.QUAT).as_vector(self._rotation_rep)
        return pose

    def _encode_image(self, input_key: str, inputs: dict[str, Any]) -> np.ndarray:
        """Encode image from raw inputs."""
        frame = inputs[input_key]
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        w, h = self._image_size
        return image.resize_with_pad_per_frame(w, h, PilImage.Resampling.BILINEAR, frame)

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
