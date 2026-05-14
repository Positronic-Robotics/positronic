"""DreamZero codecs: observation encoder + action decoder for roboarena format."""

from functools import partial
from typing import Any

import configuronic as cfn
import numpy as np
from PIL import Image as PilImage

from positronic.cfg import codecs
from positronic.dataset import Signal, transforms
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, Get
from positronic.drivers.roboarm import command
from positronic.policy.codec import Codec, lerobot_action, lerobot_image, lerobot_state

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 176


def _reshape_grip(values):
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


class DreamZeroObservationCodec(Codec):
    """Observation encoder for DreamZero's roboarena inference format.

    For training (training_encoder): outputs flat LeRobot keys (state.joint_position,
    state.gripper_position, video columns).
    For inference (encode): outputs roboarena keys (observation/joint_position, etc.)
    with images resized to 320x176 and session_id for stateful frame tracking.
    """

    def __init__(
        self,
        wrist_camera: str = 'image.wrist',
        exterior_camera_1: str = 'image.exterior',
        exterior_camera_2: str | None = None,
        image_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
    ):
        self._wrist_camera = wrist_camera
        self._exterior_camera_1 = exterior_camera_1
        self._exterior_camera_2 = exterior_camera_2 if exterior_camera_2 is not None else exterior_camera_1
        self._image_size = image_size

        w, h = image_size
        self._derive_transforms = {
            'state.joint_position': self._derive_joint_position,
            'state.gripper_position': self._derive_gripper_position,
            'video.wrist_image_left': partial(self._derive_image, wrist_camera),
            'video.exterior_image_1_left': partial(self._derive_image, exterior_camera_1),
            'video.exterior_image_2_left': partial(self._derive_image, self._exterior_camera_2),
            'task': Get('task', ''),
        }

        self._training_meta = {
            'lerobot_features': {
                'state.joint_position': lerobot_state(7),
                'state.gripper_position': lerobot_state(1),
                'video.wrist_image_left': lerobot_image(w, h),
                'video.exterior_image_1_left': lerobot_image(w, h),
                'video.exterior_image_2_left': lerobot_image(w, h),
            },
            'gr00t_modality': {
                'state': {
                    'joint_position': {'start': 0, 'end': 7, 'original_key': 'state.joint_position'},
                    'gripper_position': {'start': 0, 'end': 1, 'original_key': 'state.gripper_position'},
                },
                'video': {
                    'wrist_image_left': {'original_key': 'video.wrist_image_left'},
                    'exterior_image_1_left': {'original_key': 'video.exterior_image_1_left'},
                    'exterior_image_2_left': {'original_key': 'video.exterior_image_2_left'},
                },
                'annotation': {
                    'language.language_instruction': {'original_key': 'task_index'},
                    'language.language_instruction_2': {'original_key': 'task_index'},
                    'language.language_instruction_3': {'original_key': 'task_index'},
                },
            },
        }

    def _derive_joint_position(self, episode: Episode) -> Signal[Any]:
        return transforms.astype(episode['robot_state.q'], np.float32)

    def _derive_gripper_position(self, episode: Episode) -> Signal[Any]:
        return transforms.Elementwise(episode['grip'], _reshape_grip)

    def _derive_image(self, input_key: str, episode: Episode) -> Signal[Any]:
        w, h = self._image_size
        return image.resize_with_pad(w, h, signal=episode[input_key])

    def _encode_image(self, input_key: str, inputs: dict[str, Any]) -> np.ndarray:
        w, h = self._image_size
        frame = inputs.get(input_key)
        if frame is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        return image.resize_with_pad_per_frame(w, h, PilImage.Resampling.BILINEAR, np.asarray(frame))

    def encode(self, inputs: dict[str, Any]) -> dict[str, Any]:
        joint_pos = np.asarray(inputs['robot_state.q'], dtype=np.float32).reshape(-1)
        grip = np.asarray(inputs['grip'], dtype=np.float32).reshape(-1)

        obs = {
            'observation/joint_position': joint_pos,
            'observation/gripper_position': grip,
            'observation/wrist_image_left': self._encode_image(self._wrist_camera, inputs),
            'observation/exterior_image_0_left': self._encode_image(self._exterior_camera_1, inputs),
            'observation/exterior_image_1_left': self._encode_image(self._exterior_camera_2, inputs),
        }

        if 'task' in inputs:
            obs['prompt'] = inputs['task']

        return obs

    def dummy_encoded(self, data=None) -> dict[str, Any]:
        w, h = self._image_size
        return {
            'observation/joint_position': np.zeros(7, dtype=np.float32),
            'observation/gripper_position': np.zeros(1, dtype=np.float32),
            'observation/wrist_image_left': np.zeros((h, w, 3), dtype=np.uint8),
            'observation/exterior_image_0_left': np.zeros((h, w, 3), dtype=np.uint8),
            'observation/exterior_image_1_left': np.zeros((h, w, 3), dtype=np.uint8),
            'prompt': 'warmup',
        }

    @property
    def meta(self):
        return {'image_sizes': self._image_size}

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, **self._derive_transforms)


class DreamZeroActionCodec(Codec):
    """Action codec for DreamZero: GR00T-style split signals for training,
    flat action vector decoding for inference.

    Training: derives ``action.joint_position`` and ``action.gripper_position``
    as separate dataset columns with ``gr00t_modality.action`` metadata.
    Inference: decodes flat ``(num_joints+1,)`` action vector into
    ``JointPosition`` command + ``target_grip``.
    """

    def __init__(self, tgt_joints_key: str, tgt_grip_key: str, num_joints: int = 7):
        self._tgt_joints_key = tgt_joints_key
        self._tgt_grip_key = tgt_grip_key
        self._num_joints = num_joints

        self._training_meta = {
            'lerobot_features': {
                'action': lerobot_action(num_joints + 1),
                'action.joint_position': lerobot_state(num_joints),
                'action.gripper_position': lerobot_state(1),
            },
            'gr00t_modality': {
                'action': {
                    'joint_position': {'start': 0, 'end': num_joints, 'original_key': 'action.joint_position'},
                    'gripper_position': {'start': 0, 'end': 1, 'original_key': 'action.gripper_position'},
                }
            },
        }

    def _derive_action_joints(self, episode: Episode) -> Signal[Any]:
        return transforms.astype(episode[self._tgt_joints_key], np.float32)

    def _derive_action_grip(self, episode: Episode) -> Signal[Any]:
        return transforms.Elementwise(episode[self._tgt_grip_key], _reshape_grip)

    def _encode_action(self, episode: Episode):
        return transforms.concat(episode[self._tgt_joints_key], episode[self._tgt_grip_key], dtype=np.float32)

    def _decode_single(self, data: dict, context: dict | None) -> dict:
        action = data['action']
        joints = action[: self._num_joints]
        grip = action[self._num_joints].item()
        return {'robot_command': command.JointPosition(positions=joints), 'target_grip': grip}

    @property
    def training_encoder(self):
        return Derive(
            meta=self._training_meta,
            action=self._encode_action,
            **{
                'action.joint_position': self._derive_action_joints,
                'action.gripper_position': self._derive_action_grip,
            },
        )


@cfn.config(
    wrist_camera='image.wrist',
    exterior_camera_1='image.exterior',
    exterior_camera_2=None,
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
)
def dreamzero_obs(
    wrist_camera: str, exterior_camera_1: str, exterior_camera_2: str | None, image_size: tuple[int, int]
):
    """DreamZero observation codec (3 cameras + joint state)."""
    return DreamZeroObservationCodec(
        wrist_camera=wrist_camera,
        exterior_camera_1=exterior_camera_1,
        exterior_camera_2=exterior_camera_2,
        image_size=tuple(image_size),
    )


@cfn.config(num_joints=7)
def dreamzero_action(tgt_joints_key: str, tgt_grip_key: str, num_joints: int):
    """DreamZero action codec (GR00T split signals + flat inference decode)."""
    return DreamZeroActionCodec(tgt_joints_key=tgt_joints_key, tgt_grip_key=tgt_grip_key, num_joints=num_joints)


# Composed codec: DreamZero observation + action + timing
_action = dreamzero_action.override(tgt_joints_key='robot_commands.joints', tgt_grip_key='target_grip')
joints = codecs.compose.override(obs=dreamzero_obs, action=_action, fps=15.0)

_traj_action = dreamzero_action.override(tgt_joints_key='robot_state.q', tgt_grip_key='grip')
joints_traj = codecs.compose.override(obs=dreamzero_obs, action=_traj_action, fps=15.0)

# IK variants: reconstruct joint targets from recorded EE targets via IK
_ik_action = codecs.ik_joints_action.override(tgt_joints_key='robot_commands.joints', tgt_grip_key='target_grip')


@cfn.config(solver='dls_limits')
def _ik_dreamzero_action(solver: str):
    """IK signal derivation composed with DreamZero action codec."""
    from positronic.drivers.roboarm.ik import DLSIKSolver, DLSIKSolverWithLimits, DmControlIKSolver
    from positronic.policy.action import IKJointsAction

    solver_map = {'dm_control': DmControlIKSolver, 'dls': DLSIKSolver, 'dls_limits': DLSIKSolverWithLimits}
    ik = IKJointsAction(solver_cls=solver_map[solver])
    return ik | DreamZeroActionCodec(tgt_joints_key='robot_commands.joints', tgt_grip_key='target_grip')


joints_ik = codecs.compose.override(obs=dreamzero_obs, action=_ik_dreamzero_action, fps=15.0)
joints_ik_sim = joints_ik.override(**{'action.solver': 'dm_control'})
