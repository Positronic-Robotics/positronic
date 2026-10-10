from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic import geom, keys
from positronic.dataset import transforms
from positronic.dataset.episode import Episode, select_timeline
from positronic.dataset.signal import Signal
from positronic.dataset.transforms.episode import Derive, Group, Identity
from positronic.drivers.roboarm import command
from positronic.drivers.roboarm.ik import ik_joints_from_episode

from .base import Codec
from .metadata import ACTION, LEROBOT_FEATURES, lerobot_action

RotRep = geom.Rotation.Representation


class UnpackActionChunk(Codec):
    """Select prediction arrays and split their time axis into action records.

    ``fields`` maps output names to paths of literal dictionary keys or sequence indices.
    ``squeeze_dims`` removes leading size-one dimensions before the time axis.
    For example, ``UnpackActionChunk({'joints': [0, 'q']}, squeeze_dims=1)`` reads
    ``({'q': array_of_shape_1_T_7}, info)`` as T records containing a ``(7,)`` joints array.
    Observations and training columns pass through unchanged.
    """

    WIRE_NAME = 'unpack_action_chunk'

    def __init__(self, fields: Mapping[str, Sequence[str | int]], *, squeeze_dims: int = 0):
        if not fields:
            raise ValueError('At least one prediction field is required')
        if type(squeeze_dims) is not int or squeeze_dims < 0:
            raise ValueError('squeeze_dims must be a non-negative integer')
        for name, path in fields.items():
            if (
                not isinstance(name, str)
                or not isinstance(path, Sequence)
                or isinstance(path, (str, bytes))
                or any(type(part) not in (str, int) for part in path)
            ):
                raise ValueError('Prediction fields require string names and paths of string keys or integer indices')
        self._fields = {name: tuple(path) for name, path in fields.items()}
        self._squeeze_dims = squeeze_dims

    def encode(self, data: dict) -> dict:
        return data

    def _array(self, data: Any, name: str, path: Sequence[str | int]) -> np.ndarray:
        for part in path:
            data = data[part]
        if (
            not isinstance(data, np.ndarray)
            or data.ndim <= self._squeeze_dims
            or any(size != 1 for size in data.shape[: self._squeeze_dims])
        ):
            raise ValueError(
                f'Prediction {name!r} must be an array with {self._squeeze_dims} '
                'leading size-one dimensions and a time axis'
            )
        return data.reshape(data.shape[self._squeeze_dims :])

    def decode(self, data: Any) -> list[dict[str, Any]]:
        arrays = {name: self._array(data, name, path) for name, path in self._fields.items()}
        horizons = {len(array) for array in arrays.values()}
        if len(horizons) != 1:
            raise ValueError(f'Prediction fields must share one horizon, got {sorted(horizons)}')
        return [{name: array[index] for name, array in arrays.items()} for index in range(horizons.pop())]

    @property
    def training_encoder(self):
        return Identity()

    def to_spec(self):
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {
                'fields': {name: list(path) for name, path in self._fields.items()},
                'squeeze_dims': self._squeeze_dims,
            },
        }


class JointPositionAction(Codec):
    """Decode named joint and grip predictions into an absolute robot command."""

    WIRE_NAME = 'joint_position_action'

    def __init__(self, joints_key: str, grip_key: str, num_joints: int = 7):
        self._joints_key = joints_key
        self._grip_key = grip_key
        self._num_joints = num_joints

    def _decode_single(self, data: dict) -> dict:
        return {
            keys.ROBOT_COMMAND: command.JointPosition(
                positions=np.asarray(data[self._joints_key]).reshape(self._num_joints)
            ),
            keys.TARGET_GRIP: np.asarray(data[self._grip_key]).item(),
        }

    def to_spec(self):
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {'joints_key': self._joints_key, 'grip_key': self._grip_key, 'num_joints': self._num_joints},
        }


class AbsolutePositionAction(Codec):
    WIRE_NAME = 'absolute_position_action'

    def __init__(self, tgt_ee_pose_key: str, tgt_grip_key: str, rotation_rep: RotRep | str = RotRep.QUAT):
        self.rot_rep = RotRep(rotation_rep)
        self.tgt_ee_pose_key = tgt_ee_pose_key
        self.tgt_grip_key = tgt_grip_key

        ee_dim = self.rot_rep.size + 3
        self._training_meta = {LEROBOT_FEATURES: {ACTION: lerobot_action(ee_dim + 1)}}

    def encode(self, data):
        return {}

    def _decode_single(self, data: dict) -> dict:
        action_vector = data[ACTION]
        target_pose = geom.Transform3D.from_vector(action_vector[:-1], self.rot_rep)
        target_grip = action_vector[-1].item()
        return {keys.ROBOT_COMMAND: command.CartesianPosition(pose=target_pose), keys.TARGET_GRIP: target_grip}

    def _encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        pose = episode[self.tgt_ee_pose_key]
        pose = transforms.recode_transform(RotRep.QUAT, self.rot_rep, pose)
        return transforms.concat(
            pose,
            episode[self.tgt_grip_key],
            dtype=np.float32,
            timelines=(select_timeline(pose.timelines + episode[self.tgt_grip_key].timelines),),
        )

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, **{ACTION: self._encode_episode})

    def to_spec(self):
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {
                'tgt_ee_pose_key': self.tgt_ee_pose_key,
                'tgt_grip_key': self.tgt_grip_key,
                'rotation_rep': self.rot_rep.value,
            },
        }


class AbsoluteJointsAction(Codec):
    WIRE_NAME = 'absolute_joints_action'

    def __init__(self, tgt_joints_key: str, tgt_grip_key: str, num_joints: int = 7):
        self.tgt_joints_key = tgt_joints_key
        self.tgt_grip_key = tgt_grip_key
        self.num_joints = num_joints

        self._training_meta = {LEROBOT_FEATURES: {ACTION: lerobot_action(num_joints + 1)}}

    def encode(self, data):
        return {}

    def _decode_single(self, data: dict) -> dict:
        action_vector = data[ACTION]
        if action_vector.shape[-1] != self.num_joints + 1:
            raise ValueError(f'Expected action vector of size {self.num_joints + 1}, got {action_vector.shape[-1]}')

        joint_positions = action_vector[: self.num_joints]
        target_grip = action_vector[-1].item()
        return {keys.ROBOT_COMMAND: command.JointPosition(positions=joint_positions), keys.TARGET_GRIP: target_grip}

    def _encode_episode(self, episode: Episode) -> Signal[np.ndarray]:
        return transforms.concat(
            episode[self.tgt_joints_key],
            episode[self.tgt_grip_key],
            dtype=np.float32,
            timelines=(select_timeline(episode[self.tgt_joints_key].timelines + episode[self.tgt_grip_key].timelines),),
        )

    @property
    def training_encoder(self):
        return Derive(meta=self._training_meta, **{ACTION: self._encode_episode})

    def to_spec(self):
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {
                'tgt_joints_key': self.tgt_joints_key,
                'tgt_grip_key': self.tgt_grip_key,
                'num_joints': self.num_joints,
            },
        }


class IKJointsAction(Codec):
    """Signal-level codec that replaces EE pose targets with joint targets via IK.

    Training: replaces ``tgt_ee_pose_key`` with ``tgt_joints_key`` in the episode.
    Inference: pass-through.
    Compose with AbsoluteJointsAction for inference decoding.
    """

    def __init__(
        self,
        solver_cls,
        *,
        tgt_ee_pose_key=keys.TARGET_EE_POSE,
        current_q_key=keys.JOINTS,
        tgt_joints_key=keys.TARGET_JOINTS,
    ):
        self.solver_cls = solver_cls
        self.tgt_ee_pose_key = tgt_ee_pose_key
        self.current_q_key = current_q_key
        self.tgt_joints_key = tgt_joints_key

    def encode(self, data):
        return data

    def _decode_single(self, data: dict) -> dict:
        return data

    def _derive_joints(self, episode: Episode):
        return ik_joints_from_episode(episode, self.solver_cls, self.tgt_ee_pose_key, self.current_q_key)

    @property
    def training_encoder(self):
        return Group(
            Derive(meta=None, **{self.tgt_joints_key: self._derive_joints}), Identity(remove=[self.tgt_ee_pose_key])
        )


class JointDeltaAction(Codec):
    """DROID-style joint-delta action decoder (inference only).

    Scales the model's per-step joint velocities (clipped to ``[-1, 1]``) by ``MAX_JOINT_DELTA``
    into a ``JointDelta`` command; the driver integrates each delta onto the live measured joints.
    """

    WIRE_NAME = 'joint_delta_action'

    # General DROID form scales each normalized velocity by its own per-joint delta limit, then
    # renorms the velocity vector so no joint exceeds its limit:
    #     RELATIVE_MAX_JOIN_DELTA = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    #     MAX_JOINT_DELTA = RELATIVE_MAX_JOIN_DELTA.max()
    #     MAX_JOINT_VEL = RELATIVE_MAX_JOIN_DELTA / MAX_JOINT_DELTA
    #     max_vel_norm = (np.abs(velocities) / MAX_JOINT_VEL).max()
    #     if max_vel_norm > 1.0:
    #         velocities = velocities / max_vel_norm
    # All seven limits are equal, so MAX_JOINT_VEL is all ones and, on the [-1, 1]-clipped
    # velocities, max_vel_norm <= 1 — the renorm never fires and the scaling collapses to one
    # multiply by the scalar below.
    MAX_JOINT_DELTA = 0.2

    def __init__(self, num_joints: int = 7):
        self.num_joints = num_joints

    def encode(self, data):
        return {}

    def _decode_single(self, data: dict) -> dict:
        action_vector = data[ACTION]
        if action_vector.shape[-1] != self.num_joints + 1:
            raise ValueError(f'Expected action vector of size {self.num_joints + 1}, got {action_vector.shape[-1]}')

        action_vector = action_vector.clip(-1.0, 1.0)
        velocities = action_vector[: self.num_joints] * self.MAX_JOINT_DELTA
        grip = 1.0 if action_vector[self.num_joints].item() > 0.5 else 0.0
        return {keys.ROBOT_COMMAND: command.JointDelta(velocities=velocities), keys.TARGET_GRIP: grip}

    def to_spec(self):
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: {'num_joints': self.num_joints}}


class SetControlMode(Codec):
    """Sets the control mode a chunk executes under on every robot command it carries (inference only).

    Composes left of an action decoder (``SetControlMode(mode) | action``).
    """

    WIRE_NAME = 'set_control_mode'

    def __init__(self, mode: command.ControlModeType | dict[str, Any]):
        parsed = command.from_wire(mode) if isinstance(mode, dict) else mode
        if not isinstance(parsed, command.ControlModeType):
            raise ValueError('Expected an arm control mode')
        self._mode = parsed

    def encode(self, data):
        return data

    def _decode_single(self, data: dict) -> dict:
        # The command family also holds the pose/joint vectors a recording unfolds into, so what carries a
        # mode is decided by type rather than by name.
        stamped = {
            key: replace(cmd, mode=self._mode)
            for key, cmd in data.items()
            if keys.is_robot_command(key) and isinstance(cmd, command.CommandType)
        }
        return {**data, **stamped} if stamped else data

    def to_spec(self):
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: {'mode': command.to_wire(self._mode)}}
