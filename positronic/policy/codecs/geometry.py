from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic import geom
from positronic import keys as obs_keys
from positronic.dataset import Signal
from positronic.dataset.episode import Episode
from positronic.dataset.transforms import Elementwise, lazy_sequence
from positronic.dataset.transforms.episode import Derive, EpisodeTransform, FromValue, Group, Identity
from positronic.drivers.roboarm import command
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.ik import assert_default_frame, change_frame, ee_frame
from positronic.drivers.roboarm.models import DEFAULT_FRAME

from .base import Codec

_QUAT = geom.Rotation.Representation.QUAT


class ChangeEEFrame(Codec):
    """Convert poses between the embodiment's ``default`` frame and the frame the policy speaks.

    ``transform`` expresses the policy's frame relative to ``default`` — the frame every embodiment declares in
    its model and reports poses in, so one policy-side constant serves every rig that honours the contract.
    Keys in ``keys`` go ``pose * transform`` on encode and ``pose * transform⁻¹`` on decode; absent keys and
    joint-space commands are left alone. A ``CartesianDelta`` has no anchor pose to convert against, so it
    travels with the frame it is expressed in and the driver applies it there.

    Which side of the ``remote`` marker it sits on decides who converts, the rig or the server. Compose it left
    of the observation/action codecs.
    """

    WIRE_NAME = 'change_ee_frame'

    @staticmethod
    def _move(value: Any, transform: geom.Transform3D) -> Any:
        """A pose vector or an arm command, re-expressed through ``transform``."""
        match value:
            case command.CartesianPosition(pose, mode):
                return command.CartesianPosition(pose=pose * transform, mode=mode)
            case command.CartesianDelta(delta, frame, mode):
                return command.CartesianDelta(delta=delta, frame=transform.inv * frame, mode=mode)
            case command.JointPosition() | command.JointDelta():
                return value
            case _:
                return change_frame(value, transform)

    class _ChangeEpisodeFrames(EpisodeTransform):
        """Move episode pose signals into the policy frame and record their frame."""

        def __init__(self, codec: 'ChangeEEFrame'):
            self._codec = codec

        def _derive_pose(self, key: str, episode):
            codec = self._codec
            return Elementwise(episode[key], lazy_sequence(partial(codec._move, transform=codec._transform)))

        def __call__(self, episode):
            codec = self._codec
            assert_default_frame(episode)
            existing = ee_frame(episode)
            if not np.allclose(existing.as_matrix, np.eye(4)):
                raise ValueError(
                    f'episode poses already sit at {existing.as_vector(_QUAT).tolist()} relative to '
                    f'{DEFAULT_FRAME!r}; ``transform`` names the policy frame from there, so re-expressing an '
                    'episode a codec already moved would train on a frame the checkpoint does not declare'
                )
            derived: dict[str, Any] = {roboarm_keys.EE_FRAME: FromValue(codec._transform.as_vector(_QUAT))}
            derived.update({key: partial(self._derive_pose, key) for key in codec._keys if key in episode})
            return Group(Derive(**derived), Identity())(episode)

        @property
        def meta(self):
            return self._codec.meta

    def __init__(
        self,
        transform: geom.Transform3D | Sequence[float],
        keys: tuple[str, ...] = (obs_keys.EE_POSE, obs_keys.TARGET_EE_POSE, obs_keys.ROBOT_COMMAND),
    ):
        if not isinstance(transform, geom.Transform3D):  # the ``[tx,ty,tz,qw,qx,qy,qz]`` vector a wire spec carries
            transform = geom.Transform3D.from_vector(np.asarray(transform, dtype=np.float64), _QUAT)
        self._transform = transform
        self._keys = tuple(keys)

    def _apply(self, data: dict, transform: geom.Transform3D) -> dict:
        moved = {key: self._move(data[key], transform) for key in self._keys if key in data}
        return {**data, **moved} if moved else data

    def encode(self, data):
        return self._apply(data, self._transform)

    def _decode_single(self, data: dict) -> dict:
        return self._apply(data, self._transform.inv)

    @property
    def training_encoder(self) -> EpisodeTransform:
        return ChangeEEFrame._ChangeEpisodeFrames(self)

    @property
    def meta(self):
        return {roboarm_keys.EE_FRAME: self._transform.as_vector(_QUAT).tolist()}

    def to_spec(self):
        # Lists, not tuples, so the spec is identical before and after a wire round-trip.
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {'transform': self._transform.as_vector(_QUAT).tolist(), 'keys': list(self._keys)},
        }


class ConvertPose(Codec):
    """Convert selected observation poses and recorded pose signals to float32 vectors.

    The offset multiplies each rotation on the right; translation and frame metadata stay unchanged.
    Wire offsets are wxyz quaternions. Selected fields are required. Decoded actions pass through.
    """

    WIRE_NAME = 'convert_pose'

    def __init__(
        self,
        output_rotation: str,
        *,
        keys: Sequence[str] = (obs_keys.EE_POSE,),
        input_rotation: str = geom.Rotation.Representation.QUAT.value,
        rotation_offset: geom.Rotation | Sequence[float] = (1, 0, 0, 0),
    ):
        self._keys = tuple(keys)
        self._input_rotation = geom.Rotation.Representation(input_rotation)
        self._output_rotation = geom.Rotation.Representation(output_rotation)
        if not isinstance(rotation_offset, geom.Rotation):
            rotation_offset = geom.Rotation.from_quat(np.asarray(rotation_offset))
        self._rotation_offset = rotation_offset

    def _convert(self, value: Any) -> np.ndarray:
        vector = np.asarray(value)
        expected = 3 + self._input_rotation.size
        if vector.shape != (expected,):
            raise ValueError(f'Expected a pose vector with {expected} values, got shape {vector.shape}')
        pose = geom.Transform3D.from_vector(vector, self._input_rotation)
        converted = geom.Transform3D(pose.translation, pose.rotation * self._rotation_offset)
        return converted.as_vector(self._output_rotation).astype(np.float32)

    def encode(self, data: dict[str, Any]) -> dict[str, Any]:
        return {**data, **{key: self._convert(data[key]) for key in self._keys}}

    def decode(self, data: Any) -> Any:
        return data

    def _derive_pose(self, key: str, episode: Episode) -> Signal[Any]:
        return Elementwise(episode[key], lazy_sequence(self._convert))

    @property
    def training_encoder(self) -> EpisodeTransform:
        return Group(Derive(meta=None, **{key: partial(self._derive_pose, key) for key in self._keys}), Identity())

    def to_spec(self) -> dict[str, Any]:
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {
                'output_rotation': self._output_rotation.value,
                'keys': list(self._keys),
                'input_rotation': self._input_rotation.value,
                'rotation_offset': self._rotation_offset.as_quat.tolist(),
            },
        }
