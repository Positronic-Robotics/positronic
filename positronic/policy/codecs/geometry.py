from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic import geom
from positronic import keys as obs_keys
from positronic.dataset.transforms.episode import EpisodeTransform, map_signals
from positronic.drivers.roboarm.ik import change_frame
from positronic.drivers.roboarm.models import EEFrame, ee_frames_entry
from positronic.policy import training

from .base import Codec

_QUAT = geom.Rotation.Representation.QUAT


class ChangeEEFrame(Codec):
    """Convert poses between a site of the embodiment's model and the frame the policy speaks.

    The policy speaks ``transform`` from ``site``, and states it as the ``EE_FRAMES`` entry of each key in ``keys``.
    The rig reports its poses at a site with no offset and refuses a policy that names another site. Keys in
    ``keys`` go ``pose * transform`` on encode and ``pose * transform⁻¹`` on decode; absent keys and joint-space
    commands are left alone. A ``CartesianDelta`` has no anchor pose to convert against, so it travels with the
    frame it is expressed in and the driver applies it there.

    Which side of the ``remote`` marker it sits on decides who converts, the rig or the server. Compose it left
    of the observation/action codecs.
    """

    WIRE_NAME = 'change_ee_frame'
    WIRE_VERSION = 2

    def __init__(
        self,
        site: str,
        transform: geom.Transform3D,
        keys: tuple[str, ...] = (obs_keys.EE_POSE, obs_keys.TARGET_EE_POSE, obs_keys.ROBOT_COMMAND),
    ):
        self._frame = EEFrame(site=site, offset=transform)
        self._keys = tuple(keys)

    @classmethod
    def from_spec_args(cls, **args: Any) -> Self:
        # A wire spec carries the transform as a ``[tx,ty,tz,qw,qx,qy,qz]`` vector.
        args['transform'] = geom.Transform3D.from_vector(np.asarray(args['transform'], dtype=np.float64), _QUAT)
        return cls(**args)

    def _apply(self, data: dict, transform: geom.Transform3D) -> dict:
        moved = {key: change_frame(data[key], transform) for key in self._keys if key in data}
        return {**data, **moved} if moved else data

    def encode(self, data):
        return self._apply(data, self._frame.offset)

    def _decode_single(self, data: dict) -> dict:
        return self._apply(data, self._frame.offset.inv)

    @property
    def training_encoder(self) -> EpisodeTransform:
        return training.ChangeEEFrame(self._frame.site, self._frame.offset, self._keys)

    @property
    def meta(self):
        return ee_frames_entry(dict.fromkeys(self._keys, self._frame))

    def to_spec(self):
        # Lists, not tuples, so the spec is identical before and after a wire round-trip.
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {
                'site': self._frame.site,
                'transform': self._frame.offset.as_vector(_QUAT).tolist(),
                'keys': list(self._keys),
            },
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
        converted = pose * geom.Transform3D(rotation=self._rotation_offset)
        return converted.as_vector(self._output_rotation).astype(np.float32)

    def encode(self, data: dict[str, Any]) -> dict[str, Any]:
        return {**data, **{key: self._convert(data[key]) for key in self._keys}}

    def decode(self, data: Any) -> Any:
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        return map_signals(self._convert, self._keys)

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
