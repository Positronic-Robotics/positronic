"""The frame requests, the frame replies and the session parameters of the teleop service.

A session prepares until the teleoperator presses Ready. Then each inference call of the client stack is one frame: the
request is the observation that the stack encodes, and the reply is a chunk of one element.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, check_int_range

from positronic import geom, keys
from positronic.utils import check_positive_int

from .clutch import OperatorPosition

# The fields of a frame request. Each camera image of the session is a field under its camera name.
TASK = 'task'
# The pose of an arm, xyz and a wxyz quaternion. The clutch starts control from it. A field of each arm, named by
# `keys.arm_channel`.
EE_POSE = 'ee_pose'
# The grip of an arm. The service holds it until control comes on. A field of each arm, as `EE_POSE` is.
GRIP = 'grip'


class Arms(Enum):
    """The arms that a session drives, as the arm names of their channels."""

    # One arm, on the channels with no arm name. The right controller drives it.
    ONE = keys.ONE_ARM
    # Two arms. The left controller drives the left arm, and the right controller drives the right arm.
    BIMANUAL = keys.BIMANUAL_ARMS

    def arm_fields(self) -> tuple[str, ...]:
        """The fields of a frame request that carry the state of the arms."""
        return tuple(keys.arm_channel(field, arm) for arm in self.value for field in (EE_POSE, GRIP))


# The width and the height of each camera image that a frame request carries, and of the view of the headset.
VIEW_SIZE = (640, 480)

# The port of the sessions of a teleop service on a host and a port. The WebXR page is on 5005 and the window on 5006.
DEFAULT_PORT = 5007


def default_socket_path() -> Path:
    """The socket of the teleop service: in the runtime directory of the user, or in a directory of the user under
    the temporary directory."""
    runtime = os.environ.get('XDG_RUNTIME_DIR')
    if runtime:
        return Path(runtime) / 'positronic-teleop.sock'
    return Path(tempfile.gettempdir()) / f'positronic-{os.getuid()}' / 'teleop.sock'


def camera_names(value: Any) -> tuple[str, ...]:
    """``value`` as camera names. Raise for a string, or for a list that holds something other than names."""
    if isinstance(value, str) or not isinstance(value, list | tuple) or not all(isinstance(n, str) for n in value):
        raise ValueError(f'cameras must be a list of camera names, got {value!r}')
    return tuple(value)


@dataclass(frozen=True)
class SessionParams:
    """The parameters of a session. A client sets them as JSON values in the query of its address, with an
    ``OperatorPosition`` or ``Arms`` as its name.

    An ``operator_position`` of ``None`` is UMI mode: control is always on, and the hand pose is the robot pose.
    ``arms`` names the arms that the session drives, and so the controllers that drive them.
    ``cameras`` names the cameras that the client sends. ``headset_camera`` names the one that the headset shows;
    ``None`` is the first one. The client sends the camera images as JPEG at ``jpeg_quality``. With a
    ``video_bitrate`` in bits per second, it sends them as H.264 at that bit rate and ``video_fps`` in place of JPEG.
    """

    operator_position: OperatorPosition | None = OperatorPosition.FRONT
    arms: Arms = Arms.ONE
    cameras: tuple[str, ...] = ()
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    headset_camera: str | None = None
    video_bitrate: int | None = None
    video_fps: int = 30

    def __post_init__(self) -> None:
        if self.operator_position is not None and not isinstance(self.operator_position, OperatorPosition):
            names = ', '.join(OperatorPosition.__members__)
            raise ValueError(
                f'operator_position must be one of {names}, or null for UMI mode, got {self.operator_position!r}'
            )
        if not isinstance(self.arms, Arms):
            raise ValueError(f'arms must be one of {", ".join(Arms.__members__)}, got {self.arms!r}')
        check_int_range(self.jpeg_quality, 0, 100, 'jpeg_quality')
        if self.video_bitrate is not None:
            check_positive_int(self.video_bitrate, 'video_bitrate')
        check_positive_int(self.video_fps, 'video_fps')
        if fields := {TASK, *self.arms.arm_fields()} & set(self.cameras):
            raise ValueError(f'a camera name must not be a field of a frame request, got {sorted(fields)}')
        if self.headset_camera is not None and self.headset_camera not in self.cameras:
            raise ValueError(
                f'headset_camera must be one of the cameras {list(self.cameras)}, got {self.headset_camera!r}'
            )

    @classmethod
    def from_json(cls, operator_position: Any, arms: Any, cameras: Any, **params: Any) -> SessionParams:
        """The parameters that the JSON values of a session name. Raise for a name that is no ``OperatorPosition`` or
        no ``Arms``."""
        if isinstance(operator_position, str):
            operator_position = OperatorPosition.__members__.get(operator_position, operator_position)
        if isinstance(arms, str):
            arms = Arms.__members__.get(arms, arms)
        return cls(operator_position, arms, camera_names(cameras), **params)

    def to_json(self) -> dict[str, Any]:
        """The parameters as JSON values."""
        values = asdict(self).items()
        return {name: value.name if isinstance(value, Enum) else value for name, value in values}

    def operator(self) -> geom.Transform3D | None:
        """The transform of the operator position, or ``None`` in UMI mode."""
        return None if self.operator_position is None else self.operator_position.value


def session_query(**params: Any) -> str:
    """``params`` as the query of a session address, each value as a JSON literal. Raise for a name or a value that
    ``SessionParams`` refuses."""
    SessionParams.from_json(**{**SessionParams().to_json(), **params})
    return urlencode({name: json.dumps(value) for name, value in params.items()})
