"""The frame requests, the frame replies and the session parameters of the teleop service.

A session prepares until the teleoperator presses Ready. Then each inference call of the client stack is one frame: the
request is the observation that the stack encodes, and the reply is a chunk of one element.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY

from positronic import geom

from .clutch import OperatorPosition

# The fields of a frame request. Each camera image of the session is a field under its camera name.
TASK = 'task'
# The pose of the robot, xyz and a wxyz quaternion. The clutch starts control from it.
EE_POSE = 'ee_pose'
# The grip of the robot. The service holds it until control comes on.
GRIP = 'grip'

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
    ``OperatorPosition`` as its name.

    An ``operator_position`` of ``None`` is UMI mode: control is always on, and the hand pose is the robot pose.
    ``cameras`` names the cameras that the client sends. ``headset_camera`` names the one that the headset shows;
    ``None`` is the first one.
    """

    operator_position: OperatorPosition | None = OperatorPosition.FRONT
    cameras: tuple[str, ...] = ()
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    headset_camera: str | None = None

    def __post_init__(self) -> None:
        if self.operator_position is not None and not isinstance(self.operator_position, OperatorPosition):
            names = ', '.join(OperatorPosition.__members__)
            raise ValueError(
                f'operator_position must be one of {names}, or null for UMI mode, got {self.operator_position!r}'
            )
        if type(self.jpeg_quality) is not int or not 0 <= self.jpeg_quality <= 100:
            raise ValueError(f'jpeg_quality must be an integer between 0 and 100, got {self.jpeg_quality!r}')
        if fields := {TASK, EE_POSE, GRIP} & set(self.cameras):
            raise ValueError(f'a camera name must not be a field of a frame request, got {sorted(fields)}')
        if self.headset_camera is not None and self.headset_camera not in self.cameras:
            raise ValueError(
                f'headset_camera must be one of the cameras {list(self.cameras)}, got {self.headset_camera!r}'
            )

    @classmethod
    def from_json(cls, operator_position: Any, cameras: Any, **params: Any) -> SessionParams:
        """The parameters that the JSON values of a session name. Raise for a name that is no ``OperatorPosition``."""
        if isinstance(operator_position, str):
            operator_position = OperatorPosition.__members__.get(operator_position, operator_position)
        return cls(operator_position, camera_names(cameras), **params)

    def to_json(self) -> dict[str, Any]:
        """The parameters as JSON values."""
        values = asdict(self).items()
        return {name: value.name if isinstance(value, OperatorPosition) else value for name, value in values}

    def operator(self) -> geom.Transform3D | None:
        """The transform of the operator position, or ``None`` in UMI mode."""
        return None if self.operator_position is None else self.operator_position.value


def session_query(**params: Any) -> str:
    """``params`` as the query of a session address, each value as a JSON literal. Raise for a name or a value that
    ``SessionParams`` refuses."""
    SessionParams.from_json(**{**SessionParams().to_json(), **params})
    return urlencode({name: json.dumps(value) for name, value in params.items()})
