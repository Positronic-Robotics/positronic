"""The requests, the replies and the session parameters between ``TeleopClient`` and the teleop service.

Each request is the observation of one inference call of protocol v3, and each reply is its result. Both are maps,
with ``KIND`` naming the request or the reply. ``TeleopClient`` sends ``READY`` requests until the teleoperator is
ready, and then one ``FRAME`` request at a time.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY

from positronic import geom

from .clutch import OperatorPosition


class Request(StrEnum):
    # Wait a short time for the teleoperator to be ready for the trial.
    READY = 'ready'
    # Send the state of the robot, and answer with the next frame of the teleoperator input.
    FRAME = 'frame'


class Reply(StrEnum):
    # The teleoperator is not ready yet. Ask again.
    WAITING = 'waiting'
    READY = 'ready'
    CANCELLED = 'cancelled'
    # The ready limit of the service ended before the teleoperator was ready.
    EXPIRED = 'expired'
    FRAME = 'frame'


KIND = 'kind'

# The fields of a frame request.
TASK = 'task'
# The pose of the robot, xyz and a wxyz quaternion. The clutch starts control from it.
EE_POSE = 'ee_pose'
# Camera images from `Obs`, by camera name. A request carries them at most at the image rate of the client, and the
# service keeps the last ones it got.
IMAGES = 'images'

# The fields of a frame reply.
# The pose that the robot must go to, xyz and a wxyz quaternion. It is present only while control is on.
TARGET = 'target'
# The grip that the robot must go to. It is present with `TARGET`.
GRIP = 'grip'
# True in the one reply after the Start task press.
START_TASK = 'start_task'
# A `PolicyEnd` value, present in the reply after a Complete or a Give up.
END = 'end'


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
    ``cameras`` names the cameras that the client sends.
    """

    operator_position: OperatorPosition | None = OperatorPosition.FRONT
    cameras: tuple[str, ...] = ()
    image_fps: float = 15.0
    still_clock_timeout: float = 1.0
    jpeg_quality: int = DEFAULT_JPEG_QUALITY

    def __post_init__(self) -> None:
        for name, value in (('image_fps', self.image_fps), ('still_clock_timeout', self.still_clock_timeout)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive, got {value}')

    @classmethod
    def from_json(cls, operator_position: str | None, cameras: Any, **params: Any) -> SessionParams:
        """The parameters that the JSON values of a session name. Raise for a name that is no ``OperatorPosition``."""
        names = [position.name for position in OperatorPosition]
        if operator_position is not None and operator_position not in names:
            raise ValueError(f'operator_position must be one of {", ".join(names)}, or null for UMI mode')
        position = None if operator_position is None else OperatorPosition[operator_position]
        return cls(position, camera_names(cameras), **params)

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
