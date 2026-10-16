"""The session between ``TeleopPolicy`` and the teleop service.

The policy sends one request and waits for its answer. The first request is ``ready``, and every later one is
``frame``. Each message is one msgpack map, with ``KIND`` naming the request or the reply.
"""

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, deserialise, encode_jpeg, serialise
from positronic_wire.wire import ClientConnection

from positronic import geom


class Request(StrEnum):
    # Wait until the teleoperator is ready for the trial.
    READY = 'ready'
    # Send the display state, and answer with the next frame of the teleoperator input.
    FRAME = 'frame'


class Reply(StrEnum):
    READY = 'ready'
    CANCELLED = 'cancelled'
    FRAME = 'frame'
    ERROR = 'error'


class Event(StrEnum):
    """A press that the teleoperator made since the previous frame."""

    # Turn control on or off.
    CLUTCH = 'clutch'
    # The real attempt starts. The moves before it are trial moves.
    START_TASK = 'start_task'
    # End the episode: the teleoperator thinks that the task is done.
    COMPLETE = 'complete'
    # End the episode: the teleoperator stops before the task is done.
    GIVE_UP = 'give_up'


# The fields of a message.
KIND = 'kind'
DISPLAY = 'display'
# The pose of the right controller, xyz and a wxyz quaternion. It is absent when the headset does not track it.
POSE = 'pose'
GRIP = 'grip'
EVENTS = 'events'
MESSAGE = 'message'

# The fields of the display state.
TASK = 'task'
CONTROL = 'control'
ATTEMPT_STARTED = 'attempt_started'
# Camera images from `Obs`, JPEG-encoded, by camera name. A frame request carries them at most at the image rate of the
# policy, and the service keeps the last ones it got. It is absent from the other requests.
IMAGES = 'images'


def default_socket_path() -> Path:
    """The socket of the teleop service: in the runtime directory of the user, or in a directory of the user under
    the temporary directory."""
    runtime = os.environ.get('XDG_RUNTIME_DIR')
    if runtime:
        return Path(runtime) / 'positronic-teleop.sock'
    return Path(tempfile.gettempdir()) / f'positronic-{os.getuid()}' / 'teleop.sock'


class Cancelled(Exception):
    """The teleoperator cancelled the trial before it started."""


class ServiceError(Exception):
    """The teleop service answered a request with an error."""


@dataclass(frozen=True)
class Frame:
    """The teleoperator input since the previous frame. ``pose`` is ``None`` while the headset does not track it."""

    pose: geom.Transform3D | None
    grip: float
    events: tuple[Event, ...]


class TeleopSession:
    """The policy's end of one session with the teleop service."""

    def __init__(
        self,
        connection: ClientConnection,
        ready_timeout: float,
        frame_timeout: float,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ):
        self._connection = connection
        self._ready_timeout = ready_timeout
        self._frame_timeout = frame_timeout
        self._jpeg_quality = jpeg_quality

    def _ask(self, message: dict[str, Any], reply: Reply, timeout: float) -> dict[str, Any]:
        self._connection.send(serialise(message))
        answer = deserialise(self._connection.recv(timeout))
        match Reply(answer[KIND]):
            case Reply.ERROR:
                raise ServiceError(answer[MESSAGE])
            case Reply.CANCELLED:
                raise Cancelled('the teleoperator cancelled the trial')
            case kind if kind is not reply:
                raise ValueError(f'the teleop service answered {kind} to a {message[KIND]} request')
        return answer

    def ready(self) -> None:
        """Return when the teleoperator is ready. Raise ``Cancelled`` when the teleoperator cancels, and
        ``TimeoutError`` when no reply comes within the ready timeout."""
        try:
            self._ask({KIND: Request.READY}, Reply.READY, self._ready_timeout)
        except TimeoutError as timeout:
            raise TimeoutError(f'the teleoperator was not ready within {self._ready_timeout} s') from timeout

    def frame(self, display: dict[str, Any], images: Mapping[str, np.ndarray] | None = None) -> Frame:
        """Send ``display``, with ``images`` JPEG-encoded when given, and answer with the next frame."""
        if images is not None:
            display = {
                **display,
                IMAGES: {name: encode_jpeg(image, self._jpeg_quality) for name, image in images.items()},
            }
        answer = self._ask({KIND: Request.FRAME, DISPLAY: display}, Reply.FRAME, self._frame_timeout)
        pose = answer.get(POSE)
        return Frame(
            pose=None
            if pose is None
            else geom.Transform3D.from_vector(np.asarray(pose), geom.Rotation.Representation.QUAT),
            grip=float(answer[GRIP]),
            events=tuple(Event(event) for event in answer[EVENTS]),
        )

    def close(self) -> str:
        return self._connection.close()
