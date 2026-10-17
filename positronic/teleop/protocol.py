"""The requests and the replies between ``TeleopClient`` and the teleop service.

Each request is the observation of one inference call of protocol v3, and each reply is its result. Both are maps,
with ``KIND`` naming the request or the reply. ``TeleopClient`` sends ``READY`` requests until the teleoperator is
ready, and then one ``FRAME`` request at a time.
"""

import os
import tempfile
from enum import StrEnum
from pathlib import Path


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
