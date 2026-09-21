"""Env-server wire names and msgpack encoding for numpy arrays and plain data.

This module must work in an isolated interpreter without Positronic installed.
Arrays use raw bytes, ``dtype.str``, and shape for compatibility between numpy versions.
"""

import functools
from collections.abc import Callable
from enum import Enum
from typing import Any, cast

import msgpack
import numpy as np

CMD = 'cmd'
OK = 'ok'
TASKS = 'tasks'
SPEC = 'spec'
TOKEN = 'token'
ACTION = 'action'
ERROR = 'error'


class Command(Enum):
    TASKS = 'tasks'
    RESET = 'reset'
    STEP = 'step'
    CLOSE = 'close'


# Arm command tags shared by clients and environment servers.
CARTESIAN = 'cartesian'
CARTESIAN_DELTA = 'cartesian_delta'
JOINT_POS = 'joint_pos'
JOINT_DELTA = 'joint_vel'  # Wire spelling used by existing environment servers.
HOLD = 'hold'
CANONICAL_COMMAND_TYPES = (CARTESIAN, CARTESIAN_DELTA, JOINT_POS, JOINT_DELTA, HOLD)

# An action carries one entry per arm the embodiment drives, each naming the arm it moves; an embodiment
# with one arm leaves it unnamed and sends a list of one.
ACTION_ARMS = 'arms'
ARM_NAME = 'name'  # The arm's name, or ``None`` for the sole arm of a one-armed embodiment.
ACTION_COMMAND = 'command'
ACTION_GRIP = 'grip'  # Closure in [0, 1].

COMMAND_TYPE = 'type'
COMMAND_POSE = 'pose'  # CARTESIAN — an absolute pose, [t(3), R(9)]
COMMAND_DELTA = 'delta'  # CARTESIAN_DELTA — a relative pose, same encoding
COMMAND_JOINT_POS = 'q'  # JOINT_POS — absolute joint targets
COMMAND_JOINT_DELTA = 'dq'  # JOINT_DELTA — per-step joint deltas
COMMAND_MODE = 'mode'  # Optional control law.

# Response fields; required fields for reset and step are defined by EnvProtocol.
FRAME_OBS = 'obs'
FRAME_META = 'meta'
FRAME_ROBOT_META = 'robot_meta'
FRAME_CONTROL_DT = 'control_dt'
FRAME_DONE = 'done'
FRAME_SUCCESS = 'success'


def sole_arm_action(command: dict[str, Any], grip: float) -> dict[str, Any]:
    """An action for one unnamed arm — the shape a single-arm embodiment sends."""
    return {ACTION_ARMS: [{ARM_NAME: None, ACTION_COMMAND: command, ACTION_GRIP: grip}]}


def sole_arm(action: dict[str, Any]) -> dict[str, Any]:
    """The one arm entry of ``action``, for an env whose model has a single arm.

    Raises when the client drives a different number of arms than the env has, which no env can act on.
    """
    arms = action[ACTION_ARMS]
    if len(arms) != 1:
        raise ValueError(f'this env drives one arm, the action carries {len(arms)}')
    return arms[0]


def _pack(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ('V', 'O', 'c'):
            raise ValueError(f'Unsupported dtype: {obj.dtype}')
        return {b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str, b'shape': obj.shape}
    if isinstance(obj, np.generic):
        return {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}
    return obj


def _unpack(obj):
    if b'__ndarray__' in obj:
        # A bytearray keeps the decoded array writable.
        return np.ndarray(buffer=bytearray(obj[b'data']), dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])
    return obj


# msgpack's default autoreset=True makes packb return bytes.
encode = cast(Callable[[Any], bytes], functools.partial(msgpack.packb, default=_pack))
decode = functools.partial(msgpack.unpackb, object_hook=_unpack)
