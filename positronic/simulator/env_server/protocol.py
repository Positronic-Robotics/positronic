"""The wire contract for the remote env-server boundary: the command vocabulary and the msgpack codec.

This module is **positronic-free** — it imports only ``msgpack`` and ``numpy`` — so it can be
imported (or copied) into a benchmark's isolated interpreter alongside the dumb server without
dragging in pimm or the rest of positronic. Only raw numpy arrays and plain-data dicts cross the
wire; every canonical<->raw mapping lives client-side in the ``EnvAdapter``.

Arrays travel as raw bytes plus their ``dtype.str`` and shape, so a numpy-2 server round-trips a
numpy-1 client unchanged.
"""

import functools
from typing import Any

import msgpack
import numpy as np

# The canonical command contract: the tag on every arm command a client puts on the wire. It is total — one
# contract carries every policy onto every embodiment — so an env adoption converts each of these into
# whatever its own controller natively takes. Owned here because both interpreters spell the tags:
# positronic's ``EnvAdapter`` writes them, an env venv's own decoder reads them, and this is the module both
# sides import.
CARTESIAN = 'cartesian'
CARTESIAN_DELTA = 'cartesian_delta'
JOINT_POS = 'joint_pos'
JOINT_VEL = 'joint_vel'
HOLD = 'hold'
CANONICAL_COMMAND_TYPES = (CARTESIAN, CARTESIAN_DELTA, JOINT_POS, JOINT_VEL, HOLD)

# The action a client puts on the wire: the tagged arm command, and the gripper closure alongside it.
ACTION_COMMAND = 'command'
ACTION_GRIP = 'grip'

# The tagged command's own fields: the tag, the one value each tag carries (``hold`` carries none), and the
# control law the command pins.
COMMAND_TYPE = 'type'
COMMAND_POSE = 'pose'  # CARTESIAN — an absolute pose, [t(3), R(9)]
COMMAND_DELTA = 'delta'  # CARTESIAN_DELTA — a relative pose, same encoding
COMMAND_JOINT_POS = 'q'  # JOINT_POS — absolute joint targets
COMMAND_JOINT_VEL = 'dq'  # JOINT_VEL — per-step joint deltas
COMMAND_MODE = 'mode'  # any tag — the pinned control mode, absent when the command pins none

# The address every env-server script is spawned with: its launcher builds the command in positronic's
# interpreter, its ``env.py`` parser declares it in the adoption's own, so a rename that misses one side
# fails at spawn rather than at import.
OPT_HOST = '--host'
OPT_PORT = '--port'

# The request envelope: the verb the client sends, the arguments each verb carries, and the replies the
# server writes itself (every other reply is an env's own frame).
REQUEST_CMD = 'cmd'
REQUEST_SPEC = 'spec'  # CMD_TASKS — the eval config's task selection
REQUEST_TOKEN = 'token'  # CMD_RESET — the adapter's reset token
REQUEST_ACTION = 'action'  # CMD_STEP — the tagged command plus grip
CMD_TASKS = 'tasks'
CMD_RESET = 'reset'
CMD_STEP = 'step'
CMD_CLOSE = 'close'
RESPONSE_TASKS = 'tasks'  # CMD_TASKS — the env's own task records
RESPONSE_OK = 'ok'  # CMD_CLOSE's acknowledgement
RESPONSE_ERROR = 'error'  # any verb: the server caught an exception and the client re-raises it

# The frames an env reports back. ``reset`` carries the observation, the scene meta, the robot model identity
# and the control period; ``step`` carries the observation, the terminal, the control period, and — where the
# env judges one — its success. ``horizon`` is the episode limit the env enforces itself, in sim-seconds,
# absent when the env enforces none.
FRAME_OBS = 'obs'
FRAME_META = 'meta'
FRAME_ROBOT_META = 'robot_meta'
FRAME_CONTROL_DT = 'control_dt'
FRAME_HORIZON = 'horizon'
FRAME_DONE = 'done'
FRAME_SUCCESS = 'success'


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
        # ``bytearray`` (not the raw msgpack ``bytes``) backs a writable array, so the socket path
        # matches the in-process path for envs/adapters that mutate a decoded buffer in place.
        return np.ndarray(buffer=bytearray(obj[b'data']), dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])
    return obj


def encode(obj: Any) -> bytes:
    """*obj* as a msgpack frame, with the numpy envelope applied to arrays and scalars inside it."""
    # ``packb`` is annotated ``bytes | None``, for a streaming mode this call does not use; every caller here
    # hands the result straight to a socket, so the frame is materialized.
    packed = msgpack.packb(obj, default=_pack)
    assert packed is not None
    return packed


decode = functools.partial(msgpack.unpackb, object_hook=_unpack)
