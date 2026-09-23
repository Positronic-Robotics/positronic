"""A policy served by a roboarena server, with the DROID codec in front of it.

The server announces its `PolicyServerConfig` on connect. Each episode reads that config and builds its codec
and the observation keys it sends from it.

FOOTGUN: the server rejects an observation carrying a key it did not ask for, and one missing a key it did, so
the keys come from the announced config and the message is held to them before it is sent.
"""

import uuid
from collections.abc import Mapping
from contextlib import closing
from threading import Lock
from typing import Any

import numpy as np
from positronic_wire import roboarena as roboarena_wire

from positronic.offboard.roboarena import RoboarenaClient
from positronic.policy import Policy, PolicyRun, Runtime, Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.codec import ACTION
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable
from positronic.vendors.dreamzero import codecs, roboarena

# The action space this codec decodes: seven absolute joint positions and a gripper. A server announcing
# another one answers a chunk the codec would read as joints and the arm would execute.
JOINT_POSITION_SPACE = 'joint_position'

# How many exterior images `CODEC` writes. A server asking for more names a key the codec cannot fill, so the
# handshake refuses the count rather than building a key set no observation satisfies.
EXTERIOR_IMAGES = 2


def wanted_keys(config: Mapping[str, Any]) -> frozenset[str]:
    """Exactly the observation keys the announced `config` asks for."""
    if config[roboarena.NEEDS_STEREO_CAMERA]:
        raise ValueError('the roboarena server asks for stereo cameras, which this policy does not send')
    if config[roboarena.ACTION_SPACE] != JOINT_POSITION_SPACE:
        raise ValueError(
            f'the roboarena server answers {config[roboarena.ACTION_SPACE]!r} actions, and this codec decodes '
            f'{JOINT_POSITION_SPACE!r}; a chunk read in the wrong space moves the arm wrongly'
        )
    keys = {roboarena.JOINT_POSITION, roboarena.GRIPPER_POSITION, roboarena.PROMPT}
    if config[roboarena.NEEDS_WRIST_CAMERA]:
        keys.add(roboarena.WRIST_IMAGE)
    exteriors = config[roboarena.NUM_EXTERIOR_CAMERAS]
    if not 0 <= exteriors <= EXTERIOR_IMAGES:
        raise ValueError(
            f'the roboarena server asks for {exteriors} exterior cameras, and this codec writes '
            f'{EXTERIOR_IMAGES}; every observation would miss a key that server requires'
        )
    keys.update(exterior_camera(i) for i in range(exteriors))
    if config[roboarena.NEEDS_SESSION_ID]:
        keys.add(roboarena.SESSION_ID)
    return frozenset(keys)


def exterior_camera(index: int) -> str:
    """The key a roboarena server reads the `index`-th exterior camera under, counting `index` from 0.

    A roboarena server numbers the exterior cameras from 1, and the codec writes them numbered from 0, the way
    DreamZero's own server reads them.
    """
    return roboarena.exterior_image(index + 1)


def renaming() -> dict[str, str]:
    """What each exterior key the codec writes is called on the wire.

    FOOTGUN: every slot the codec writes is renamed, whatever the server asked for. Renaming only the asked-for
    ones leaves slot 1 under slot 0's wire name, and one silently overwrites the other before the message is
    filtered.
    """
    return {roboarena.exterior_image(i): exterior_camera(i) for i in range(EXTERIOR_IMAGES)}


def image_size(config: Mapping[str, Any]) -> tuple[int, int]:
    """The (width, height) to encode frames at, from the announced `config`.

    The protocol states the resolution as (height, width). The server rejects a frame of any other size.
    """
    resolution = config[roboarena.RESOLUTION]
    if resolution is None:
        raise ValueError('the roboarena server announced no image resolution, so there is no size to encode at')
    height, width = resolution
    return int(width), int(height)


# `droid` repeats the first exterior view in the second slot, and a server reading two different
# over-shoulder views is sent two.
CODEC = codecs.droid_3cam

# The configuronic path of the codec setting that the announced resolution reaches.
IMAGE_SIZE_OVERRIDE = 'obs.image_size'


def local_stack(config: Mapping[str, Any]) -> Sequential:
    """The stack in front of a server announcing `config`: the codec, and a schedule at its training cadence.

    `ChunkedSchedule` plays each chunk from the moment it arrives. `PauseOnUnavailable` holds both while the arm
    is unavailable.
    """
    codec = CODEC.override(**{IMAGE_SIZE_OVERRIDE: image_size(config)}).instantiate()
    return Sequential(PauseOnUnavailable(), ChunkedSchedule(fps=codec.meta[policy_keys.ACTION_FPS]), codec)


# The key the chunk arrives under in the server's reply, which is a mapping rather than a bare array.
ACTIONS_FIELD = 'actions'

# How many values one action in `JOINT_POSITION_SPACE` carries: seven joints and a gripper. A row of another
# width is read by the codec as joints anyway, and the arm executes it.
JOINT_POSITION_WIDTH = 8


class RoboarenaEndpoint:
    """The inference function one episode submits: an observation in, the chunk the server answers."""

    def __init__(self, client: RoboarenaClient, config: Mapping[str, Any]):
        self._client = client
        self._keys = wanted_keys(config)
        self._renames = renaming()
        # A stateful server tells episodes apart by this id, so each episode gets its own.
        self._session_id = str(uuid.uuid4()) if config[roboarena.NEEDS_SESSION_ID] else ''

    def _message(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """`obs` in the wire's own names, holding the announced keys and no others."""
        named = {self._renames.get(key, key): value for key, value in obs.items()}
        if self._session_id:
            named[roboarena.SESSION_ID] = self._session_id
        sent = {key: value for key, value in named.items() if key in self._keys}
        if missing := sorted(self._keys - sent.keys()):
            raise ValueError(f'the roboarena server asked for {missing}, which this observation does not carry')
        return sent

    def _answer(self, obs: Mapping[str, Any]) -> np.ndarray:
        try:
            reply = self._client.infer(self._message(obs))
        except roboarena_wire.TextAnswer as e:
            raise RuntimeError(
                f'the roboarena server at {self._client.url} answered an error instead of an action chunk: {e.text}'
            ) from e
        return np.asarray(reply[ACTIONS_FIELD])

    def __call__(self, obs: Mapping[str, Any]) -> list[dict[str, Any]]:
        chunk = self._answer(obs)
        # One action, or a chunk of them. The codec decodes one row at a time either way.
        rows = chunk[np.newaxis, :] if chunk.ndim == 1 else chunk
        if rows.ndim != 2 or rows.shape[-1] != JOINT_POSITION_WIDTH:
            raise ValueError(
                f'the roboarena server answered a chunk of shape {rows.shape} this codec cannot decode. One '
                f'{JOINT_POSITION_SPACE!r} action is {JOINT_POSITION_WIDTH} values, and a row of another width '
                'reaches the arm as joint positions'
            )
        if rows.dtype.kind not in 'fiu' or not np.isfinite(rows).all():
            raise ValueError(
                'the roboarena server answered an action that is not a finite number, and the codec decodes '
                'such a value into a joint position the arm then holds'
            )
        return [{ACTION: row} for row in rows]


class RoboarenaPolicy(Policy):
    """A policy served by the roboarena server at `address`, with the DROID codec in front of it.

    Each episode opens its own connection. The codec's geometry and the cameras it sends come from the config
    the server announces on that connection, so the stack is built when the episode starts.
    """

    def __init__(self, address: roboarena_wire.RoboarenaAddress):
        self._address = address

    def run(self, runtime: Runtime) -> PolicyRun:
        client = RoboarenaClient(self._address.host, self._address.port)
        # Held by each inference, so a failure that closes the episode waits for the one in flight.
        connection_lock = Lock()
        config = client.connect()
        try:
            endpoint = RoboarenaEndpoint(client, config)

            def infer(obs: Mapping[str, Any]) -> list[dict[str, Any]]:
                with connection_lock:
                    return endpoint(obs)

            with closing(runtime.start(local_stack(config), infer)) as stack:
                obs = yield
                while True:
                    try:
                        step = stack.send(obs)
                    except StopIteration:
                        return
                    obs = yield step
        finally:
            with connection_lock:
                client.close()
