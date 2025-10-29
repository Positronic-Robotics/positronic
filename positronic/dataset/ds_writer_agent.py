import collections.abc as cabc
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

import numpy as np

import pimm
from positronic import geom
from positronic.drivers import roboarm

from .dataset import DatasetWriter
from .episode import EpisodeWriter


class DsWriterCommandType(Enum):
    """Episode lifecycle commands for the dataset writer.

    Supported values:
    - `START_EPISODE`: Open a new episode and apply provided static data.
    - `STOP_EPISODE`: Finalize the current episode, optionally updating static data.
    - `ABORT_EPISODE`: Abort and discard the current episode.
    """

    START_EPISODE = 'start_episode'
    STOP_EPISODE = 'stop_episode'
    ABORT_EPISODE = 'abort_episode'


@dataclass
class DsWriterCommand:
    """Command message consumed by `DsWriterAgent`.

    Args:
        type: Desired episode action (start/stop/abort).
        static_data: Optional static key/value pairs to set on the episode
            when starting or right before stopping.
    """

    type: DsWriterCommandType
    static_data: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def START(static_data: dict[str, Any] | None = None):
        return DsWriterCommand(DsWriterCommandType.START_EPISODE, static_data or {})

    @staticmethod
    def STOP(static_data: dict[str, Any] | None = None):
        return DsWriterCommand(DsWriterCommandType.STOP_EPISODE, static_data or {})

    @staticmethod
    def ABORT():
        return DsWriterCommand(DsWriterCommandType.ABORT_EPISODE)


# Serializer contract for inputs:
# - Register signals with `DsWriterAgent.add_signal(name, serializer=None)`.
# - If `serializer` is omitted or None, the value is passed through unchanged.
# - If a callable is provided, it is invoked as serializer(value) and can return:
#     * a transformed value -> recorded under the same signal name
#     * a dict mapping suffix -> value -> expands into multiple signals recorded as name+suffix
#         - use "" (empty string) to keep the base name as-is
#         - any dict entry with value None is skipped (not recorded)
#     * None -> the sample is dropped (not recorded)
Serializer = Callable[[Any], Any | dict[str, Any]]


class Serializers:
    """Namespace of built-in serializers for convenience.

    Usage:
        from positronic.dataset.ds_writer_agent import Serializers
        agent = DsWriterAgent(writer)
        agent.add_signal("ee_pose", Serializers.transform_3d)

    Notes:
        - Also exported as `Serializers` (US spelling) for convenience.
    """

    @staticmethod
    def transform_3d(x: geom.Transform3D) -> np.ndarray:
        """Serialize a Transform3D into a 7D vector [tx, ty, tz, qx, qy, qz, qw]."""
        return np.concatenate([x.translation, x.rotation.as_quat])

    @staticmethod
    def robot_state(state: roboarm.State) -> dict[str, np.ndarray] | None:
        if state.status == roboarm.RobotStatus.RESETTING:
            return None
        return {'.q': state.q, '.dq': state.dq, '.ee_pose': Serializers.transform_3d(state.ee_pose)}

    @staticmethod
    def robot_command(command: roboarm.command.CommandType) -> dict[str, np.ndarray | int] | None:
        match command:
            case roboarm.command.CartesianPosition(pose):
                return {'.pose': Serializers.transform_3d(pose)}
            case roboarm.command.JointPosition(positions):
                return {'.joints': positions}
            case roboarm.command.JointDelta(delta):
                return {'.joint_deltas': delta}
            case roboarm.command.Reset():
                return {'.reset': 1}

    @staticmethod
    def camera_images(data: pimm.shared_memory.NumpySMAdapter) -> np.ndarray:
        """Extract array from NumpySMAdapter for storage."""
        return data.array


def _append(ep_writer: EpisodeWriter, name: str, value: Any, ts_ns: int, extra_ts: dict[str, int] | None = None):
    if isinstance(value, dict):
        items = ((name + suffix, v) for suffix, v in value.items())
    else:
        items = ((name, value),)

    for full_name, v in items:
        if v is None:
            continue
        ep_writer.append(full_name, v, ts_ns, extra_ts)


class TimeMode(IntEnum):
    """Mode of timestamping for the dataset writer."""

    CLOCK = 0
    MESSAGE = 1


class DsWriterAgent(pimm.ControlSystem):
    """Streams input signals into episodes based on control commands.

    Listens on `command` for `DsWriterCommand` messages controlling the
    episode lifecycle.

    On `START_EPISODE`, opens a new `EpisodeWriter` from the provided
    `DatasetWriter` and applies `static_data`. While an episode is open, any
    updated input signal (from `inputs`) is appended with the current timestamp
    from `clock`. `STOP_EPISODE` finalizes the writer after applying
    `static_data`; `ABORT_EPISODE` aborts and discards it. Invalid or
    out-of-order commands are ignored with a log message.

    `TimeMode` selects whether timestamps come from the control loop clock
    (`CLOCK`) or from the producing message (`MESSAGE`).
    """

    def __init__(self, ds_writer: DatasetWriter, poll_hz: float = 1000.0, time_mode: TimeMode = TimeMode.CLOCK):
        self.ds_writer = ds_writer
        self._poll_hz = float(poll_hz)
        self._time_mode = time_mode
        self.command = pimm.ControlSystemReceiver[DsWriterCommand](self, default=None)

        self._inputs: dict[str, pimm.ControlSystemReceiver[Any]] = {}
        self._serializers: dict[str, Callable[[Any], Any | dict[str, Any]]] = {}

    def add_signal(self, name: str, serializer: Serializer | None = None):
        self._inputs[name] = pimm.ControlSystemReceiver[Any](self, default=None)
        if serializer is not None:
            self._serializers[name] = serializer

    @property
    def inputs(self) -> dict[str, pimm.ControlSystemReceiver[Any]]:
        return _KeyFrozenMapping(self._inputs)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        """Main loop: process commands and append updated inputs to the episode.

        Refactored to reduce complexity: command handling, serialization, and
        appending are split into helpers.
        """
        limiter = pimm.utils.RateLimiter(clock, hz=self._poll_hz)
        ep_writer: EpisodeWriter | None = None
        ep_counter = 0

        try:
            while not should_stop.value:
                cmd_msg = self.command.read()
                if cmd_msg.updated:
                    ep_writer, ep_counter = self._handle_command(cmd_msg.data, ep_writer, ep_counter)

                if ep_writer is not None:
                    for name, reader in self._inputs.items():
                        msg = reader.read()
                        if msg.updated:
                            world_time_ns, message_time_ns = clock.now_ns(), msg.ts
                            primary_ts = world_time_ns if self._time_mode == TimeMode.CLOCK else message_time_ns

                            extra_ts = {'message': message_time_ns, 'system': pimm.world.SystemClock().now_ns()}
                            # Only add 'world' if clock is not system clock
                            if not isinstance(clock, pimm.world.SystemClock):
                                extra_ts['world'] = world_time_ns

                            serializer = self._serializers.get(name)
                            value = msg.data
                            if serializer is not None:
                                value = serializer(value)
                            _append(ep_writer, name, value, primary_ts, extra_ts)

                yield pimm.Sleep(limiter.wait_time())
        finally:
            if ep_writer is not None:
                try:
                    ep_writer.abort()
                finally:
                    ep_writer.__exit__(None, None, None)
                    print(f'DsWriterAgent: [ABORT] Episode {ep_counter}')

    def _handle_command(self, cmd: DsWriterCommand, ep_writer: EpisodeWriter | None, ep_counter: int):
        match cmd.type:
            case DsWriterCommandType.START_EPISODE:
                if ep_writer is None:
                    ep_counter += 1
                    print(f'DsWriterAgent: [START] Episode {ep_counter}')
                    ep_writer = self.ds_writer.new_episode()
                    for k, v in cmd.static_data.items():
                        ep_writer.set_static(k, v)
                else:
                    print('Episode already started, ignoring start command')
            case DsWriterCommandType.STOP_EPISODE:
                if ep_writer is not None:
                    for k, v in cmd.static_data.items():
                        ep_writer.set_static(k, v)
                    ep_writer.__exit__(None, None, None)
                    print(f'DsWriterAgent: [STOP] Episode {ep_counter}')
                    ep_writer = None
                else:
                    print('Episode not started, ignoring stop command')
            case DsWriterCommandType.ABORT_EPISODE:
                if ep_writer is not None:
                    ep_writer.abort()
                    ep_writer.__exit__(None, None, None)
                    print(f'DsWriterAgent: [ABORT] Episode {ep_counter}')
                    ep_writer = None
                else:
                    print('Episode not started, ignoring abort command')
        return ep_writer, ep_counter


class _KeyFrozenMapping(cabc.MutableMapping):
    """Mapping wrapper that freezes the set of keys but allows updating values.

    Note: The Python stdlib has no built-in mapping that allows mutating
    values while preventing key additions/removals. `MappingProxyType` makes
    the entire mapping read-only, which isn't suitable here, so we provide a
    minimal wrapper to enforce "frozen keys, mutable values".

    - Setting a value for an existing key is allowed and updates the backing dict.
    - Adding a new key raises TypeError.
    - Deleting any key raises TypeError.
    """

    def __init__(self, backing: dict[str, Any]):
        self._backing = backing

    def __getitem__(self, key):
        return self._backing[key]

    def __setitem__(self, key, value):
        if key not in self._backing:
            raise TypeError('inputs keys are frozen; cannot add new key')
        self._backing[key] = value

    def __delitem__(self, key):
        raise TypeError('inputs keys are frozen; cannot delete keys')

    def __iter__(self):
        return iter(self._backing)

    def __len__(self):
        return len(self._backing)

    def __repr__(self) -> str:
        return f'KeyFrozenMapping({self._backing!r})'
