import collections.abc as cabc
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable

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
    START_EPISODE = "start_episode"
    STOP_EPISODE = "stop_episode"
    ABORT_EPISODE = "abort_episode"


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


# Serializer contract for inputs:
# - If None is provided for a signal name in signals_spec, the value is passed through unchanged.
# - If a callable is provided, it is invoked as serializer(value) and can return:
#     * a transformed value -> recorded under the same signal name
#     * a dict mapping suffix -> value -> expands into multiple signals recorded as name+suffix
#         - use "" (empty string) to keep the base name as-is
#         - any dict entry with value None is skipped (not recorded)
#     * None -> the sample is dropped (not recorded)
# - Serializers may expose a ``names`` attribute:
#     * string or list[str]: metadata for the base signal
#     * dict[str, str | list[str]]: metadata per derived signal suffix ("" keeps the base name)
Serializer = Callable[[Any], Any | dict[str, Any]]


def names(metadata: str | list[str] | dict[str, str | list[str]]):
    """Attach feature-name metadata to a serializer function."""

    def _decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        setattr(fn, 'names', metadata)
        return fn

    return _decorator


class Serializers:
    """Namespace of built-in serializers for convenience.

    Usage:
        from positronic.dataset.ds_writer_agent import Serializers
        spec = {"ee_pose": Serializers.transform_3d}
        agent = DsWriterAgent(writer, spec)

    Notes:
        - Also exported as `Serializers` (US spelling) for convenience.
    """

    @staticmethod
    @names(['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    def transform_3d(x: geom.Transform3D) -> np.ndarray:
        """Serialize a Transform3D into a 7D vector [tx, ty, tz, qx, qy, qz, qw]."""
        return np.concatenate([x.translation, x.rotation.as_quat])

    @staticmethod
    @names({'.q': 'joints', '.dq': 'joint velocities', '.ee_pose': ['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']})
    def robot_state(state: roboarm.State) -> dict[str, np.ndarray] | None:
        if state.status == roboarm.RobotStatus.RESETTING:
            return None
        return {
            '.q': state.q,
            '.dq': state.dq,
            '.ee_pose': Serializers.transform_3d(state.ee_pose),
        }

    @staticmethod
    @names({'.pose': 'target pose', '.joints': 'target joints'})
    def robot_command(command: roboarm.command.CommandType) -> dict[str, np.ndarray | int] | None:
        match command:
            case roboarm.command.CartesianMove(pose):
                return {'.pose': Serializers.transform_3d(pose)}
            case roboarm.command.JointMove(positions):
                return {'.joints': positions}
            case roboarm.command.Reset():
                return {'.reset': 1}


def _extract_names(serializer: Callable[[Any], Any]) -> dict[str, list[str]] | None:
    names = getattr(serializer, 'names', None)
    if names is None:
        return None
    if isinstance(names, str):
        return {'': [names]}
    if isinstance(names, list):
        return {'': names}
    if isinstance(names, dict):
        out: dict[str, list[str]] = {}
        for suffix, value in names.items():
            key = '' if suffix in ('', None) else suffix
            if isinstance(value, str):
                out[key] = [value]
            elif isinstance(value, list):
                out[key] = value
            else:
                raise TypeError("Serializer names mapping values must be string or list")
        return out
    raise TypeError("Serializer names attribute must be a string, sequence, or mapping")


def _append_processed(ep_writer: EpisodeWriter, name: str, value: Any, ts_ns: int) -> None:
    if isinstance(value, dict):
        items = ((name + suffix, v) for suffix, v in value.items())
    else:
        items = ((name, value), )

    for full_name, v in items:
        if v is None:
            continue
        ep_writer.append(full_name, v, ts_ns)


class TimeMode(IntEnum):
    """Mode of timestamping for the dataset writer."""
    CLOCK = 0
    MESSAGE = 1


class DsWriterAgent(pimm.ControlSystem):
    """Streams input signals into episodes based on control commands.

    Listens on `command` for `DsWriterCommand` messages controlling the
    episode lifecycle. On `START_EPISODE`, opens a new `EpisodeWriter` from
    the provided `DatasetWriter` and applies `static_data`. While an episode
    is open, any updated input signal (from `inputs`) is appended with the
    current timestamp from `clock`. `STOP_EPISODE` finalizes the writer after
    applying `static_data`; `ABORT_EPISODE` aborts and discards it. Invalid or
    out-of-order commands are ignored with a log message.
    """

    def __init__(self,
                 ds_writer: DatasetWriter,
                 signals_spec: dict[str, Serializer | None],
                 poll_hz: float = 1000.0,
                 time_mode: TimeMode = TimeMode.CLOCK):
        self.ds_writer = ds_writer
        self._poll_hz = float(poll_hz)
        self._time_mode = time_mode
        self.command = pimm.ControlSystemReceiver[DsWriterCommand](self)

        self._inputs: dict[str, pimm.ControlSystemReceiver[Any]] = {
            name: pimm.ControlSystemReceiver[Any](self)
            for name in (signals_spec or [])
        }
        self._inputs_view = _KeyFrozenMapping(self._inputs)
        # Only keep explicitly provided serializers; None means pass-through
        self._serializers = {name: serializer for name, serializer in signals_spec.items() if serializer is not None}
        self._signal_meta_specs: dict[str, dict[str, list[str]]] = {}
        for name, serializer in self._serializers.items():
            names = _extract_names(serializer)
            if names is not None:
                self._signal_meta_specs[name] = names

    @property
    def inputs(self) -> dict[str, pimm.ControlSystemReceiver[Any]]:
        # Expose a mapping with frozen keys; values can be updated for existing keys.
        return self._inputs_view  # type: ignore[return-value]

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        """Main loop: process commands and append updated inputs to the episode.

        Refactored to reduce complexity: command handling, serialization, and
        appending are split into helpers.
        """
        limiter = pimm.utils.RateLimiter(clock, hz=self._poll_hz)
        commands = pimm.DefaultReceiver(pimm.ValueUpdated(self.command), (None, False))

        signals = {
            name: pimm.DefaultReceiver(pimm.ValueUpdated(reader), (None, False))
            for name, reader in self._inputs.items()
        }
        ep_writer: EpisodeWriter | None = None
        ep_counter = 0

        try:
            while not should_stop.value:
                cmd, cmd_updated = commands.value
                if cmd_updated:
                    ep_writer, ep_counter = self._handle_command(cmd, ep_writer, ep_counter)

                if ep_writer is not None:
                    for name, reader in signals.items():
                        msg = reader.read()
                        value, updated = msg.data
                        if updated:
                            serializer = self._serializers.get(name)
                            if serializer is not None:
                                value = serializer(value)
                            time_ns = clock.now_ns() if self._time_mode == TimeMode.CLOCK else msg.ts
                            _append_processed(ep_writer, name, value, time_ns)

                yield pimm.Sleep(limiter.wait_time())
        finally:
            if ep_writer is not None:
                ep_writer.__exit__(None, None, None)

    def _handle_command(self, cmd: DsWriterCommand, ep_writer: EpisodeWriter | None,
                        ep_counter: int) -> tuple[EpisodeWriter | None, int]:
        match cmd.type:
            case DsWriterCommandType.START_EPISODE:
                if ep_writer is None:
                    ep_counter += 1
                    print(f"DsWriterAgent: [START] Episode {ep_counter}")
                    ep_writer = self.ds_writer.new_episode()
                    for base_name, names_map in self._signal_meta_specs.items():
                        for suffix, name_list in names_map.items():
                            ep_writer.set_signal_meta(base_name + suffix, names=name_list)
                    for k, v in cmd.static_data.items():
                        ep_writer.set_static(k, v)
                else:
                    print("Episode already started, ignoring start command")
            case DsWriterCommandType.STOP_EPISODE:
                if ep_writer is not None:
                    for k, v in cmd.static_data.items():
                        ep_writer.set_static(k, v)
                    ep_writer.__exit__(None, None, None)
                    print(f"DsWriterAgent: [STOP] Episode {ep_counter}")
                    ep_writer = None
                else:
                    print("Episode not started, ignoring stop command")
            case DsWriterCommandType.ABORT_EPISODE:
                if ep_writer is not None:
                    ep_writer.abort()
                    ep_writer.__exit__(None, None, None)
                    print(f"DsWriterAgent: [ABORT] Episode {ep_counter}")
                    ep_writer = None
                else:
                    print("Episode not started, ignoring abort command")
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
            raise TypeError("inputs keys are frozen; cannot add new key")
        self._backing[key] = value

    def __delitem__(self, key):
        raise TypeError("inputs keys are frozen; cannot delete keys")

    def __iter__(self):
        return iter(self._backing)

    def __len__(self):
        return len(self._backing)

    def __repr__(self) -> str:
        return f"KeyFrozenMapping({self._backing!r})"
