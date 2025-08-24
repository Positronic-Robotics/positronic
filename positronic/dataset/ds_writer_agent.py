from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import collections.abc as cabc

import pimm

from .core import DatasetWriter, EpisodeWriter


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


class DsWriterAgent:
    """Streams input signals into episodes based on control commands.

    Listens on `command` for `DsWriterCommand` messages controlling the
    episode lifecycle. On `START_EPISODE`, opens a new `EpisodeWriter` from
    the provided `DatasetWriter` and applies `static_data`. While an episode
    is open, any updated input signal (from `inputs`) is appended with the
    current timestamp from `clock`. `STOP_EPISODE` finalizes the writer after
    applying `static_data`; `ABORT_EPISODE` aborts and discards it. Invalid or
    out-of-order commands are ignored with a log message.
    """
    command: pimm.SignalReader[DsWriterCommand] = pimm.NoOpReader()

    def __init__(self, ds_writer: DatasetWriter, signal_names: list[str], poll_hz: float = 1000.0):
        self.ds_writer = ds_writer
        self._poll_hz = float(poll_hz)

        self._inputs: dict[str, pimm.SignalReader[Any]] = {
            name: pimm.NoOpReader[Any]() for name in (signal_names or [])
        }
        self._inputs_view = _KeyFrozenMapping(self._inputs)

    @property
    def inputs(self) -> dict[str, pimm.SignalReader[Any]]:
        # Expose a mapping with frozen keys; values can be updated for existing keys.
        return self._inputs_view  # type: ignore[return-value]

    def run(self, should_stop: pimm.SignalReader, clock: pimm.Clock):
        limiter = pimm.utils.RateLimiter(clock, hz=self._poll_hz)
        commands = pimm.DefaultReader(pimm.ValueUpdated(self.command), (None, False))

        signals = {
            name: pimm.DefaultReader(pimm.ValueUpdated(reader), (None, False))
            for name, reader in self._inputs.items()
        }
        ep_writer: EpisodeWriter | None = None

        while not should_stop.value:
            cmd, cmd_updated = commands.value
            if cmd_updated:
                cmd: DsWriterCommand
                match cmd.type:
                    case DsWriterCommandType.START_EPISODE:
                        if ep_writer is None:
                            ep_writer = self.ds_writer.new_episode()
                            for k, v in cmd.static_data.items():
                                ep_writer.set_static(k, v)
                        else:
                            print("Episode already started, ignoring start command")
                    case DsWriterCommandType.STOP_EPISODE:
                        if ep_writer is not None:
                            for k, v in cmd.static_data.items():
                                ep_writer.set_static(k, v)
                            ep_writer.__exit__(None, None, None)
                            ep_writer = None
                        else:
                            print("Episode not started, ignoring stop command")
                    case DsWriterCommandType.ABORT_EPISODE:
                        if ep_writer is not None:
                            ep_writer.abort()
                            ep_writer.__exit__(None, None, None)
                            ep_writer = None
                        else:
                            print("Episode not started, ignoring abort command")

            if ep_writer is not None:
                for name, reader in signals.items():
                    value, updated = reader.value
                    if updated:
                        ep_writer.append(name, value, clock.now_ns())

            yield pimm.Sleep(limiter.wait_time())


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
