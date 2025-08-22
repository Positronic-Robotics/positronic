from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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
    command: pimm.SignalReader[DsWriterCommand]

    def __init__(self, ds_writer: DatasetWriter, signal_names: list[str], poll_hz: float = 1000.0):
        self.ds_writer = ds_writer
        self._poll_hz = float(poll_hz)

        Inputs = namedtuple("Inputs", signal_names)
        self.inputs = Inputs(*[pimm.NoOpReader[Any]() for _ in signal_names])

    def run(self, should_stop: pimm.SignalReader, clock: pimm.Clock):
        limiter = pimm.utils.RateLimiter(clock, hz=self._poll_hz)
        commands = pimm.DefaultReader(pimm.ValueUpdated(self.command), (None, False))

        signals = {
            name: pimm.DefaultReader(pimm.ValueUpdated(getattr(self.inputs, name)), (None, False))
            for name in self.inputs._fields
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
