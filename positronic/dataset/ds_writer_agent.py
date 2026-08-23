import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, TypeAlias

import pimm
from positronic.utils import frozen_keys_dict

from .dataset import DatasetWriter
from .episode import EpisodeWriter
from .serializers import Serializer, StatefulSerializer, _PureSerializer, expand_suffixed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# A factory of context managers the caller brackets each record-flush I/O section with (default inert).
ContextFactory: TypeAlias = Callable[[], AbstractContextManager[Any]]


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


class TimeMode(IntEnum):
    """Mode of timestamping for the dataset writer."""

    CLOCK = 0
    MESSAGE = 1


class DsWriterAgent(pimm.ControlSystem):
    """Streams input signals into episodes based on control commands.

    Listens on `command` for `DsWriterCommand` messages controlling the
    episode lifecycle.

    On `START_EPISODE`, opens a new `EpisodeWriter` from the provided
    `DatasetWriter` and applies `static_data`. On the opening turn, samples that
    predate START — the previous episode's last command and any pre-reset frame —
    are dropped, while a genuine post-START value is kept; the producer's post-reset
    frame-0 arrives the next turn and is the first recorded sample. While open,
    each updated input signal (from `inputs`) is appended with the current
    timestamp from `clock`. `STOP_EPISODE` and `ABORT_EPISODE` are handled after
    that turn's inputs, so the trial's last frame is recorded before STOP
    finalizes the writer; samples timestamped after STOP — whatever the next
    trial's prepare moves, or sensor data the async real path queues — are
    dropped, and ABORT discards
    the episode. Invalid or out-of-order commands are ignored with a log message.

    `TimeMode` selects whether timestamps come from the control loop clock
    (`CLOCK`) or from the producing message (`MESSAGE`).

    ``virtual_time`` makes the recorder yield to ride the producer's clock — sim lockstep, where the
    simulator is the sole time-master — instead of pacing itself at ``poll_hz`` (real/background).
    """

    def __init__(
        self,
        ds_writer: DatasetWriter,
        poll_hz: float = 1000.0,
        time_mode: TimeMode = TimeMode.CLOCK,
        virtual_time: bool = False,
        # rules-allow: misleading-name — timing is the only use, and a name for the mechanism alone reads as
        # a context of unsaid purpose. It generalises when a second kind of caller arrives.
        telemetry_span: ContextFactory = nullcontext,
    ):
        self.ds_writer = ds_writer
        self._poll_hz = float(poll_hz)
        self._time_mode = time_mode
        self._virtual_time = virtual_time
        # An opaque context factory wrapped around the writer's serialize+append work; the default is
        # inert. The caller decides what it brackets — the writer never learns.
        self._telemetry_span = telemetry_span
        self.command = pimm.ControlSystemReceiver[DsWriterCommand](self)

        self._inputs: dict[str, pimm.ControlSystemReceiver[Any]] = {}
        self._serializers: dict[str, StatefulSerializer] = {}

    def add_signal(self, name: str, serializer: Serializer | StatefulSerializer | None = None):
        self._inputs[name] = pimm.ControlSystemReceiver[Any](self)
        if serializer is not None:
            if not isinstance(serializer, StatefulSerializer):
                serializer = _PureSerializer(serializer)
            self._serializers[name] = serializer

    @property
    def inputs(self) -> dict[str, pimm.ControlSystemReceiver[Any]]:
        return frozen_keys_dict(self._inputs)

    def _record(self, ep_writer: EpisodeWriter, name: str, msg: pimm.Message, clock: pimm.Clock) -> None:
        """Append one input's sample, stamped as ``time_mode`` selects and carrying every clock beside it."""
        world_time_ns, message_time_ns = clock.now_ns(), msg.ts
        primary_ts = world_time_ns if self._time_mode == TimeMode.CLOCK else message_time_ns

        extra_ts = {'message': message_time_ns, 'system': pimm.world.SystemClock().now_ns()}
        # Only add 'world' if clock is not system clock
        if not isinstance(clock, pimm.world.SystemClock):
            extra_ts['world'] = world_time_ns

        with self._telemetry_span():
            serializer = self._serializers.get(name)
            value = msg.data
            if serializer is not None:
                value = serializer(value)
            for full_name, v in expand_suffixed(name, value):
                if v is not None:
                    ep_writer.append(full_name, v, primary_ts, extra_ts)

    def _record_window(self, ep_writer: EpisodeWriter, clock: pimm.Clock, after: int | None, before: int | None):
        """Append every input sample delivered this turn that falls inside ``(after, before]``."""
        for name, reader in self._inputs.items():
            msg = pimm.read_updated(reader)
            if msg is not None and (after is None or msg.ts > after) and (before is None or msg.ts <= before):
                self._record(ep_writer, name, msg, clock)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        """Main loop: process commands and append updated inputs to the episode."""
        limiter = pimm.utils.RateLimiter(clock, hz=self._poll_hz)
        pace = (lambda: pimm.Yield()) if self._virtual_time else limiter.wait
        ep_writer: EpisodeWriter | None = None
        ep_counter = 0

        try:
            while not should_stop.value:
                cmd = pimm.read_updated(self.command)
                # Open before draining this turn's inputs, but finalize/abort after them: the trial's last
                # frame, queued by the producer a control period earlier, is recorded before STOP closes the
                # writer.
                opened_at, stop_at, stop_cmd = None, None, None
                if cmd is not None:
                    if cmd.data.type == DsWriterCommandType.START_EPISODE:
                        was_open = ep_writer is not None
                        ep_writer, ep_counter = self._handle_command(cmd.data, ep_writer, ep_counter)
                        opened_at = cmd.ts if ep_writer is not None and not was_open else None
                    else:
                        stop_at, stop_cmd = cmd.ts, cmd.data

                if ep_writer is not None:
                    self._record_window(ep_writer, clock, opened_at, stop_at)

                if stop_cmd is not None:
                    ep_writer, ep_counter = self._handle_command(stop_cmd, ep_writer, ep_counter)

                yield pace()
        finally:
            cmd = pimm.read_updated(self.command)
            if cmd is not None:
                ep_writer, ep_counter = self._handle_command(cmd.data, ep_writer, ep_counter)

            if ep_writer is not None:
                try:
                    ep_writer.abort()
                finally:
                    ep_writer.__exit__(None, None, None)
                    logger.info(f'DsWriterAgent: [ABORT] Episode {ep_counter}')

    def _handle_command(self, cmd: DsWriterCommand, ep_writer: EpisodeWriter | None, ep_counter: int):
        match cmd.type:
            case DsWriterCommandType.START_EPISODE:
                if ep_writer is None:
                    ep_counter += 1
                    logger.info(f'DsWriterAgent: [START] Episode {ep_counter}')
                    for ser in self._serializers.values():
                        ser.reset()
                    ep_writer = self.ds_writer.new_episode()
                    for k, v in cmd.static_data.items():
                        ep_writer.set_static(k, v)
                else:
                    logger.warning('Episode already started, ignoring start command')
            case DsWriterCommandType.STOP_EPISODE:
                if ep_writer is not None:
                    for k, v in cmd.static_data.items():
                        ep_writer.set_static(k, v)
                    ep_writer.__exit__(None, None, None)
                    logger.info(f'DsWriterAgent: [STOP] Episode {ep_counter} {ep_writer.meta.get("path", "unknown")}')
                    ep_writer = None
                else:
                    logger.warning('Episode not started, ignoring stop command')
            case DsWriterCommandType.ABORT_EPISODE:
                if ep_writer is not None:
                    ep_writer.abort()
                    ep_writer.__exit__(None, None, None)
                    logger.info(f'DsWriterAgent: [ABORT] Episode {ep_counter}')
                    ep_writer = None
                else:
                    logger.warning('Episode not started, ignoring abort command')
        return ep_writer, ep_counter
