import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, TypeAlias

import pimm
from positronic.utils import frozen_keys_dict

from .dataset import DatasetWriter
from .episode import META_PATH, EpisodeWriter
from .serializers import Serializer, StatefulSerializer, _PureSerializer, expand_suffixed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# A factory of context managers the caller brackets each record-flush I/O section with (default inert).
ContextFactory: TypeAlias = Callable[[], AbstractContextManager[Any]]

# A factory of dataset writers, one per path a command names.
DatasetFactory: TypeAlias = Callable[[Path], DatasetWriter]


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
        output_path: Which dataset `START_EPISODE` records into. `None` records nowhere: the episode runs
            and nothing is written. The other actions name no dataset.
        static_data: Optional static key/value pairs to set on the episode
            when starting or right before stopping.
    """

    type: DsWriterCommandType
    output_path: Path | None = None
    static_data: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def START(output_path: Path | None, static_data: dict[str, Any] | None = None):
        return DsWriterCommand(DsWriterCommandType.START_EPISODE, output_path, static_data or {})

    @staticmethod
    def STOP(static_data: dict[str, Any] | None = None):
        return DsWriterCommand(DsWriterCommandType.STOP_EPISODE, static_data=static_data or {})

    @staticmethod
    def ABORT():
        return DsWriterCommand(DsWriterCommandType.ABORT_EPISODE)


class TimeMode(IntEnum):
    """Mode of timestamping for the dataset writer."""

    CLOCK = 0
    MESSAGE = 1


class _Recording:
    """The episode a run has open, and the datasets it opened to hold them.

    One command moves it: a START opens an episode, a STOP finishes it, an ABORT discards it. An episode a
    START named no dataset for records nowhere, and its ``writer`` stays ``None``. A command that does not fit
    the state it finds is ignored with a log message.
    """

    def __init__(self, dataset_factory: DatasetFactory, serializers: dict[str, StatefulSerializer]):
        self._dataset_factory = dataset_factory
        self._serializers = serializers
        self._datasets: dict[Path, DatasetWriter] = {}
        self._open_datasets = ExitStack()
        self._episode_open = False
        self.writer: EpisodeWriter | None = None

    def handle(self, cmd: DsWriterCommand) -> None:
        match cmd.type, self._episode_open:
            case DsWriterCommandType.START_EPISODE, False:
                self._start(cmd)
            case DsWriterCommandType.START_EPISODE, True:
                logger.warning('Episode already started, ignoring start command')
            case _, False:
                logger.warning(f'Episode not started, ignoring {cmd.type.value} command')
            case DsWriterCommandType.STOP_EPISODE, True:
                self._stop(cmd.static_data)
            case _, True:
                self._abort()

    def close(self) -> None:
        """Discard an episode still open, and close every dataset this recording opened."""
        self._abort()
        self._open_datasets.close()

    def _start(self, cmd: DsWriterCommand) -> None:
        """Open an episode in the dataset the command names, stamped with its static data."""
        self._episode_open = True
        if cmd.output_path is None:
            return
        for ser in self._serializers.values():
            ser.reset()
        if cmd.output_path not in self._datasets:  # a dataset numbers its episodes, off the disk it holds
            ds_writer = self._dataset_factory(cmd.output_path)
            self._datasets[cmd.output_path] = self._open_datasets.enter_context(ds_writer)
        self.writer = self._datasets[cmd.output_path].new_episode()
        self._set_statics(self.writer, cmd.static_data)
        logger.info(f'DsWriterAgent: [START] {self._episode_path(self.writer)}')

    def _stop(self, static_data: dict[str, Any]) -> None:
        """Finish the open episode, stamped with the command's static data."""
        self._episode_open = False
        if (writer := self.writer) is None:
            return
        self._set_statics(writer, static_data)
        writer.__exit__(None, None, None)
        logger.info(f'DsWriterAgent: [STOP] {self._episode_path(writer)}')
        self.writer = None

    def _abort(self) -> None:
        """Discard the open episode."""
        self._episode_open = False
        if (writer := self.writer) is None:
            return
        try:
            writer.abort()
        finally:  # the writer holds a file handle per signal; release them even if the discard fails
            writer.__exit__(None, None, None)
        logger.info(f'DsWriterAgent: [ABORT] {self._episode_path(writer)}')
        self.writer = None

    @staticmethod
    def _set_statics(writer: EpisodeWriter, static_data: dict[str, Any]) -> None:
        for k, v in static_data.items():
            writer.set_static(k, v)

    @staticmethod
    def _episode_path(writer: EpisodeWriter) -> str:
        """Where the episode is written, as its own writer reports it."""
        return writer.meta.get(META_PATH, 'unknown')


class DsWriterAgent(pimm.ControlSystem):
    """Streams input signals into episodes based on control commands.

    Listens on `command` for `DsWriterCommand` messages controlling the
    episode lifecycle.

    On `START_EPISODE`, opens a new `EpisodeWriter` in the dataset the command names — ``dataset_factory``
    opens that dataset, once per name — and applies `static_data`. A `START_EPISODE` that names no dataset
    opens an episode that records nowhere. The opening turn records what each
    input holds, whenever that value was produced. While open, each updated input
    signal (from `inputs`) is appended with the current timestamp from `clock`.
    `STOP_EPISODE` and `ABORT_EPISODE` are handled after that turn's inputs, so
    the trial's last frame is recorded before STOP finalizes the writer; samples
    timestamped after STOP — whatever the next trial's prepare moves, or sensor
    data the async real path queues — are dropped, and ABORT discards the
    episode. Invalid or out-of-order commands are ignored with a log message.

    `TimeMode` selects whether timestamps come from the control loop clock
    (`CLOCK`) or from the producing message (`MESSAGE`).

    ``virtual_time`` makes the recorder yield to ride the producer's clock — sim lockstep, where the
    simulator is the sole time-master — instead of pacing itself at ``poll_hz`` (real/background).
    """

    def __init__(
        self,
        dataset_factory: DatasetFactory,
        poll_hz: float = 1000.0,
        time_mode: TimeMode = TimeMode.CLOCK,
        virtual_time: bool = False,
        # rules-allow: misleading-name — timing is the only use, and a name for the mechanism alone reads as
        # a context of unsaid purpose. It generalises when a second kind of caller arrives.
        telemetry_span: ContextFactory = nullcontext,
    ):
        # A picklable factory, never a lambda: the recorder may be spawned as a background process, and it
        # opens its datasets there.
        self._dataset_factory = dataset_factory
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

    def _record_window(self, ep_writer: EpisodeWriter, clock: pimm.Clock, before: int | None, opening: bool):
        """Append this turn's input samples, dropping any stamped after ``before``.

        The opening turn reads every channel outright rather than only what arrived, so a channel silent
        since the last episode still contributes what it holds.
        """
        for name, reader in self._inputs.items():
            msg = reader.read() if opening else pimm.read_updated(reader)
            if msg is not None and (before is None or msg.ts <= before):
                self._record(ep_writer, name, msg, clock)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        """Main loop: process commands and append updated inputs to the episode."""
        limiter = pimm.utils.RateLimiter(clock, hz=self._poll_hz)
        pace = (lambda: pimm.Yield()) if self._virtual_time else limiter.wait
        recording = _Recording(self._dataset_factory, self._serializers)
        sampled: EpisodeWriter | None = None  # the writer this loop last took a window for

        try:
            while not should_stop.value:
                cmd = pimm.read_updated(self.command)
                stop = None
                if cmd is not None:
                    if cmd.data.type == DsWriterCommandType.START_EPISODE:
                        recording.handle(cmd.data)
                    else:
                        stop = cmd

                if recording.writer is not None:
                    before = stop.ts if stop is not None else None
                    self._record_window(recording.writer, clock, before, recording.writer is not sampled)
                    sampled = recording.writer

                if stop is not None:
                    recording.handle(stop.data)

                yield pace()

            if (cmd := pimm.read_updated(self.command)) is not None:
                recording.handle(cmd.data)
        finally:
            recording.close()
