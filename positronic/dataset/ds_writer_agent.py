import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

import pimm
from positronic.utils import frozen_keys_dict

from .dataset import DatasetWriter
from .episode import EpisodeWriter
from .serializers import Serializer, StatefulSerializer, Timestamped, _PureSerializer, expand_suffixed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


class TrajectoryOverrideSerializer(StatefulSerializer):
    """Flatten policy trajectories into a single monotonic per-point stream.

    A policy emits whole trajectories ``[(abs_ts_ns, value), ...]``. A newer
    trajectory overrides the overlapping tail of the previous one
    (last-writer-wins on the timeline): given ``1:[1..10]`` then ``2:[5..15]``
    the recorded stream is ``1@1..4`` then ``2@5..15``. A point is committed
    only once a newer trajectory starting after it proves it final; the
    remainder is drained by :meth:`flush` at episode end.

    Bare (non-trajectory) inputs — teleop single commands / scalar grip — pass
    straight through ``inner`` at the agent timestamp (legacy behaviour), so the
    shared ``wire.wire`` path keeps working for data collection and replay.

    HACK: lossy. Drops the notion of a *predicted* trajectory and cannot
    represent overlapping schedulers (RTC/temporal ensembling) that replan into
    the already-committed past — such points are dropped to keep timestamps
    strictly increasing. Faithful full-command recording needs an
    object-valued Signal (``Kind.OBJECT``); tracked in TODO(positronic#NNN).
    """

    def __init__(self, inner: Serializer | None):
        self._inner = inner
        self._buffer: list[tuple[int, Any]] = []  # latest trajectory, (abs_ts_ns, value), ts-sorted
        self._last_ts: int | None = None

    def reset(self) -> None:
        self._buffer = []
        self._last_ts = None

    def _encode(self, value: Any) -> Any:
        return self._inner(value) if self._inner is not None else value

    def _committable(self, points: list[tuple[int, Any]]) -> list[Timestamped]:
        # Guard only bites in the overlap-degrade case (RTC replanning into the
        # past); under ChunkedSchedule the prefix is always already ahead.
        if self._last_ts is not None:
            points = [(ts, v) for ts, v in points if ts > self._last_ts]
        if points:
            self._last_ts = points[-1][0]
        return [Timestamped(ts, self._encode(v)) for ts, v in points]

    def __call__(self, message: Any) -> Any | list[Timestamped]:
        if not isinstance(message, list):
            # Bare value (teleop Reset/Cartesian, scalar grip): one-shot, agent-timestamped.
            return self._encode(message)
        if not message:
            # Empty trajectory is the cancel signal (the Harness emits it at episode end):
            # drop the buffered tail so flush() does not commit canceled waypoints.
            self._buffer = []
            return []

        start = message[0][0]
        # Buffer is ts-sorted: everything before the new trajectory's start is
        # final; the rest is overridden and dropped by the reassignment below.
        cut = next((i for i, (ts, _) in enumerate(self._buffer) if ts >= start), len(self._buffer))
        committed = self._committable(self._buffer[:cut])
        self._buffer = list(message)
        return committed

    def flush(self, now_ns: int | None = None) -> list[Timestamped]:
        # At episode end, commit only points already due (ts <= now_ns); the
        # remaining future-scheduled tail never executed, so drop it. ``now_ns``
        # is None only for callers wanting the legacy "commit everything".
        points = self._buffer if now_ns is None else [(ts, v) for ts, v in self._buffer if ts <= now_ns]
        out = self._committable(points)
        self._buffer = []
        return out


def _append(ep_writer: EpisodeWriter, name: str, value: Any, ts_ns: int, extra_ts: dict[str, int] | None = None):
    for full_name, v in expand_suffixed(name, value):
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
    `DatasetWriter` and applies `static_data`. The opening turn drains the input
    channels without recording — the inter-episode home command and any pre-reset
    frame linger there from before START — so recording starts the next turn and
    the producer's post-reset frame-0 is the first sample. While open, each
    updated input signal (from `inputs`) is appended with the current timestamp
    from `clock`. `STOP_EPISODE` finalizes the writer after applying
    `static_data`; `ABORT_EPISODE` aborts and discards it. Invalid or
    out-of-order commands are ignored with a log message.

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
    ):
        self.ds_writer = ds_writer
        self._poll_hz = float(poll_hz)
        self._time_mode = time_mode
        self._virtual_time = virtual_time
        self.command = pimm.ControlSystemReceiver[DsWriterCommand](self, default=None)

        self._inputs: dict[str, pimm.ControlSystemReceiver[Any]] = {}
        self._serializers: dict[str, StatefulSerializer] = {}

    def add_signal(self, name: str, serializer: Serializer | StatefulSerializer | None = None):
        self._inputs[name] = pimm.ControlSystemReceiver[Any](self, default=None)
        if serializer is not None:
            if not isinstance(serializer, StatefulSerializer):
                serializer = _PureSerializer(serializer)
            self._serializers[name] = serializer

    @property
    def inputs(self) -> dict[str, pimm.ControlSystemReceiver[Any]]:
        return frozen_keys_dict(self._inputs)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        """Main loop: process commands and append updated inputs to the episode."""
        limiter = pimm.utils.RateLimiter(clock, hz=self._poll_hz)
        pace = (lambda: pimm.Yield()) if self._virtual_time else limiter.wait
        ep_writer: EpisodeWriter | None = None
        ep_counter = 0

        try:
            while not should_stop.value:
                cmd_msg = self.command.read()
                opened = False
                if cmd_msg.updated:
                    was_open = ep_writer is not None
                    ep_writer, ep_counter = self._handle_command(cmd_msg.data, ep_writer, ep_counter, cmd_msg.ts)
                    opened = ep_writer is not None and not was_open

                if ep_writer is not None:
                    for name, reader in self._inputs.items():
                        msg = reader.read()
                        # On the turn the episode opens, drain the channels without recording: the
                        # inter-episode home command and any pre-reset frame linger there from before START,
                        # so consuming them and recording from the next turn makes the producer's post-reset
                        # frame-0 (published after the recorder, in the same round) the first sample.
                        if msg.updated and not opened:
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
                            # Gate on `Timestamped` so plain list-valued samples
                            # (e.g. list-state vectors) still go through `_append`.
                            # Empty list matches too — used as the cancel signal.
                            if isinstance(value, list) and (not value or isinstance(value[0], Timestamped)):
                                for sample in value:
                                    _append(ep_writer, name, sample.value, sample.ts, None)
                            else:
                                _append(ep_writer, name, value, primary_ts, extra_ts)

                yield pace()
        finally:
            cmd_msg = self.command.read()
            if cmd_msg.updated:
                ep_writer, ep_counter = self._handle_command(cmd_msg.data, ep_writer, ep_counter, cmd_msg.ts)

            if ep_writer is not None:
                try:
                    ep_writer.abort()
                finally:
                    ep_writer.__exit__(None, None, None)
                    logger.info(f'DsWriterAgent: [ABORT] Episode {ep_counter}')

    def _handle_command(
        self, cmd: DsWriterCommand, ep_writer: EpisodeWriter | None, ep_counter: int, now_ns: int | None = None
    ):
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
                    for name, ser in self._serializers.items():
                        for sample in ser.flush(now_ns):
                            _append(ep_writer, name, sample.value, sample.ts, None)
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
