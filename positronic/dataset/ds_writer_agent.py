import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

import numpy as np

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm.command import CartesianPosition, CommandType, JointDelta, JointPosition, Reset
from positronic.utils import frozen_keys_dict

from .dataset import DatasetWriter
from .episode import EpisodeWriter

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
    SUSPEND_EPISODE = 'suspend_episode'


@dataclass
class DsWriterCommand:
    """Command message consumed by `DsWriterAgent`.

    Args:
        type: Desired episode action (start/stop/abort/suspend).
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

    @staticmethod
    def SUSPEND():
        return DsWriterCommand(DsWriterCommandType.SUSPEND_EPISODE)


# Serializer contract for inputs:
# - Register signals with `DsWriterAgent.add_signal(name, serializer=None)`.
# - If `serializer` is omitted or None, the value is passed through unchanged.
# - If a callable is provided, it is invoked as serializer(value) and can return:
#     * a transformed value -> recorded under the same signal name
#     * a dict mapping suffix -> value -> expands into multiple signals recorded as name+suffix
#         - use "" (empty string) to keep the base name as-is
#         - any dict entry with value None is skipped (not recorded)
#     * None -> the sample is dropped (not recorded)
#     * a list[Timestamped] -> a self-timestamped stream: each item is recorded at
#       its own ``ts_ns`` (not the agent loop/message timestamp). An empty list
#       defers (records nothing now); a StatefulSerializer may emit the remainder
#       later from ``flush()``. The per-item ``value`` follows the same
#       value/dict/None rules above.
Serializer = Callable[[Any], Any | dict[str, Any]]


@dataclass
class Timestamped:
    """A sample paired with its own absolute timestamp (ns)."""

    ts: int
    value: Any


class StatefulSerializer:
    """Base for serializers registered with ``DsWriterAgent``.

    ``reset`` is called automatically at the start of each episode.
    The default implementation is a no-op, suitable for pure serializers.
    Subclasses that maintain per-episode state should override ``reset``.
    """

    def reset(self) -> None:
        pass

    def __call__(self, value: Any) -> Any | dict[str, Any] | list['Timestamped']:
        raise NotImplementedError

    def flush(self, now_ns: int | None = None) -> list['Timestamped']:
        """Drain any buffered samples at episode end (mirror of ``reset``).

        Called once on ``STOP_EPISODE`` before the episode is finalized. ``now_ns``
        is the episode-end time; serializers that buffer future-scheduled samples
        use it to drop the un-executed tail. The default keeps stateless
        serializers a no-op.
        """
        return []


class _PureSerializer(StatefulSerializer):
    """Wraps a plain callable so every serializer has a uniform interface."""

    def __init__(self, fn: Callable[[Any], Any | dict[str, Any]]):
        self._fn = fn

    def __call__(self, value: Any) -> Any | dict[str, Any]:
        return self._fn(value)


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
        """Serialize a Transform3D into a 7D vector [tx, ty, tz, qw, qx, qy, qz]."""
        return x.as_vector(geom.Rotation.Representation.QUAT)

    class ContinuousTransform3D(StatefulSerializer):
        """Stateful serializer that canonicalises quaternion signs for temporal continuity.

        Each quaternion is flipped to the sign closest to the previous frame,
        avoiding arbitrary sign jumps from the double-cover ambiguity.
        """

        def __init__(self):
            self._prev: geom.Rotation | None = None

        def reset(self):
            self._prev = None

        def __call__(self, x: geom.Transform3D) -> np.ndarray:
            rotation = x.rotation
            if self._prev is not None:
                rotation = geom.quat_closest(rotation, self._prev)
            self._prev = rotation
            return geom.Transform3D(x.translation, rotation).as_vector(geom.Rotation.Representation.QUAT)

    @staticmethod
    def robot_state(state: State) -> dict[str, np.ndarray | int] | None:
        if state.status == RobotStatus.RESETTING:
            return None
        return {
            '.q': state.q,
            '.dq': state.dq,
            '.ee_pose': Serializers.transform_3d(state.ee_pose),
            '.error': int(state.status == RobotStatus.ERROR),
        }

    @staticmethod
    def robot_command(command: CommandType) -> dict[str, np.ndarray | int] | None:
        match command:
            case CartesianPosition(pose):
                return {'.pose': Serializers.transform_3d(pose)}
            case JointPosition(positions):
                return {'.joints': positions}
            case JointDelta(delta):
                return {'.joint_deltas': delta}
            case Reset():
                return {'.reset': 1}

    @staticmethod
    def camera_images(data: pimm.shared_memory.NumpySMAdapter) -> np.ndarray:
        """Extract array from NumpySMAdapter for storage."""
        return data.array


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
            # Empty trajectory is the cancel signal (Harness STOP): drop the
            # buffered tail so flush() does not commit canceled waypoints.
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
        """Main loop: process commands and append updated inputs to the episode.

        Refactored to reduce complexity: command handling, serialization, and
        appending are split into helpers.
        """
        limiter = pimm.utils.RateLimiter(clock, hz=self._poll_hz)
        ep_writer: EpisodeWriter | None = None
        ep_counter = 0
        suspended = False

        try:
            while not should_stop.value:
                cmd_msg = self.command.read()
                if cmd_msg.updated:
                    ep_writer, ep_counter, suspended = self._handle_command(
                        cmd_msg.data, ep_writer, ep_counter, suspended, cmd_msg.ts
                    )

                if ep_writer is not None and not suspended:
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
                            # Gate on `Timestamped` so plain list-valued samples
                            # (e.g. list-state vectors) still go through `_append`.
                            # Empty list matches too — used as the cancel signal.
                            if isinstance(value, list) and (not value or isinstance(value[0], Timestamped)):
                                for sample in value:
                                    _append(ep_writer, name, sample.value, sample.ts, None)
                            else:
                                _append(ep_writer, name, value, primary_ts, extra_ts)

                yield limiter.wait()
        finally:
            cmd_msg = self.command.read()
            if cmd_msg.updated:
                ep_writer, ep_counter, suspended = self._handle_command(
                    cmd_msg.data, ep_writer, ep_counter, suspended, cmd_msg.ts
                )

            if ep_writer is not None:
                try:
                    ep_writer.abort()
                finally:
                    ep_writer.__exit__(None, None, None)
                    logger.info(f'DsWriterAgent: [ABORT] Episode {ep_counter}')

    def _handle_command(
        self,
        cmd: DsWriterCommand,
        ep_writer: EpisodeWriter | None,
        ep_counter: int,
        suspended: bool,
        now_ns: int | None = None,
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
                    suspended = False
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
                    suspended = False
                else:
                    logger.warning('Episode not started, ignoring stop command')
            case DsWriterCommandType.ABORT_EPISODE:
                if ep_writer is not None:
                    ep_writer.abort()
                    ep_writer.__exit__(None, None, None)
                    logger.info(f'DsWriterAgent: [ABORT] Episode {ep_counter}')
                    ep_writer = None
                    suspended = False
                else:
                    logger.warning('Episode not started, ignoring abort command')
            case DsWriterCommandType.SUSPEND_EPISODE:
                if ep_writer is not None:
                    logger.info(f'DsWriterAgent: [SUSPEND] Episode {ep_counter}')
                    suspended = True
                    # While suspended the loop skips inputs, so the harness's `[]` cancel
                    # never reaches the serializers. Flush at the suspend time now: commit
                    # samples already due and drop the un-executed future tail, so a later
                    # STOP_EPISODE doesn't commit waypoints the robot never executed.
                    for name, ser in self._serializers.items():
                        for sample in ser.flush(now_ns):
                            _append(ep_writer, name, sample.value, sample.ts, None)
                else:
                    logger.warning('Episode not started, ignoring suspend command')
        return ep_writer, ep_counter, suspended
