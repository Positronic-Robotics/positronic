import gc
import logging
import math
import threading
import time
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import Any

import numpy as np
from opentelemetry.trace import Span

import pimm
from pimm.time import EMITTED_WALL, EMITTED_WORLD, RECEIVED_WALL, RECEIVED_WORLD, emission_time
from positronic import keys, telemetry, telemetry_keys
from positronic.dataset.dataset import DatasetWriter
from positronic.dataset.episode import EpisodeWriter
from positronic.dataset.serializers import Serializer, expand_suffixed
from positronic.drivers.camera import CAPTURE_TIME, HANDOFF_TIME
from positronic.drivers.recording import HARNESS_WALL, HARNESS_WORLD
from positronic.drivers.roboarm.command import CartesianDelta, CartesianPosition
from positronic.drivers.roboarm.ik import ee_frames
from positronic.drivers.roboarm.keys import EE_FRAMES
from positronic.drivers.roboarm.models import EEFrame
from positronic.eval import Embodiment, Task, only_keep_written_signals
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import Answer, Commands, Obs, Policy, PolicyEnd, PolicyRun
from positronic.policy.codecs import PolicyCommandError
from positronic.policy.executor import Executor, WaitStatus
from positronic.utils import flatten_dict, frozen_view, merge_dicts

# Harness wake-up intervals on the world's clock.
POLL_PERIOD_SEC = 0.1
MIN_POLL_PERIOD_SEC = 0.005
MAX_POLL_PERIOD_SEC = 1.0


@dataclass
class Rollout:
    """One trial, its policy definition, and its unopened recording writer.

    The harness creates and owns the runtime and the generator returned by
    ``runtime.start(policy)``. The policy supplies its own dependencies.
    A ``writer`` of ``None`` records nothing.
    """

    task: Task
    policy: Policy
    writer: DatasetWriter | None


# The schedule's entries of the policy's metadata that the episode span carries, by span attribute.
_SCHEDULE_ATTRS = {
    telemetry_keys.ATTR_WAYPOINTS_HOLD_SUM_MS: f'{eval_keys.SCHEDULE}.{eval_keys.HOLD_SUM_MS}',
    telemetry_keys.ATTR_WAYPOINTS_HOLD_MAX_MS: f'{eval_keys.SCHEDULE}.{eval_keys.HOLD_MAX_MS}',
}


class _EpisodeTelemetry:
    """Parent reset, inference, and recorder spans to one episode; count its steps, late steps, encodes, and virtual
    duration."""

    # The episode span counts the steps that start this many milliseconds or more after their due time, by attribute.
    # Sorted by bound, as `bisect` needs.
    _LATE_STEP_BUCKETS = tuple(
        sorted(
            (
                (telemetry_keys.ATTR_EPISODE_STEPS_LATE_GE_1MS, 1.0),
                (telemetry_keys.ATTR_EPISODE_STEPS_LATE_GE_2MS, 2.0),
                (telemetry_keys.ATTR_EPISODE_STEPS_LATE_GE_5MS, 5.0),
                (telemetry_keys.ATTR_EPISODE_STEPS_LATE_GE_10MS, 10.0),
                (telemetry_keys.ATTR_EPISODE_STEPS_LATE_GE_20MS, 20.0),
                (telemetry_keys.ATTR_EPISODE_STEPS_LATE_GE_50MS, 50.0),
            ),
            key=lambda bucket: bucket[1],
        )
    )
    _LATE_STEP_MS = tuple(bound for _, bound in _LATE_STEP_BUCKETS)
    # The episode span counts the encodes that take more than this many nanoseconds, by attribute. Sorted by bound.
    _SLOW_ENCODE_BUCKETS = tuple(
        sorted(
            (
                (telemetry_keys.ATTR_EPISODE_ENCODES_GT_5MS, 5_000_000),
                (telemetry_keys.ATTR_EPISODE_ENCODES_GT_10MS, 10_000_000),
                (telemetry_keys.ATTR_EPISODE_ENCODES_GT_15MS, 15_000_000),
                (telemetry_keys.ATTR_EPISODE_ENCODES_GT_30MS, 30_000_000),
                (telemetry_keys.ATTR_EPISODE_ENCODES_GT_60MS, 60_000_000),
            ),
            key=lambda bucket: bucket[1],
        )
    )
    _SLOW_ENCODE_NS = tuple(bound for _, bound in _SLOW_ENCODE_BUCKETS)

    @staticmethod
    def _scalars(fields: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in fields.items() if isinstance(v, (bool, int, float, str))}

    def __init__(self) -> None:
        self._span: Span | None = None
        self._index = -1
        self._steps = 0
        # One bin per gap between the thresholds: a step or an encode adds 1 to one bin, and `end` sums the bins.
        self._late_steps = [0] * (len(self._LATE_STEP_MS) + 1)
        # The policy workers count encodes too.
        self._encodes_lock = threading.Lock()
        self._encodes = [0] * (len(self._SLOW_ENCODE_NS) + 1)
        self._virtual_start: float | None = None

    def begin(self, params: dict[str, Any]) -> None:
        """Open the episode span before preparation. Inert when telemetry is unbound."""
        self._index += 1
        self._steps = 0
        self._late_steps = [0] * (len(self._LATE_STEP_MS) + 1)
        # A new list: each sink adds to the list of its own episode.
        self._encodes = [0] * (len(self._SLOW_ENCODE_NS) + 1)
        self._virtual_start = None
        attrs: dict[str, Any] = {telemetry_keys.ATTR_EPISODE_INDEX: self._index}
        attrs.update(self._scalars(params))
        self._span = telemetry.start_span(telemetry_keys.SPAN_EPISODE, **attrs)
        telemetry.push_anchor(self._span)

    def describe_policy(self, recorded_policy_meta: Callable[[], dict[str, Any]]) -> None:
        """Copy the scalar fields of the policy's recorded metadata onto the open episode span.

        Takes a callable so that an unbound run does not build the metadata.
        """
        if self._span is None or not telemetry.enabled():
            return
        telemetry.set_attrs(self._span, **self._scalars(recorded_policy_meta()))

    def start_rollout(self, virtual_now: float) -> None:
        """Exclude preparation from the rollout's virtual duration."""
        self._virtual_start = virtual_now

    def step(self) -> None:
        self._steps += 1

    def late_step(self, late_ms: float) -> None:
        self._late_steps[bisect_right(self._LATE_STEP_MS, late_ms)] += 1

    @contextmanager
    def counting_encodes(self) -> Iterator[None]:
        """Count the encodes of the policy calls that the block submits, also the calls that end after the block."""
        if not telemetry.enabled():
            yield
            return
        # A worker keeps the sink after the episode ends, so the sink holds this episode's histogram.
        with telemetry.timings_to(partial(self._count_encode, self._encodes)):
            yield

    def _count_encode(self, encodes: list[int], name: str, start_ns: int, end_ns: int) -> None:
        if name != telemetry_keys.SPAN_POLICY_ENCODE:
            return
        with self._encodes_lock:  # one integer add; the workers and the harness thread count at once
            encodes[bisect_left(self._SLOW_ENCODE_NS, end_ns - start_ns)] += 1

    def describe_schedule(self, policy_meta: dict[str, Any]) -> None:
        """Copy the schedule's holds from the policy's recorded metadata onto the open episode span."""
        if self._span is None:
            return
        held = {attr: policy_meta[name] for attr, name in _SCHEDULE_ATTRS.items() if name in policy_meta}
        telemetry.set_attrs(self._span, **held)

    @staticmethod
    def _at_or_above(histogram: list[int]) -> list[int]:
        """The count of each threshold from a histogram whose bin `i + 1` starts at threshold `i`."""
        counts = []
        total = 0
        for n in reversed(histogram[1:]):
            total += n
            counts.append(total)
        return counts[::-1]

    def end(self, virtual_now: float, *, partial: bool = False) -> None:
        """Export the episode, including incomplete episodes interrupted by an error or shutdown."""
        if self._span is None:
            return
        virtual_s = virtual_now - self._virtual_start if self._virtual_start is not None else 0.0
        with self._encodes_lock:  # one list copy; the span gets each count of the same encodes
            encodes = list(self._encodes)
        attrs = {
            telemetry_keys.ATTR_EPISODE_STEPS: self._steps,
            telemetry_keys.ATTR_EPISODE_VIRTUAL_S: virtual_s,
            **{
                attr: n
                for (attr, _), n in zip(self._LATE_STEP_BUCKETS, self._at_or_above(self._late_steps), strict=True)
            },
            telemetry_keys.ATTR_EPISODE_ENCODES: sum(encodes),
            **{attr: n for (attr, _), n in zip(self._SLOW_ENCODE_BUCKETS, self._at_or_above(encodes), strict=True)},
        }
        if partial:
            attrs[telemetry_keys.ATTR_EPISODE_PARTIAL] = True
        telemetry.set_attrs(self._span, **attrs)
        self._span.end()
        telemetry.pop_anchor(self._span)
        self._span = None
        telemetry.force_flush()


class _GcPhase(StrEnum):
    """The phase that ``gc`` passes to a ``gc.callbacks`` entry."""

    START = 'start'
    STOP = 'stop'


class _GcMeter:
    """Sum this process's garbage-collection time, and count its collections by generation.

    A collection in any thread stops the harness thread too. Each collection costs two clock reads and integer sums,
    and the meter records no span, because young-generation collections run often.
    """

    def __init__(self) -> None:
        self._collection_started_ns = 0
        self.ns = 0
        self.collections = [0] * len(gc.get_count())

    def __call__(self, phase: str, info: dict[str, int]) -> None:
        match _GcPhase(phase):
            case _GcPhase.START:
                self._collection_started_ns = time.perf_counter_ns()
            case _GcPhase.STOP:
                self.ns += time.perf_counter_ns() - self._collection_started_ns
                self.collections[info['generation']] += 1

    @contextmanager
    def counting(self) -> Iterator[None]:
        gc.callbacks.append(self)
        try:
            yield
        finally:
            gc.callbacks.remove(self)


class _GcSteps:
    """Stamp each step span of one rollout with the collection time before the step and inside it.

    The wait window of a step opens when the previous step ends. For the first step, it opens when the rollout starts.
    """

    def __init__(self, meter: _GcMeter) -> None:
        self._meter = meter
        self._step_ended_ns = meter.ns
        self._step_ended_collections = tuple(meter.collections)

    @contextmanager
    def step(self, span: Span) -> Iterator[None]:
        started_ns = self._meter.ns
        try:
            yield
        finally:
            ended_ns, collections = self._meter.ns, tuple(self._meter.collections)
            collected = [
                g
                for g, (now, before) in enumerate(zip(collections, self._step_ended_collections, strict=True))
                if now > before
            ]
            if collected:
                telemetry.set_attrs(
                    span,
                    **{
                        telemetry_keys.ATTR_STEP_GC_WAIT_MS: (started_ns - self._step_ended_ns) / 1e6,
                        telemetry_keys.ATTR_STEP_GC_MS: (ended_ns - started_ns) / 1e6,
                        telemetry_keys.ATTR_STEP_GC_GENERATION: max(collected),
                    },
                )
            self._step_ended_ns, self._step_ended_collections = ended_ns, collections


class _RecordWait:
    """Sum the time that the samples between two steps spend in the recorder, and stamp it on the next step span.

    A sample between steps records no span of its own.
    """

    def __init__(self) -> None:
        self._ns: int | None = None

    @contextmanager
    def timing(self) -> Iterator[None]:
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self._ns = (self._ns or 0) + time.perf_counter_ns() - started_ns

    def stamp(self, step_ms: dict[str, float]) -> None:
        """Put the time summed since the previous stamp into ``step_ms``, when a sample recorded in that time."""
        if self._ns is not None:
            step_ms[telemetry_keys.ATTR_STEP_RECORD_WAIT_MS] = self._ns / 1e6
        self._ns = None


@dataclass(frozen=True)
class _PolicyEnded:
    """What a policy run returned, and the episode time at which it returned."""

    end: PolicyEnd
    time_ns: int


class Harness(pimm.ControlSystem):
    """Run episode lifecycles and emit each policy step's commands immediately.

    The policy sets the next wake-up time, clamped to 5 ms–1 s from the policy call's start. Without a
    policy step, a real rig polls every 100 ms. Simulation checks deadlines and preparation on simulator
    ticks. Every newly available answer can call the policy before its requested wake-up time.
    Real execution polls for completions at most every 5 ms while work is pending. Uncharged simulation
    handles completions before advancing time, including unrestricted chains of calls at one instant.

    Recording samples observations and privileged inputs before policy calls and at ``recording_hz`` between
    them. Each pass resets the sampling deadline. Harness appends commands after emission and waits for writer
    finalization before answering the rollout. A rollout without a writer records nothing.

    Each ``perform_task`` call runs one ``Rollout`` until its deadline, a truthy ``done`` signal, or the return of
    the policy run. A ``done`` or a return after the deadline counts as a timeout. When a ``done`` and a return come
    in one iteration, the ``done`` ends the episode. Its answer carries the terminal payload, or the error that failed
    the episode. The caller decides whether that error ends the run. Between episodes, manual commands pass through.
    Before each episode the harness asks every device to be ready, then prepares what the task names. A device
    that answers ready with an error fails the episode before it starts. A ``pimm.SignalError`` on an
    observation fails the episode before the policy reads that value. Between episodes the harness repairs a
    device in error (``_ready_devices_in_error``).
    """

    def __init__(
        self, embodiment: Embodiment, *, static_meta: dict[str, Any] | None = None, recording_hz: float = 1000
    ):
        if not math.isfinite(recording_hz) or not 0 < recording_hz <= 1e9:
            raise ValueError('recording_hz must be positive and at most 1 GHz')
        self._embodiment = embodiment
        self._static_meta = static_meta or {}
        names = [*embodiment.observations, *embodiment.commands]
        if len(names) != len(set(names)):
            raise ValueError('Observation and command signal names must be distinct')
        self._recording_period_ns = round(1e9 / recording_hz)
        self._writer: EpisodeWriter | None = None
        self._recorded_signals: set[str] = set()
        self._recorded_privileged: set[str] = set()
        self._obs_by_signal: dict[str, dict[str, Any] | pimm.SignalError | None] = {}
        # The emission time of each camera's frame in ``_obs_by_signal``, on the timeline of the policy runtime's clock.
        self._frame_ns_by_signal: dict[str, int] = {}
        # The last instruction this episode recorded, so a sample goes in only when the instruction changes.
        self._recorded_instruction: str | None = None
        # The newest goal the embodiment published in this episode, which the policy gets over the task's instruction.
        self._goal: str | None = None
        self._clock: pimm.Clock | None = None
        self._telemetry = _EpisodeTelemetry()
        self._gc = _GcMeter()
        self._record_wait = _RecordWait()
        # A ready call the harness made between episodes, by device name, until the device answers it.
        self._repairs: dict[str, pimm.calls.Answer[None]] = {}

        self.observations = pimm.ReceiverDict(self, names=embodiment.observations)
        self.commands = pimm.EmitterDict(self, names=embodiment.commands)
        self.privileged = pimm.ReceiverDict(self)
        self.ready = pimm.calls.CallerDict[None, None](self, names=embodiment.ready_handlers)
        self.prepare = pimm.calls.CallerDict[Any, None](self, names=embodiment.prepare_handlers)

        self.perform_task = pimm.calls.ControlSystemHandler[Rollout, dict[str, Any]](self)
        self.manual_command = pimm.ControlSystemReceiver(self)
        self.deadline_ns = pimm.ControlSystemEmitter[int | None](self)
        self.robot_meta_in = pimm.DefaultingReceiver(self, default={})
        self.arm_meta_in = pimm.ReceiverDict(self, names=embodiment.arm_meta_sources)
        self.goal = pimm.ControlSystemReceiver[str](self)
        self.done = pimm.DefaultingReceiver[dict](self, default={})

    def _yield(self, delay_sec: float = POLL_PERIOD_SEC) -> pimm.Command:
        """Yield one simulator tick, or sleep for ``delay_sec`` on a real rig."""
        return pimm.Yield() if self._embodiment.simulated else pimm.Sleep(delay_sec)

    def _wait(self, should_stop: pimm.SignalReceiver, answer: pimm.calls.Answer[Any]) -> pimm.Run[None]:
        """Wait until ``answer`` is done or the world stops. Raise the error ``answer`` carries."""
        while not answer.done() and not should_stop.value:
            yield self._yield()
        if answer.done():
            answer.result()

    def _ready_devices(self, should_stop: pimm.SignalReceiver) -> pimm.Run[None]:
        """Ask every device to be ready and wait for the answers. Raise the first error that a device answers."""
        yield from self._wait(should_stop, pimm.calls.all_of([ready(None) for ready in self.ready.values()]))

    def _ready_devices_in_error(self) -> None:
        """Ask each device whose observation of the same name carries a ``pimm.SignalError`` to be ready.

        Log the error of a device that does not repair. The next call finds its error and asks again.
        """
        for name, answer in list(self._repairs.items()):
            if not answer.done():
                continue
            del self._repairs[name]
            # rules-allow: swallowed-error — the device keeps its error, so the next call asks it again.
            try:
                answer.result()
            except Exception as e:
                logging.error(f'{name} is not ready after the harness asked: {e}')
        for name in (self.ready.keys() & self._embodiment.observations.keys()) - self._repairs.keys():
            message = self.observations[name].read()
            if message is not None and isinstance(message.data, pimm.SignalError):
                logging.warning(f'{name} sends an error, so the harness asks it to be ready: {message.data}')
                self._repairs[name] = self.ready[name](None)

    def _prepare(self, should_stop: pimm.SignalReceiver, args: dict[str, Any]) -> pimm.Run[None]:
        """Prepare only the named devices and wait for them. Empty args leave every device as it is."""
        unknown = sorted(set(args) - set(self.prepare))
        if unknown:
            rig = self._embodiment.descriptor or 'this rig'
            raise ValueError(f'{unknown} is not something {rig} readies; it readies {sorted(self.prepare)}')
        yield from self._wait(should_stop, pimm.calls.all_of([self.prepare[name](arg) for name, arg in args.items()]))

    def _statics(self) -> dict[str, Any]:
        return self._embodiment.static_meta | self._static_meta | self.robot_meta_in.value | self._arm_statics()

    def _arm_statics(self) -> dict[str, dict[str, Any]]:
        """The meta that each arm has emitted, in the shape ``Embodiment.arm_meta_sources`` states."""
        statics: dict[str, dict[str, Any]] = {}
        for joint_signal, receiver in self.arm_meta_in.items():
            if (msg := receiver.read()) is not None:
                merge_dicts(statics, {key: {joint_signal: value} for key, value in msg.data.items()})
        return statics

    @staticmethod
    def _recorded_policy_meta(own_meta: dict[str, Any], runtime_meta: dict[str, Any]) -> dict[str, Any]:
        """The policy's own metadata and what it set on the runtime, under the names the recording uses."""
        merged = deepcopy(flatten_dict(own_meta)) | flatten_dict(runtime_meta)
        return {f'{policy_keys.POLICY_META}.{k}': v for k, v in merged.items()}

    def _build_episode_meta(self, rollout: Rollout, runtime: Executor) -> dict[str, Any]:
        task = rollout.task
        meta = self._statics()
        meta[eval_keys.UNIVERSE] = 'sim' if self._embodiment.simulated else 'real'
        meta[eval_keys.EMBODIMENT] = self._embodiment.descriptor
        meta[eval_keys.CHARGE_INFERENCE_TIME] = task.charge_inference_time or not self._embodiment.simulated
        if task.timeout_sec is not None:  # the recorder takes no nulls, and an unbounded episode has none
            meta[eval_keys.TIMEOUT] = task.timeout_sec
        meta.update(self._recorded_policy_meta(rollout.policy.meta(), runtime.metadata_snapshot()))
        meta.update(task.meta)
        meta[keys.TASK] = self._instruction(task)
        return meta

    def _instruction(self, task: Task) -> str:
        """The text the policy gets: the newest goal that this episode read, or the task's instruction before any."""
        return self._goal if self._goal is not None else task.instruction

    @staticmethod
    def _record_io_span() -> AbstractContextManager[object]:
        return telemetry.span(telemetry_keys.SPAN_RECORD_IO, telemetry.Level.STEP)

    def _append(
        self,
        values: dict[str, Any],
        timestamps: pimm.Time,
        timing: Callable[[], AbstractContextManager[object]] = _record_io_span,
    ) -> None:
        if self._writer is None:
            return
        with timing():
            for name, value in values.items():
                self._writer.append(name, value.copy() if isinstance(value, np.ndarray) else value, timestamps)
                self._recorded_signals.add(name)

    def _record_instruction(self, instruction: str) -> None:
        """Record the instruction that the policy gets, at the harness's time, when it differs from the last one."""
        if self._writer is None or instruction == self._recorded_instruction:
            return
        assert self._clock is not None, 'an episode records only inside run'
        emitted = emission_time(self._clock)
        timestamps = pimm.Time(
            **emitted, **{HARNESS_WORLD: emitted[EMITTED_WORLD], HARNESS_WALL: emitted[EMITTED_WALL]}
        )
        self._append({eval_keys.INSTRUCTION: instruction}, timestamps)
        self._recorded_instruction = instruction

    def _convert_and_record(
        self,
        name: str,
        message: pimm.Message,
        serializer: Serializer | None,
        step_ms: dict[str, float] | None = None,
        timing: Callable[[], AbstractContextManager[object]] = _record_io_span,
    ) -> dict[str, Any]:
        stamp = message.time
        if self._writer is not None and not message.updated:
            assert self._clock is not None, 'an episode records only inside run'
            # A held message arrived before this read, so it records at the read.
            stamp = emission_time(self._clock)
        started_ns = time.perf_counter_ns()
        value = serializer(message.data) if serializer is not None else message.data
        values = {
            full_name: entry.copy() if isinstance(entry, np.ndarray) else entry
            for full_name, entry in expand_suffixed(name, value)
            if entry is not None
        }
        if step_ms is not None:
            step_ms[telemetry_keys.ATTR_STEP_CONVERT_MS_PREFIX + name] = (time.perf_counter_ns() - started_ns) / 1e6
        if self._writer is not None:
            world, wall = (RECEIVED_WORLD, RECEIVED_WALL) if RECEIVED_WORLD in stamp else (EMITTED_WORLD, EMITTED_WALL)
            timestamps = pimm.Time(**message.time, **{HARNESS_WORLD: stamp[world], HARNESS_WALL: stamp[wall]})
            self._append(values, timestamps, timing)
        return values

    def _time_frame(self, name: str, stamps: pimm.Time, emitted_ns: int | None, step_ms: dict[str, float]) -> None:
        """Put the two parts of the age of camera frame ``name`` into ``step_ms``.

        The wait from the hand-off ``emitted_ns`` to the read, before any conversion, is on the runtime clock. The
        driver's time from capture to hand-off is on the camera's clock, and is absent when the driver stamps no
        capture time.
        """
        if emitted_ns is not None and self._clock is not None:
            step_ms[telemetry_keys.ATTR_STEP_FRAME_WAIT_MS_PREFIX + name] = (self._clock.now_ns() - emitted_ns) / 1e6
        if CAPTURE_TIME in stamps and HANDOFF_TIME in stamps:
            step_ms[telemetry_keys.ATTR_STEP_FRAME_DRIVER_MS_PREFIX + name] = (
                stamps[HANDOFF_TIME] - stamps[CAPTURE_TIME]
            ) / 1e6

    def _sample(
        self,
        step_ms: dict[str, float] | None = None,
        timing: Callable[[], AbstractContextManager[object]] = _record_io_span,
    ) -> None:
        """Read each observation, and record each new message. ``timing`` brackets each append to the recorder."""
        for name, obs in self._embodiment.observations.items():
            read_started_ns = time.perf_counter_ns()
            message = self.observations[name].read()
            if step_ms is not None:
                step_ms[telemetry_keys.ATTR_STEP_READ_MS_PREFIX + name] = (
                    time.perf_counter_ns() - read_started_ns
                ) / 1e6
            if message is None or isinstance(message.data, pimm.SignalError):
                self._obs_by_signal[name] = message.data if message is not None else None
                self._frame_ns_by_signal.pop(name, None)
            else:
                emitted_ns = message.time.get(EMITTED_WORLD if self._embodiment.simulated else EMITTED_WALL)
                is_frame = name.startswith(keys.IMAGE_PREFIX)
                if is_frame and step_ms is not None:
                    self._time_frame(name, message.time, emitted_ns, step_ms)
                if message.updated or not isinstance(self._obs_by_signal.get(name), dict):
                    self._obs_by_signal[name] = self._convert_and_record(name, message, obs.serializer, step_ms, timing)
                    if is_frame and emitted_ns is not None:
                        self._frame_ns_by_signal[name] = emitted_ns
        if self._writer is not None:
            for name, receiver in self.privileged.items():
                message = receiver.read()
                if message is None or isinstance(message.data, pimm.SignalError):
                    continue
                if message.updated or name not in self._recorded_privileged:
                    self._convert_and_record(name, message, None, timing=timing)
                    self._recorded_privileged.add(name)

    def read_obs(self, task: Task, withheld_poses: frozenset[str], step_ms: dict[str, float]) -> Obs | None:
        """Read sensors, reusing each signal's serialized fields until a new message arrives. Leave out every pose
        in ``withheld_poses``.

        Copy updated arrays because devices may reuse their buffers while inference still reads them.
        Return ``None`` if a required signal has no message. Raise the ``pimm.SignalError`` a signal carries.
        Conversion errors propagate.
        Put each signal's read and conversion durations into ``step_ms``.
        Give the policy the newest goal of this episode, or the task's instruction before any. Record that text as
        the ``eval_keys.INSTRUCTION`` signal each time a returned observation carries a new one.
        """
        self._sample(step_ms)
        inputs: dict[str, Any] = {}
        complete = True
        for values in self._obs_by_signal.values():
            if isinstance(values, pimm.SignalError):
                # Repeated raises of a cached error must not accumulate traceback frames.
                raise values.with_traceback(None)
            if values is None:
                complete = False
            else:
                inputs.update(values)
        if not complete:
            return None
        for name in withheld_poses:
            inputs.pop(name, None)
        if (goal := pimm.value_updated(self.goal)) is not None:
            self._goal = goal
        inputs[keys.TASK] = instruction = self._instruction(task)
        self._record_instruction(instruction)
        inputs[keys.DESCRIPTOR] = self._embodiment.descriptor
        return frozen_view(inputs)

    @staticmethod
    def _refuse_unstated_pose_commands(commands: Commands, stated_poses: frozenset[str]) -> None:
        """Raise when ``commands`` holds a pose command on a channel whose pose signal is not in ``stated_poses``."""
        unstated = sorted(
            name
            for name, command in commands.items()
            if isinstance(command, CartesianPosition | CartesianDelta)
            and f'{name}{keys.POSE_SUFFIX}' not in stated_poses
        )
        if unstated:
            raise PolicyCommandError(
                f'The policy sends a pose command on {unstated}, and states no {EE_FRAMES} entry for the pose of that '
                'channel. State the frame of each pose that the policy reads or commands.'
            )

    def _step(
        self,
        task: Task,
        withheld_poses: frozenset[str],
        stated_poses: frozenset[str],
        runtime: Executor,
        policy_run: PolicyRun,
        due_ns: int | None,
        gc_steps: _GcSteps,
    ) -> int | _PolicyEnded | None:
        """Read sensors, call the policy, emit commands, and return its clamped next wake-up time. Return a
        ``_PolicyEnded`` when the policy run returns, and ``None`` without an observation.

        ``due_ns`` is the wake-up time the previous step returned, or ``None`` for the first step. The step span
        records the step's durations, and the garbage-collection time before and inside the step. A pose command on a
        channel whose pose is not in ``stated_poses`` fails the step.
        """
        step_ms: dict[str, float] = {}
        with telemetry.span(telemetry_keys.SPAN_HARNESS_STEP, telemetry.Level.STEP) as span, gc_steps.step(span):
            self._record_wait.stamp(step_ms)
            try:
                if due_ns is not None and runtime.time_ns >= due_ns:
                    step_ms[telemetry_keys.ATTR_STEP_LATE_MS] = late_ms = (runtime.time_ns - due_ns) / 1e6
                    self._telemetry.late_step(late_ms)
                observe_started_ns = time.perf_counter_ns()
                obs = self.read_obs(task, withheld_poses, step_ms)
                policy_started_ns = time.perf_counter_ns()
                step_ms[telemetry_keys.ATTR_STEP_OBSERVE_MS] = (policy_started_ns - observe_started_ns) / 1e6
                if obs is None:
                    return None
                runtime.start_tick(min(self._frame_ns_by_signal.values(), default=None))
                started_at_ns = runtime.time_ns
                try:
                    step = policy_run.send(obs)
                except StopIteration as end:
                    if not isinstance(end.value, PolicyEnd):
                        raise ValueError(f'a policy run returned {end.value!r}, which is not a PolicyEnd') from None
                    return _PolicyEnded(end.value, runtime.time_ns)
                finally:
                    step_ms[telemetry_keys.ATTR_STEP_POLICY_MS] = (time.perf_counter_ns() - policy_started_ns) / 1e6
                assert step is not None, 'a policy must yield a Step for each observation'
                if unknown := sorted(step.commands.keys() - self._embodiment.commands.keys()):
                    raise PolicyCommandError(
                        f'The step names {unknown}, and the embodiment takes only {sorted(self._embodiment.commands)}'
                    )
                self._refuse_unstated_pose_commands(step.commands, stated_poses)
                emit_started_ns = time.perf_counter_ns()
                self._telemetry.step()
                messages = {name: self.commands[name].emit(value) for name, value in step.commands.items()}
                step_ms[telemetry_keys.ATTR_STEP_EMIT_MS] = (time.perf_counter_ns() - emit_started_ns) / 1e6
                if self._writer is not None:
                    for name, message in messages.items():
                        self._convert_and_record(name, message, self._embodiment.commands[name].serializer)
            finally:
                if span.is_recording():
                    telemetry.set_attrs(span, **step_ms)
        period_sec = (step.resume_at_ns - started_at_ns) / 1e9
        period_sec = min(MAX_POLL_PERIOD_SEC, max(MIN_POLL_PERIOD_SEC, period_sec))
        return started_at_ns + round(period_sec * 1e9)

    @staticmethod
    def _trial_terminal(
        done: pimm.Message[dict] | None, ended: _PolicyEnded | None, now_ns: int, deadline_ns: int | None, timeline: str
    ) -> dict[str, Any] | None:
        """A done signal or a policy end after the deadline counts as a timeout, not a success.

        The env's ``done`` is its ground truth, so it ends the episode in place of a policy end of the same iteration.
        """
        if done is not None and done.data and (deadline_ns is None or done.time[timeline] <= deadline_ns):
            return {**done.data, eval_keys.TERMINATED: True}
        if ended is not None and (deadline_ns is None or ended.time_ns <= deadline_ns):
            return {
                eval_keys.ENDED_BY: eval_keys.EndedBy.POLICY,
                eval_keys.POLICY_END: ended.end.value,
                eval_keys.TERMINATED: True,
            }
        if deadline_ns is not None and now_ns >= deadline_ns:
            return {eval_keys.TERMINATED: False}
        return None

    def _wait_for_next_tick(self, runtime: Executor, resume_at_ns: int) -> pimm.Run[tuple[Answer[Any], ...]]:
        """Return completions before advancing time; otherwise follow simulator ticks or poll real time.

        Shutdown is checked between episode iterations. Pending inference may delay shutdown;
        this wait intentionally does not poll should_stop. KeyboardInterrupt propagates through normal cleanup.
        """
        while True:
            result = runtime.wait(timeout_sec=POLL_PERIOD_SEC)
            match result.status:
                case WaitStatus.ANSWERS_READY:
                    return result.completed
                case WaitStatus.CAN_ADVANCE:
                    break
                case WaitStatus.TIMED_OUT:
                    continue
        # A positive real-time sleep gives this loop its own wake-up, independent of other loops' timers.
        delay_ns = max(1, resume_at_ns - runtime.time_ns)
        if runtime.has_pending:
            delay_ns = min(delay_ns, round(MIN_POLL_PERIOD_SEC * 1e9))
        yield self._yield(delay_ns / 1e9)
        return runtime.take_completed()

    @contextmanager
    def _recording(self, dataset_writer: DatasetWriter | None) -> Iterator[None]:
        try:
            with ExitStack() as stack:
                if dataset_writer is not None:
                    with telemetry.span(telemetry_keys.SPAN_RECORD_IO):
                        dataset = stack.enter_context(dataset_writer)
                        writer = dataset.new_episode()
                        self._writer = writer.__enter__()
                        stack.push(telemetry.traced(telemetry_keys.SPAN_RECORD_IO)(writer.__exit__))
                yield
        finally:
            self._writer = None
            self._recorded_signals.clear()
            self._recorded_privileged.clear()
            self._obs_by_signal.clear()
            self._frame_ns_by_signal.clear()
            self._recorded_instruction = None
            self._goal = None
            self._record_wait = _RecordWait()

    def _rollout(
        self,
        should_stop: pimm.SignalReceiver,
        task: Task,
        withheld_poses: frozenset[str],
        stated_poses: frozenset[str],
        runtime: Executor,
        policy_run: PolicyRun,
        deadline_ns: int | None,
        gc_steps: _GcSteps,
    ) -> pimm.Run[dict[str, Any] | None]:
        payload = None
        resume_at_ns = None
        ended = None
        sample_at_ns = runtime.time_ns
        completed = ()
        while not should_stop.value and payload is None:
            sampled_ns = runtime.time_ns
            if completed or resume_at_ns is None or sampled_ns >= resume_at_ns:
                stepped = self._step(task, withheld_poses, stated_poses, runtime, policy_run, resume_at_ns, gc_steps)
                if isinstance(stepped, _PolicyEnded):
                    ended = stepped
                else:
                    resume_at_ns = stepped
                sample_at_ns = sampled_ns + self._recording_period_ns
            elif self._writer is not None and sampled_ns >= sample_at_ns:
                # The summed wait goes on the next step span, so it is timed only while step spans record.
                self._sample(
                    timing=self._record_wait.timing if telemetry.enabled(telemetry.Level.STEP) else self._record_io_span
                )
                sample_at_ns = sampled_ns + self._recording_period_ns
            # No complete observation yet: read the sensors again after one poll period.
            wake_at_ns = runtime.time_ns + round(POLL_PERIOD_SEC * 1e9) if resume_at_ns is None else resume_at_ns
            if ended is not None:
                wake_at_ns = ended.time_ns
            if self._writer is not None:
                wake_at_ns = min(wake_at_ns, sample_at_ns)
            if deadline_ns is not None:
                wake_at_ns = min(wake_at_ns, deadline_ns)
            completed = yield from self._wait_for_next_tick(runtime, wake_at_ns)
            if call := next(self.perform_task.incoming(), None):
                call.set_exception(RuntimeError('An episode is already running'))
            pimm.read_updated(self.manual_command)
            payload = self._trial_terminal(
                pimm.read_updated(self.done),
                ended,
                runtime.time_ns,
                deadline_ns,
                EMITTED_WORLD if self._embodiment.simulated else EMITTED_WALL,
            )
        if self._writer is not None:
            self._sample()
        return payload

    @staticmethod
    def _policy_frames(policy: Policy, runtime: Executor) -> dict[str, EEFrame]:
        """The frames that ``policy`` states. Call it after ``runtime.start(policy)``, before the first observation.

        The policy states its frames as ``EE_FRAMES`` in ``meta()``, or its run writes them to ``runtime`` as it
        starts, as a remote run does once its session opens. Raise when both state the frame of one pose signal.
        """
        stated, recorded = ee_frames(policy.meta()), ee_frames(runtime.metadata_snapshot())
        if both := stated.keys() & recorded.keys():
            raise ValueError(f'Only one component of the policy may state the {EE_FRAMES} of {sorted(both)}')
        return stated | recorded

    def _stated_poses(self, declared: dict[str, EEFrame]) -> frozenset[str]:
        """The rig's pose signals whose frame ``declared`` states. A policy observes and commands only these poses.

        Raise when the policy states another site than the rig, or states frames on a rig that states none, such as an
        arm with no model.
        """
        rig = ee_frames(self._statics())
        if declared and not rig:
            raise ValueError(
                f'The rig states no {EE_FRAMES}, so it runs only a policy that speaks joints, and the policy states '
                f'frames for {sorted(declared)}'
            )
        mismatched = {
            name: (frame.site, rig[name].site)
            for name, frame in declared.items()
            if name in rig and frame.site != rig[name].site
        }
        if mismatched:
            raise ValueError(
                'The policy declares poses at a site the rig does not report them at, as {signal: (declared, rig)}: '
                f'{mismatched}'
            )
        return frozenset(declared.keys() & rig.keys())

    def _pose_signals(self) -> frozenset[str]:
        """Every pose signal of the rig."""
        # The embodiment lists every pose its serializers write; an arm with no model states none of them.
        return frozenset(self._embodiment.static_meta.get(eval_keys.POSE_SIGNALS, ())).union(ee_frames(self._statics()))

    def _run_episode(
        self, clock: pimm.Clock, should_stop: pimm.SignalReceiver, rollout: Rollout
    ) -> pimm.Run[dict[str, Any] | None]:
        """Prepare, run, record, and return the rig. Return None when stopped without a terminal payload."""
        task = rollout.task
        self._telemetry.begin(task.meta)
        # Name the policy before the reset, so that an episode that ends in its reset still names it.
        self._telemetry.describe_policy(lambda: self._recorded_policy_meta(rollout.policy.meta(), {}))
        with telemetry.span(telemetry_keys.SPAN_RESET):
            yield from self._prepare(should_stop, task.prepare_args)
        if should_stop.value:
            return None

        runtime = Executor(
            clock.now_ns, simulated=self._embodiment.simulated, charge_inference_time=task.charge_inference_time
        )
        # Start the policy in the scope, so that the calls it submits before its first step count too.
        with self._telemetry.counting_encodes():
            policy_run = None
            try:
                policy_run = runtime.start(rollout.policy)
                stated_poses = self._stated_poses(self._policy_frames(rollout.policy, runtime))
                withheld_poses = self._pose_signals() - stated_poses
                deadline_ns = clock.now_ns() + round(task.timeout_sec * 1e9) if task.timeout_sec is not None else None
                self.deadline_ns.emit(deadline_ns)
                self._telemetry.start_rollout(clock.now())
                gc_steps = _GcSteps(self._gc)
                try:
                    with self._recording(rollout.writer):
                        payload = yield from self._rollout(
                            should_stop, task, withheld_poses, stated_poses, runtime, policy_run, deadline_ns, gc_steps
                        )
                        if self._writer is not None:
                            static = only_keep_written_signals(
                                self._build_episode_meta(rollout, runtime) | (payload or {}), self._recorded_signals
                            )
                            for name, value in static.items():
                                self._writer.set_static(name, value)
                finally:
                    self.deadline_ns.emit(None)
            finally:
                # Write the server's metadata before cleanup, so that a cleanup failure keeps it on the span.
                self._telemetry.describe_policy(
                    lambda: self._recorded_policy_meta(rollout.policy.meta(), runtime.metadata_snapshot())
                )
                self._telemetry.describe_schedule(runtime.metadata_snapshot())
                # Cleanup stops at the first error. Later resources may remain open; do not add nested
                # finally blocks to guarantee their closure.
                logging.info('Closing the policy runtime')
                runtime.close()
                logging.info('Policy runtime closed')
                if policy_run is not None:
                    logging.info('Closing the policy')
                    policy_run.close()
                    logging.info('Policy closed')

        self._telemetry.end(clock.now())
        if payload is not None:
            back_args = {k: v for k, v in task.prepare_args.items() if k != eval_keys.SCENE}
            # rules-allow: swallowed-error — the move back is cleanup, and the recording is already complete.
            try:
                yield from self._prepare(should_stop, back_args)
            except Exception as exc:
                logging.error(f'The rig failed to go back after the episode: {exc}')
        return payload

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> pimm.Run[None]:
        self._clock = clock
        for emitter in self.commands.values():
            # A command without a device receiver still needs the world's emission timestamps.
            emitter._clock = clock
        # Episode spans must end before leaving the scope that closes the telemetry provider.
        with telemetry.bind_from_env(telemetry_keys.HARNESS_PROCESS), ExitStack() as gc_scope:
            if telemetry.enabled(telemetry.Level.STEP):
                gc_scope.enter_context(self._gc.counting())
            while not should_stop.value:
                call = next(self.perform_task.incoming(), None)
                manual = pimm.value_updated(self.manual_command)
                # An idle done signal must not terminate the next episode, and an idle goal must not steer it.
                pimm.read_updated(self.done)
                pimm.read_updated(self.goal)
                if call is not None:
                    payload = None
                    failure: BaseException = pimm.calls.HandlerStopped()
                    # rules-allow: swallowed-error — the answer carries the error; the caller decides if the run ends
                    try:
                        yield from self._ready_devices(should_stop)
                        payload = yield from self._run_episode(clock, should_stop, call.request)
                    except Exception as exc:
                        failure = exc
                    finally:
                        self._telemetry.end(clock.now(), partial=True)
                        if payload is None:
                            call.set_exception(failure)
                    if payload is not None:
                        call.set_result(payload)
                elif manual is not None:
                    for name, value in manual.items():
                        self.commands[name].emit(value)
                if not should_stop.value:
                    self._ready_devices_in_error()
                    yield self._yield()
