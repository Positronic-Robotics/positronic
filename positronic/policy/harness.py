import logging
import math
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from opentelemetry.trace import Span

import pimm
from pimm.time import EMITTED_WALL, EMITTED_WORLD, RECEIVED_WALL, RECEIVED_WORLD
from positronic import keys, telemetry, telemetry_keys
from positronic.dataset.dataset import DatasetFactory
from positronic.dataset.episode import EpisodeWriter
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.dataset.serializers import Serializer, expand_suffixed
from positronic.dataset.time import HARNESS_WALL, HARNESS_WORLD
from positronic.drivers.roboarm.ik import assert_default_frame
from positronic.eval import Embodiment, Observation, Task
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import Answer, Obs, Policy, PolicyRun
from positronic.policy.executor import Executor, WaitStatus
from positronic.utils import flatten_dict, frozen_view

# Harness wake-up intervals on the world's clock.
POLL_PERIOD_SEC = 0.1
MIN_POLL_PERIOD_SEC = 0.005
MAX_POLL_PERIOD_SEC = 1.0


@dataclass
class Rollout:
    """One trial, its complete policy definition, and the path it records into.

    The harness creates and owns the runtime and the generator returned by
    ``runtime.start(policy)``. The policy supplies its own dependencies.
    An ``output_path`` of ``None`` records nothing.
    """

    task: Task
    policy: Policy
    output_path: Path | None


class _EpisodeTelemetry:
    """Parent reset, inference, and recorder spans to one episode; count its steps and virtual duration."""

    def __init__(self) -> None:
        self._span: Span | None = None
        self._index = -1
        self._steps = 0
        self._virtual_start: float | None = None

    def begin(self, params: dict[str, Any]) -> None:
        """Open the episode span before preparation. Inert when telemetry is unbound."""
        self._index += 1
        self._steps = 0
        self._virtual_start = None
        attrs: dict[str, Any] = {telemetry_keys.ATTR_EPISODE_INDEX: self._index}
        attrs.update({k: v for k, v in params.items() if isinstance(v, (bool, int, float, str))})
        self._span = telemetry.start_span(telemetry_keys.SPAN_EPISODE, **attrs)
        telemetry.push_anchor(self._span)

    def start_rollout(self, virtual_now: float) -> None:
        """Exclude preparation from the rollout's virtual duration."""
        self._virtual_start = virtual_now

    def step(self) -> None:
        self._steps += 1

    def end(self, virtual_now: float, *, partial: bool = False) -> None:
        """Export the episode, including incomplete episodes interrupted by an error or shutdown."""
        if self._span is None:
            return
        virtual_s = virtual_now - self._virtual_start if self._virtual_start is not None else 0.0
        attrs = {telemetry_keys.ATTR_EPISODE_STEPS: self._steps, telemetry_keys.ATTR_EPISODE_VIRTUAL_S: virtual_s}
        if partial:
            attrs[telemetry_keys.ATTR_EPISODE_PARTIAL] = True
        telemetry.set_attrs(self._span, **attrs)
        self._span.end()
        telemetry.pop_anchor(self._span)
        self._span = None
        telemetry.force_flush()


class Harness(pimm.ControlSystem):
    """Run episode lifecycles and emit each policy step's commands immediately.

    The policy sets the next wake-up time, clamped to 5 ms–1 s from the policy call's start. Without a
    policy step, a real rig polls every 100 ms. Simulation checks deadlines and preparation on simulator
    ticks. Every newly available answer can call the policy before its requested wake-up time.
    Real execution polls for completions at most every 5 ms while work is pending. Uncharged simulation
    handles completions before advancing time, including unrestricted chains of calls at one instant.

    Recording samples observations and privileged inputs before policy calls and at ``recording_hz`` between
    them. Each pass resets the sampling deadline. Harness appends commands after emission and waits for writer
    finalization before answering the rollout. A rollout without an output path, or ``record=False``, writes nothing.

    Each ``perform_task`` call runs one ``Rollout`` until its deadline or a truthy ``done`` signal.
    Its answer carries the terminal payload, or the error that failed the episode. The caller decides
    whether that error ends the run. Between episodes, manual commands pass through.
    Before each episode the harness asks every device to be ready, then prepares what the task names. A device
    that answers ready with an error fails the episode before it starts. A ``pimm.SignalError`` on an
    observation fails the episode before the policy reads that value. Between episodes the harness repairs a
    device in error (``_ready_devices_in_error``).
    """

    def __init__(
        self,
        embodiment: Embodiment,
        *,
        static_meta: dict[str, Any] | None = None,
        privileged: dict[str, Observation] | None = None,
        dataset_factory: DatasetFactory | None = None,
        recording_hz: float = 1000,
        record: bool = True,
    ):
        if not math.isfinite(recording_hz) or not 0 < recording_hz <= 1e9:
            raise ValueError('recording_hz must be positive and at most 1 GHz')
        self._embodiment = embodiment
        self._static_meta = static_meta or {}
        self._privileged = privileged or {}
        names = [*embodiment.observations, *embodiment.commands, *self._privileged]
        if len(names) != len(set(names)):
            raise ValueError('Observation, command and privileged signal names must be distinct')
        self._dataset_factory = (
            (dataset_factory or partial(LocalDatasetWriter, video_encoder=embodiment.video_encoder)) if record else None
        )
        self._recording_period_ns = round(1e9 / recording_hz)
        self._writer: EpisodeWriter | None = None
        self._recorded_privileged: set[str] = set()
        self._obs_by_signal: dict[str, dict[str, Any] | pimm.SignalError | None] = {}
        self._telemetry = _EpisodeTelemetry()
        # A ready call the harness made between episodes, by device name, until the device answers it.
        self._repairs: dict[str, pimm.calls.Answer[None]] = {}

        self.observations = pimm.ReceiverDict(self, names=embodiment.observations)
        self.commands = pimm.EmitterDict(self, names=embodiment.commands)
        self.privileged = pimm.ReceiverDict(self, names=self._privileged)
        self.ready = pimm.calls.CallerDict[None, None](self, names=embodiment.ready_handlers)
        self.prepare = pimm.calls.CallerDict[Any, None](self, names=embodiment.prepare_handlers)

        self.perform_task = pimm.calls.ControlSystemHandler[Rollout, dict[str, Any]](self)
        self.manual_command = pimm.ControlSystemReceiver(self)
        self.deadline_ns = pimm.ControlSystemEmitter[int | None](self)
        self.robot_meta_in = pimm.DefaultingReceiver(self, default={})
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
        return self._embodiment.static_meta | self._static_meta | self.robot_meta_in.value

    def _build_episode_meta(self, rollout: Rollout, runtime: Executor) -> dict[str, Any]:
        task = rollout.task
        meta = self._statics()
        meta[eval_keys.UNIVERSE] = 'sim' if self._embodiment.simulated else 'real'
        meta[eval_keys.EMBODIMENT] = self._embodiment.descriptor
        meta[eval_keys.CHARGE_INFERENCE_TIME] = task.charge_inference_time or not self._embodiment.simulated
        if task.timeout_sec is not None:  # the recorder takes no nulls, and an unbounded episode has none
            meta[eval_keys.TIMEOUT] = task.timeout_sec
        for k, v in deepcopy(flatten_dict(rollout.policy.meta()) | flatten_dict(runtime.metadata)).items():
            meta[f'{policy_keys.POLICY_META}.{k}'] = v
        meta.update(task.meta)
        meta[keys.TASK] = task.instruction
        return meta

    def _append(self, values: dict[str, Any], time: pimm.Time) -> None:
        if self._writer is None:
            return
        world, wall = (RECEIVED_WORLD, RECEIVED_WALL) if RECEIVED_WORLD in time else (EMITTED_WORLD, EMITTED_WALL)
        timestamps = pimm.Time(**time, **{HARNESS_WORLD: time[world], HARNESS_WALL: time[wall]})
        with telemetry.span(telemetry_keys.SPAN_RECORD_IO):
            for name, value in values.items():
                self._writer.append(name, value.copy() if isinstance(value, np.ndarray) else value, timestamps)

    def _convert_and_record(
        self, name: str, message: pimm.Message, serializer: Serializer | None, step_ms: dict[str, float] | None = None
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        value = serializer(message.data) if serializer is not None else message.data
        values = {
            full_name: entry.copy() if isinstance(entry, np.ndarray) else entry
            for full_name, entry in expand_suffixed(name, value)
            if entry is not None
        }
        if step_ms is not None:
            step_ms[telemetry_keys.ATTR_STEP_CONVERT_MS_PREFIX + name] = (time.perf_counter_ns() - started_ns) / 1e6
        self._append(values, message.time)
        return values

    def _sample(self, step_ms: dict[str, float] | None = None) -> None:
        for name, obs in self._embodiment.observations.items():
            read_started_ns = time.perf_counter_ns()
            message = self.observations[name].read()
            if step_ms is not None:
                step_ms[telemetry_keys.ATTR_STEP_READ_MS_PREFIX + name] = (
                    time.perf_counter_ns() - read_started_ns
                ) / 1e6
            if message is None or isinstance(message.data, pimm.SignalError):
                self._obs_by_signal[name] = message.data if message is not None else None
            elif message.updated or not isinstance(self._obs_by_signal.get(name), dict):
                self._obs_by_signal[name] = self._convert_and_record(name, message, obs.serializer, step_ms)
        if self._writer is not None:
            for name, spec in self._privileged.items():
                message = self.privileged[name].read()
                if message is None or isinstance(message.data, pimm.SignalError):
                    continue
                if message.updated or name not in self._recorded_privileged:
                    self._convert_and_record(name, message, spec.serializer)
                    self._recorded_privileged.add(name)

    def read_obs(self, task: Task, step_ms: dict[str, float]) -> Obs | None:
        """Read sensors, reusing each signal's serialized fields until a new message arrives.

        Copy updated arrays because devices may reuse their buffers while inference still reads them.
        Return ``None`` if a required signal has no message. Raise the ``pimm.SignalError`` a signal carries.
        Conversion errors propagate.
        Put each signal's read and conversion durations into ``step_ms``.
        """
        assert_default_frame(self._statics())
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
        inputs[keys.TASK] = task.instruction
        inputs[keys.DESCRIPTOR] = self._embodiment.descriptor
        return frozen_view(inputs)

    def _step(self, task: Task, runtime: Executor, policy_run: PolicyRun, due_ns: int | None) -> int | None:
        """Read sensors, call the policy, emit commands, and return its clamped next wake-up time.

        ``due_ns`` is the wake-up time the previous step returned, or ``None`` for the first step. The step span
        records the step's durations.
        """
        step_ms: dict[str, float] = {}
        with telemetry.span(telemetry_keys.SPAN_HARNESS_STEP) as span:
            try:
                if due_ns is not None and runtime.time_ns >= due_ns:
                    step_ms[telemetry_keys.ATTR_STEP_LATE_MS] = (runtime.time_ns - due_ns) / 1e6
                observe_started_ns = time.perf_counter_ns()
                obs = self.read_obs(task, step_ms)
                policy_started_ns = time.perf_counter_ns()
                step_ms[telemetry_keys.ATTR_STEP_OBSERVE_MS] = (policy_started_ns - observe_started_ns) / 1e6
                if obs is None:
                    return None
                runtime.start_tick()
                started_at_ns = runtime.time_ns
                step = policy_run.send(obs)
                assert step is not None, 'a policy must yield a Step for each observation'
                emit_started_ns = time.perf_counter_ns()
                step_ms[telemetry_keys.ATTR_STEP_POLICY_MS] = (emit_started_ns - policy_started_ns) / 1e6
                self._telemetry.step()
                messages = {name: self.commands[name].emit(value) for name, value in step.commands.items()}
                step_ms[telemetry_keys.ATTR_STEP_EMIT_MS] = (time.perf_counter_ns() - emit_started_ns) / 1e6
                if self._writer is not None:
                    for name, message in messages.items():
                        self._convert_and_record(name, message, self._embodiment.commands[name].serializer)
            finally:
                telemetry.set_attrs(span, **step_ms)
        period_sec = (step.resume_at_ns - started_at_ns) / 1e9
        period_sec = min(MAX_POLL_PERIOD_SEC, max(MIN_POLL_PERIOD_SEC, period_sec))
        return started_at_ns + round(period_sec * 1e9)

    @staticmethod
    def _trial_terminal(
        done: pimm.Message[dict] | None, now_ns: int, deadline_ns: int | None, timeline: str
    ) -> dict[str, Any] | None:
        """A done signal timestamped after the deadline counts as a timeout, not a success."""
        if done is not None and done.data and (deadline_ns is None or done.time[timeline] <= deadline_ns):
            return {**done.data, eval_keys.TERMINATED: True}
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
    def _recording(self, path: Path | None) -> Iterator[None]:
        try:
            with ExitStack() as stack:
                if path is not None and self._dataset_factory is not None:
                    with telemetry.span(telemetry_keys.SPAN_RECORD_IO):
                        dataset = stack.enter_context(self._dataset_factory(path))
                        writer = dataset.new_episode()
                        self._writer = writer.__enter__()
                        stack.push(telemetry.traced(telemetry_keys.SPAN_RECORD_IO)(writer.__exit__))
                yield
        finally:
            self._writer = None
            self._recorded_privileged.clear()
            self._obs_by_signal.clear()

    def _rollout(
        self,
        should_stop: pimm.SignalReceiver,
        task: Task,
        runtime: Executor,
        policy_run: PolicyRun,
        deadline_ns: int | None,
    ) -> pimm.Run[dict[str, Any] | None]:
        payload = None
        resume_at_ns = None
        sample_at_ns = runtime.time_ns
        completed = ()
        while not should_stop.value and payload is None:
            sampled_ns = runtime.time_ns
            if completed or resume_at_ns is None or sampled_ns >= resume_at_ns:
                resume_at_ns = self._step(task, runtime, policy_run, resume_at_ns)
                sample_at_ns = sampled_ns + self._recording_period_ns
            elif self._writer is not None and sampled_ns >= sample_at_ns:
                self._sample()
                sample_at_ns = sampled_ns + self._recording_period_ns
            # No complete observation yet: read the sensors again after one poll period.
            wake_at_ns = runtime.time_ns + round(POLL_PERIOD_SEC * 1e9) if resume_at_ns is None else resume_at_ns
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
                runtime.time_ns,
                deadline_ns,
                EMITTED_WORLD if self._embodiment.simulated else EMITTED_WALL,
            )
        if self._writer is not None:
            self._sample()
        return payload

    def _run_episode(
        self, clock: pimm.Clock, should_stop: pimm.SignalReceiver, rollout: Rollout
    ) -> pimm.Run[dict[str, Any] | None]:
        """Prepare, run, record, and return the rig. Return None when stopped without a terminal payload."""
        task = rollout.task
        self._telemetry.begin(task.meta)
        with telemetry.span(telemetry_keys.SPAN_RESET):
            yield from self._prepare(should_stop, task.prepare_args)
        if should_stop.value:
            return None

        runtime = Executor(
            clock.now_ns, simulated=self._embodiment.simulated, charge_inference_time=task.charge_inference_time
        )
        policy_run = None
        try:
            policy_run = runtime.start(rollout.policy)
            deadline_ns = clock.now_ns() + round(task.timeout_sec * 1e9) if task.timeout_sec is not None else None
            self.deadline_ns.emit(deadline_ns)
            self._telemetry.start_rollout(clock.now())
            try:
                with self._recording(rollout.output_path):
                    payload = yield from self._rollout(should_stop, task, runtime, policy_run, deadline_ns)
                    if self._writer is not None:
                        for name, value in (self._build_episode_meta(rollout, runtime) | (payload or {})).items():
                            self._writer.set_static(name, value)
            finally:
                self.deadline_ns.emit(None)
        finally:
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
        for emitter in self.commands.values():
            # A command without a device receiver still needs the world's emission timestamps.
            emitter._clock = clock
        # Episode spans must end before leaving the scope that closes the telemetry provider.
        with telemetry.bind_from_env(telemetry_keys.HARNESS_PROCESS):
            while not should_stop.value:
                call = next(self.perform_task.incoming(), None)
                manual = pimm.value_updated(self.manual_command)
                # An idle done signal must not terminate the next episode.
                pimm.read_updated(self.done)
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
