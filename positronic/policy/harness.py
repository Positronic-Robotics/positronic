import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from opentelemetry.trace import Span

import pimm
from positronic import keys, telemetry, telemetry_keys
from positronic.dataset.ds_writer_agent import DsWriterCommand
from positronic.dataset.serializers import expand_suffixed
from positronic.drivers.roboarm.ik import assert_default_frame
from positronic.eval import Embodiment, Task
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

    Each ``perform_task`` call runs one ``Rollout`` until its deadline or a truthy ``done`` signal.
    Its answer carries the terminal payload. Between episodes, manual commands pass through.
    """

    def __init__(self, embodiment: Embodiment, *, static_meta: dict[str, Any] | None = None):
        self._embodiment = embodiment
        self._static_meta = static_meta or {}
        self._obs_by_signal: dict[str, dict[str, Any]] = {}
        self._telemetry = _EpisodeTelemetry()

        self.observations = pimm.ReceiverDict(self, names=embodiment.observations)
        self.commands = pimm.EmitterDict(self, names=embodiment.commands)
        self.prepare = pimm.calls.CallerDict[Any, None](self, names=embodiment.prepare_handlers)

        self.perform_task = pimm.calls.ControlSystemHandler[Rollout, dict[str, Any]](self)
        self.manual_command = pimm.ControlSystemReceiver(self)
        self.ds_command = pimm.ControlSystemEmitter[DsWriterCommand](self)
        self.deadline_ns = pimm.ControlSystemEmitter[int | None](self)
        self.robot_meta_in = pimm.DefaultingReceiver(self, default={})
        self.done = pimm.DefaultingReceiver[dict](self, default={})

    def _yield(self, delay_sec: float = POLL_PERIOD_SEC) -> pimm.Command:
        """Yield one simulator tick, or sleep for ``delay_sec`` on a real rig."""
        return pimm.Yield() if self._embodiment.simulated else pimm.Sleep(delay_sec)

    def _ready(self, should_stop: pimm.SignalReceiver, args: dict[str, Any]) -> pimm.Run[None]:
        """Prepare only the named devices and wait for them. Empty args leave every device as it is."""
        unknown = sorted(set(args) - set(self.prepare))
        if unknown:
            rig = self._embodiment.descriptor or 'this rig'
            raise ValueError(f'{unknown} is not something {rig} readies; it readies {sorted(self.prepare)}')
        ready = pimm.calls.all_of([self.prepare[name](arg) for name, arg in args.items()])
        while not ready.done() and not should_stop.value:
            yield self._yield()
        if ready.done():
            ready.result()

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

    def _read_obs(self, task: Task, step_ms: dict[str, float]) -> Obs | None:
        """Read sensors, reusing each signal's serialized fields until a new message arrives.

        Copy updated arrays because devices may reuse their buffers while inference still reads them.
        Return ``None`` if a required signal has no message. Conversion errors propagate.
        Put each signal's read and conversion durations into ``step_ms``.
        """
        inputs: dict[str, Any] = {}
        assert_default_frame(self._statics())
        for name, obs in self._embodiment.observations.items():
            read_started_ns = time.perf_counter_ns()
            message = self.observations[name].read()
            convert_started_ns = time.perf_counter_ns()
            step_ms[telemetry_keys.ATTR_STEP_READ_MS_PREFIX + name] = (convert_started_ns - read_started_ns) / 1e6
            if message is None:
                return None
            if message.updated or name not in self._obs_by_signal:
                value = message.data
                if obs.serializer is not None:
                    value = obs.serializer(value)
                self._obs_by_signal[name] = {
                    full_name: entry.copy() if isinstance(entry, np.ndarray) else entry
                    for full_name, entry in expand_suffixed(name, value)
                    if entry is not None
                }
                converted_ns = time.perf_counter_ns() - convert_started_ns
                step_ms[telemetry_keys.ATTR_STEP_CONVERT_MS_PREFIX + name] = converted_ns / 1e6
            inputs.update(self._obs_by_signal[name])
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
                obs = self._read_obs(task, step_ms)
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
                for name, value in step.commands.items():
                    self.commands[name].emit(value)
                step_ms[telemetry_keys.ATTR_STEP_EMIT_MS] = (time.perf_counter_ns() - emit_started_ns) / 1e6
            finally:
                telemetry.set_attrs(span, **step_ms)
        period_sec = (step.resume_at_ns - started_at_ns) / 1e9
        period_sec = min(MAX_POLL_PERIOD_SEC, max(MIN_POLL_PERIOD_SEC, period_sec))
        return started_at_ns + round(period_sec * 1e9)

    @staticmethod
    def _trial_terminal(done: pimm.Message[dict] | None, now_ns: int, deadline_ns: int | None) -> dict[str, Any] | None:
        """A done signal timestamped after the deadline counts as a timeout, not a success."""
        if done is not None and done.data and (deadline_ns is None or done.ts <= deadline_ns):
            return {**done.data, eval_keys.TERMINATED: True}
        if deadline_ns is not None and now_ns >= deadline_ns:
            return {eval_keys.TERMINATED: False}
        return None

    def _wait_for_next_tick(self, runtime: Executor, resume_at_ns: int | None) -> pimm.Run[tuple[Answer[Any], ...]]:
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
        if resume_at_ns is None:
            resume_at_ns = runtime.time_ns + round(POLL_PERIOD_SEC * 1e9)
        # A positive real-time sleep gives this loop its own wake-up, independent of other loops' timers.
        delay_ns = max(1, resume_at_ns - runtime.time_ns)
        if runtime.has_pending:
            delay_ns = min(delay_ns, round(MIN_POLL_PERIOD_SEC * 1e9))
        yield self._yield(delay_ns / 1e9)
        return runtime.take_completed()

    def _run_episode(
        self, clock: pimm.Clock, should_stop: pimm.SignalReceiver, rollout: Rollout
    ) -> pimm.Run[dict[str, Any] | None]:
        """Prepare, run, record, and return the rig. Return None, and abort the recording, when stopped without
        a terminal payload."""
        task = rollout.task
        self._telemetry.begin(task.meta)
        with telemetry.span(telemetry_keys.SPAN_RESET):
            yield from self._ready(should_stop, task.prepare_args)
        if should_stop.value:
            return None

        runtime = Executor(
            clock.now_ns, simulated=self._embodiment.simulated, charge_inference_time=task.charge_inference_time
        )
        policy_run = None
        recording = stopped = False
        try:
            policy_run = runtime.start(rollout.policy)
            deadline_ns = clock.now_ns() + round(task.timeout_sec * 1e9) if task.timeout_sec is not None else None
            self.deadline_ns.emit(deadline_ns)
            self._telemetry.start_rollout(clock.now())
            self.ds_command.emit(DsWriterCommand.START(rollout.output_path))
            recording = True
            payload = None
            resume_at_ns = None
            completed = ()
            while not should_stop.value and payload is None:
                if completed or resume_at_ns is None or runtime.time_ns >= resume_at_ns:
                    resume_at_ns = self._step(task, runtime, policy_run, resume_at_ns)
                wake_at_ns = resume_at_ns
                if deadline_ns is not None:
                    wake_at_ns = deadline_ns if wake_at_ns is None else min(wake_at_ns, deadline_ns)
                completed = yield from self._wait_for_next_tick(runtime, wake_at_ns)
                if call := next(self.perform_task.incoming(), None):
                    call.set_exception(RuntimeError('An episode is already running'))
                pimm.read_updated(self.manual_command)
                payload = self._trial_terminal(pimm.read_updated(self.done), runtime.time_ns, deadline_ns)
            self.deadline_ns.emit(None)
            if payload is not None:
                # The run writes its last episode values as it closes, so it closes before the snapshot.
                policy_run.close()
                self.ds_command.emit(DsWriterCommand.STOP({**self._build_episode_meta(rollout, runtime), **payload}))
                stopped = True
        finally:
            if recording and not stopped:
                self.ds_command.emit(DsWriterCommand.ABORT())
            # Cleanup stops at the first error. Later resources may remain open; do not add nested
            # finally blocks to guarantee their closure.
            if policy_run is not None:
                logging.info('Closing the policy')
                policy_run.close()
                logging.info('Policy closed')
            logging.info('Closing the policy runtime')
            runtime.close()
            logging.info('Policy runtime closed')
            self._obs_by_signal.clear()

        virtual_now = clock.now()
        # Let the recorder consume STOP while its flush still belongs to the episode span.
        yield self._yield()
        self._telemetry.end(virtual_now)
        if payload is not None:
            back_args = {k: v for k, v in task.prepare_args.items() if k != eval_keys.SCENE}
            # rules-allow: swallowed-error — the move back is cleanup, and the recording is already complete.
            try:
                yield from self._ready(should_stop, back_args)
            except Exception as exc:
                logging.error(f'The rig failed to go back after the episode: {exc}')
        return payload

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> pimm.Run[None]:
        # Episode spans must end before leaving the scope that closes the telemetry provider.
        with telemetry.bind_from_env(telemetry_keys.HARNESS_PROCESS):
            while not should_stop.value:
                call = next(self.perform_task.incoming(), None)
                manual = pimm.value_updated(self.manual_command)
                # An idle done signal must not terminate the next episode.
                pimm.read_updated(self.done)
                if call is not None:
                    payload = None
                    try:
                        payload = yield from self._run_episode(clock, should_stop, call.request)
                    finally:
                        self._telemetry.end(clock.now(), partial=True)
                        if payload is None:
                            call.set_exception(pimm.calls.HandlerStopped())
                    if payload is not None:
                        call.set_result(payload)
                elif manual is not None:
                    for name, value in manual.items():
                        self.commands[name].emit(value)
                if not should_stop.value:
                    yield self._yield()
