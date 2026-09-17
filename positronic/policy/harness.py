import logging
from collections.abc import Generator, Iterator
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

    TODO: Migrate rollout callers to processor definitions and remove caller-side episode cleanup.
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
        self._call: pimm.calls.Call[Rollout, dict[str, Any]] | None = None
        self._runtime: Executor | None = None
        self._policy_run: PolicyRun | None = None
        self._obs_by_signal: dict[str, dict[str, Any]] = {}
        self._deadline_ns: int | None = None
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

    @property
    def _task(self) -> Task:
        assert self._call is not None, 'only a live episode has a task'
        return self._call.request.task

    @property
    def _charges_wall_time(self) -> bool:
        return self._task.charge_inference_time or not self._embodiment.simulated

    def _sleep(self, delay_sec: float = POLL_PERIOD_SEC) -> pimm.Command:
        """Yield one simulator tick, or sleep for ``delay_sec`` on a real rig."""
        return pimm.Yield() if self._embodiment.simulated else pimm.Sleep(delay_sec)

    def _ready(self, should_stop: pimm.SignalReceiver, args: dict[str, Any]) -> Iterator[pimm.Command]:
        """Prepare the named devices and wait for all of them, unless shutdown interrupts the wait."""
        unknown = sorted(set(args) - set(self.prepare))
        if unknown:
            rig = self._embodiment.descriptor or 'this rig'
            raise ValueError(f'{unknown} is not something {rig} readies; it readies {sorted(self.prepare)}')
        ready = pimm.calls.all_of([self.prepare[name](arg) for name, arg in args.items()])
        while not ready.done() and not should_stop.value:
            yield self._sleep()
        if ready.done():
            ready.result()

    def _set_deadline(self, deadline_ns: int | None) -> None:
        """Keep the enforced and published deadline in sync."""
        self._deadline_ns = deadline_ns
        self.deadline_ns.emit(deadline_ns)

    def _begin_episode(
        self, clock: pimm.Clock, should_stop: pimm.SignalReceiver, call: pimm.calls.Call[Rollout, dict[str, Any]]
    ) -> Iterator[pimm.Command]:
        """Prepare the rig, start the policy generator, and open recording and the trial budget."""
        # Retain the call before setup, so a setup failure can answer its caller.
        self._call = call
        self._telemetry.begin(self._task.meta)
        with telemetry.span(telemetry_keys.SPAN_RESET):
            # An empty ask answers at once, so the episode would open on a rig that no device moved.
            if self.prepare and not self._task.prepare_args:
                rig = self._embodiment.descriptor or 'this rig'
                raise ValueError(
                    f'The trial readies nothing on {rig}, which readies {sorted(self.prepare)}; '
                    'name at least one of them in prepare_args'
                )
            yield from self._ready(should_stop, self._task.prepare_args)
        if should_stop.value:
            return
        self._runtime = Executor(
            clock.now_ns, simulated=self._embodiment.simulated, charge_inference_time=self._charges_wall_time
        )
        self._policy_run = self._runtime.start(call.request.policy)
        budget = self._task.timeout_sec
        self._set_deadline(clock.now_ns() + round(budget * 1e9) if budget is not None else None)
        self._telemetry.start_rollout(clock.now())
        self.ds_command.emit(DsWriterCommand.START(call.request.output_path))

    def _statics(self) -> dict[str, Any]:
        return self._embodiment.static_meta | self._static_meta | self.robot_meta_in.value

    def _build_episode_meta(self) -> dict[str, Any]:
        meta = self._statics()
        meta[eval_keys.UNIVERSE] = 'sim' if self._embodiment.simulated else 'real'
        meta[eval_keys.EMBODIMENT] = self._embodiment.descriptor
        meta[eval_keys.CHARGE_INFERENCE_TIME] = self._charges_wall_time
        if self._task.timeout_sec is not None:  # the recorder takes no nulls, and an unbounded episode has none
            meta[eval_keys.TIMEOUT] = self._task.timeout_sec
        assert self._call is not None, 'only a live episode has policy meta'
        for k, v in flatten_dict(self._call.request.policy.meta()).items():
            meta[f'{policy_keys.POLICY_META}.{k}'] = v
        meta.update(self._task.meta)
        meta[keys.TASK] = self._task.instruction
        return meta

    def _close_policy(self) -> None:
        """Stop workers before closing the policy they may still use. A close failure stops cleanup."""
        if self._runtime is not None:
            logging.info('Closing the policy runtime')
            self._runtime.close()
            self._runtime = None
            logging.info('Policy runtime closed')
        if self._policy_run is not None:
            logging.info('Closing the policy')
            self._policy_run.close()
            self._policy_run = None
            logging.info('Policy closed')
        self._obs_by_signal.clear()

    def _end_episode(
        self, clock: pimm.Clock, should_stop: pimm.SignalReceiver, payload: dict[str, Any] | None = None
    ) -> Iterator[pimm.Command]:
        """Stop recording and close the policy. A terminal payload also returns the rig and answers the call."""
        self._set_deadline(None)
        self.ds_command.emit(DsWriterCommand.STOP({**self._build_episode_meta(), **(payload or {})}))
        self._close_policy()
        virtual_now = clock.now()
        # Let the recorder consume STOP while its flush still belongs to the episode span.
        yield self._sleep()
        self._telemetry.end(virtual_now)

        if payload is None:
            return
        back_args = {k: v for k, v in self._task.prepare_args.items() if k != eval_keys.SCENE}
        # rules-allow: swallowed-error — the move back is cleanup, and the recording is already complete.
        try:
            yield from self._ready(should_stop, back_args)
        except Exception as exc:
            logging.error(f'The rig failed to go back after the episode: {exc}')
        assert self._call is not None, 'an episode exists only for the call that asked for it'
        self._call.set_result(payload)
        self._call = None

    def _read_obs(self) -> Obs | None:
        """Read sensors, reusing each signal's serialized fields until a new message arrives.

        Copy updated arrays because devices may reuse their buffers while inference still reads them.
        Return ``None`` if any required observation is unavailable.
        """
        inputs: dict[str, Any] = {}
        try:
            assert_default_frame(self._statics())
            for name, obs in self._embodiment.observations.items():
                message = self.observations[name].read()
                if message is None:
                    return None
                if message.updated or name not in self._obs_by_signal:
                    self._obs_by_signal.pop(name, None)
                    value = message.data
                    if obs.serializer is not None:
                        value = obs.serializer(value)
                    self._obs_by_signal[name] = {
                        full_name: entry.copy() if isinstance(entry, np.ndarray) else entry
                        for full_name, entry in expand_suffixed(name, value)
                        if entry is not None
                    }
                inputs.update(self._obs_by_signal[name])
            inputs[keys.TASK] = self._task.instruction
            inputs[keys.DESCRIPTOR] = self._embodiment.descriptor
        except pimm.NoValueException:
            return None
        return frozen_view(inputs)

    def _step(self, clock: pimm.Clock) -> int | None:
        """Read sensors, call the policy, emit commands, and return its clamped next wake-up time."""
        obs = self._read_obs()
        if obs is None:
            return None
        assert self._runtime is not None and self._policy_run is not None, 'only a live episode calls the policy'
        self._runtime.start_tick()
        started_at_ns = clock.now_ns()
        step = self._policy_run.send(obs)
        assert step is not None, 'a policy must yield a Step for each observation'
        self._telemetry.step()
        for name, value in step.commands.items():
            self.commands[name].emit(value)
        period_sec = (step.resume_at_ns - started_at_ns) / 1e9
        period_sec = min(MAX_POLL_PERIOD_SEC, max(MIN_POLL_PERIOD_SEC, period_sec))
        return started_at_ns + round(period_sec * 1e9)

    def _trial_terminal(self, done: pimm.Message[dict] | None, clock: pimm.Clock) -> dict[str, Any] | None:
        """A done signal timestamped after the deadline counts as a timeout, not a success."""
        deadline_ns = self._deadline_ns
        if done is not None and done.data and (deadline_ns is None or done.ts <= deadline_ns):
            return {**done.data, eval_keys.TERMINATED: True}
        if deadline_ns is not None and clock.now_ns() >= deadline_ns:
            return {eval_keys.TERMINATED: False}
        return None

    def _wait_for_next_tick(
        self, should_stop: pimm.SignalReceiver, clock: pimm.Clock, resume_at_ns: int | None
    ) -> Generator[pimm.Command, None, tuple[Answer[Any], ...]]:
        """Return completions before advancing time; otherwise follow simulator ticks or poll real time."""
        if self._runtime is not None:
            while not should_stop.value:
                result = self._runtime.wait(timeout_sec=POLL_PERIOD_SEC)
                match result.status:
                    case WaitStatus.ANSWERS_READY:
                        return result.completed
                    case WaitStatus.CAN_ADVANCE:
                        break
                    case WaitStatus.TIMED_OUT:
                        continue
        if should_stop.value:
            return ()
        if resume_at_ns is None:
            resume_at_ns = clock.now_ns() + round(POLL_PERIOD_SEC * 1e9)
        # A positive real-time sleep gives this loop its own wake-up, independent of other loops' timers.
        delay_ns = max(1, resume_at_ns - clock.now_ns())
        if self._runtime is not None and self._runtime.has_pending:
            delay_ns = min(delay_ns, round(MIN_POLL_PERIOD_SEC * 1e9))
        yield self._sleep(delay_ns / 1e9)
        return self._runtime.take_completed() if self._runtime is not None else ()

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        # Episode spans must end before leaving the scope that closes the telemetry provider.
        with telemetry.bind_from_env(telemetry_keys.HARNESS_PROCESS):
            try:
                resume_at_ns = None
                completed = ()
                while not should_stop.value:
                    call = next(self.perform_task.incoming(), None)
                    # Consume idle done signals and in-episode manual commands so neither leaks into a later state.
                    manual = pimm.value_updated(self.manual_command)
                    done = pimm.read_updated(self.done)

                    if self._call is not None:
                        if call is not None:
                            call.set_exception(RuntimeError('An episode is already running'))
                        if (terminal := self._trial_terminal(done, clock)) is not None:
                            yield from self._end_episode(clock, should_stop, terminal)
                    elif call is not None:
                        yield from self._begin_episode(clock, should_stop, call)
                        resume_at_ns = None
                    elif manual is not None:
                        for name, value in manual.items():
                            self.commands[name].emit(value)

                    if self._policy_run is None:
                        resume_at_ns = None
                    elif completed or resume_at_ns is None or clock.now_ns() >= resume_at_ns:
                        resume_at_ns = self._step(clock)
                    completed = yield from self._wait_for_next_tick(should_stop, clock, resume_at_ns)

                if self._policy_run is not None:
                    yield from self._end_episode(clock, should_stop)
            finally:
                # Cleanup intentionally stops at the first error. The run is ending, so remaining resource
                # closure, telemetry, and replies are not guaranteed; do not enforce them with nested finally blocks.
                self._close_policy()
                self._telemetry.end(clock.now(), partial=True)
                if self._call is not None:
                    self._call.set_exception(pimm.calls.HandlerStopped())
