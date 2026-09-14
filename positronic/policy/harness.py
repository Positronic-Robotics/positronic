import logging
import time
from collections.abc import Callable, Iterator
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
from positronic.policy.base import Policy, Runtime
from positronic.policy.executor import Executor
from positronic.utils import flatten_dict, frozen_view

# Harness wake-up intervals on the world's clock.
POLL_PERIOD_SEC = 0.1
MIN_POLL_PERIOD_SEC = 0.005
MAX_POLL_PERIOD_SEC = 1.0


class Rollout:
    """One trial, its policy factory, and the path the episode records into.

    ``build_policy(runtime)`` creates the episode's policy. The harness owns that policy and its
    runtime. An ``output_path`` of ``None`` records nothing.

    TODO: Migrate rollout callers to factories and remove caller-side episode cleanup.
    """

    def __init__(self, task: Task, build_policy: Callable[[Runtime], Policy], output_path: Path | None):
        self.task = task
        self.build_policy = build_policy
        self.output_path = output_path


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
        virtual_s = max(virtual_now - self._virtual_start, 0.0) if self._virtual_start is not None else 0.0
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

    The policy sets the next wake-up time, clamped to 5 ms–1 s from now. Without a policy step, the
    harness polls every 100 ms. Both intervals use the world's clock in simulation and on a real rig.

    Each ``perform_task`` call runs one ``Rollout`` until its deadline or a truthy ``done`` signal.
    Its answer carries the terminal payload. Between episodes, manual commands pass through.
    """

    def __init__(self, embodiment: Embodiment, *, static_meta: dict[str, Any] | None = None):
        self._embodiment = embodiment
        self._static_meta = static_meta or {}
        self._call: pimm.calls.Call[Rollout, dict[str, Any]] | None = None
        self._runtime: Executor | None = None
        self._policy: Policy | None = None
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

    def _ready(self, should_stop: pimm.SignalReceiver, args: dict[str, Any]) -> Iterator[pimm.Command]:
        """Prepare the named devices and wait for all of them, unless shutdown interrupts the wait."""
        unknown = sorted(set(args) - set(self.prepare))
        if unknown:
            rig = self._embodiment.descriptor or 'this rig'
            raise ValueError(f'{unknown} is not something {rig} readies; it readies {sorted(self.prepare)}')
        ready = pimm.calls.all_of([self.prepare[name](arg) for name, arg in args.items()])
        while not ready.done() and not should_stop.value:
            yield pimm.Sleep(POLL_PERIOD_SEC)
        if ready.done():
            ready.result()

    def _set_deadline(self, deadline_ns: int | None) -> None:
        """Keep the enforced and published deadline in sync."""
        self._deadline_ns = deadline_ns
        self.deadline_ns.emit(deadline_ns)

    def _begin_episode(
        self, clock: pimm.Clock, should_stop: pimm.SignalReceiver, call: pimm.calls.Call[Rollout, dict[str, Any]]
    ) -> Iterator[pimm.Command]:
        """Prepare the rig, construct the policy, and start recording and the trial budget."""
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
            clock, simulated=self._embodiment.simulated, charge_inference_time=self._charges_wall_time
        )
        self._policy = call.request.build_policy(self._runtime)
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
        assert self._policy is not None, 'only a live episode has policy meta'
        for k, v in flatten_dict(self._policy.meta()).items():
            meta[f'{policy_keys.POLICY_META}.{k}'] = v
        meta.update(self._task.meta)
        meta[keys.TASK] = self._task.instruction
        return meta

    def _close_policy(self) -> None:
        """Stop workers before closing the policy they may still use. A close failure stops cleanup."""
        runtime, self._runtime = self._runtime, None
        policy, self._policy = self._policy, None
        if runtime is not None:
            logging.info('Closing the policy runtime')
            runtime.close()
            logging.info('Policy runtime closed')
        if policy is not None:
            logging.info('Closing the policy')
            policy.close()
            logging.info('Policy closed')

    def _end_episode(
        self, clock: pimm.Clock, should_stop: pimm.SignalReceiver, payload: dict[str, Any] | None = None
    ) -> Iterator[pimm.Command]:
        """Stop recording and close the policy. A terminal payload also returns the rig and answers the call."""
        self._set_deadline(None)
        self.ds_command.emit(DsWriterCommand.STOP({**self._build_episode_meta(), **(payload or {})}))
        self._close_policy()
        virtual_now = clock.now()
        # Let the recorder consume STOP while its flush still belongs to the episode span.
        yield pimm.Sleep(POLL_PERIOD_SEC)
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

    def _step(self, clock: pimm.Clock) -> int | None:
        """Read sensors, call the policy, emit commands, and return its next wake-up time.

        Copies arrays because a device may reuse its buffer while submitted inference still reads it.
        Missing observations defer the policy call.
        """
        inputs: dict[str, Any] = {}
        try:
            assert_default_frame(self._statics())
            for name, obs in self._embodiment.observations.items():
                message = self.observations[name].read()
                if message is None:
                    return None
                value = message.data
                if obs.serializer is not None:
                    value = obs.serializer(value)
                inputs.update({
                    full: v.copy() if isinstance(v, np.ndarray) else v
                    for full, v in expand_suffixed(name, value)
                    if v is not None
                })
            inputs[keys.TASK] = self._task.instruction
            inputs[keys.WALL_TIME_NS] = time.time_ns()
            inputs[keys.OBS_TIME_NS] = clock.now_ns()
            inputs[keys.DESCRIPTOR] = self._embodiment.descriptor
        except pimm.NoValueException:
            return None

        assert self._runtime is not None and self._policy is not None, 'only a live episode calls the policy'
        self._runtime.start_tick()
        step = self._policy(frozen_view(inputs))
        self._telemetry.step()
        for name, value in step.commands.items():
            self.commands[name].emit(value)
        return step.resume_at_ns

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
    ) -> Iterator[pimm.Command]:
        """Let async work reach the next tick before advancing simulated time; sleep on the world's clock."""
        delay_sec = POLL_PERIOD_SEC
        if resume_at_ns is not None:
            delay_sec = (resume_at_ns - clock.now_ns()) / 1e9
            delay_sec = min(MAX_POLL_PERIOD_SEC, max(MIN_POLL_PERIOD_SEC, delay_sec))
        if self._runtime is not None:
            self._runtime.wait(clock.now_ns() + round(delay_sec * 1e9), should_stop)
        if not should_stop.value:
            yield pimm.Sleep(delay_sec)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        # Episode spans must end before leaving the scope that closes the telemetry provider.
        with telemetry.bind_from_env(telemetry_keys.HARNESS_PROCESS):
            try:
                while not should_stop.value:
                    resume_at_ns = None
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
                    elif manual is not None:
                        for name, value in manual.items():
                            self.commands[name].emit(value)

                    if self._policy is not None:
                        resume_at_ns = self._step(clock)
                    yield from self._wait_for_next_tick(should_stop, clock, resume_at_ns)

                if self._policy is not None:
                    yield from self._end_episode(clock, should_stop)
            finally:
                # Cleanup intentionally stops at the first error. The run is ending, so remaining resource
                # closure, telemetry, and replies are not guaranteed; do not enforce them with nested finally blocks.
                self._close_policy()
                self._telemetry.end(clock.now(), partial=True)
                if self._call is not None:
                    self._call.set_exception(pimm.calls.HandlerStopped())
