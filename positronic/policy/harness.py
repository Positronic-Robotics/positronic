import contextlib
import time
from collections.abc import Generator, Iterator, Mapping
from typing import Any

import numpy as np
from opentelemetry.trace import Span

import pimm
from positronic import keys, telemetry, telemetry_keys
from positronic.dataset.ds_writer_agent import DsWriterCommand
from positronic.dataset.serializers import expand_suffixed
from positronic.drivers.roboarm.ik import assert_default_frame
from positronic.eval import Embodiment, Task
from positronic.policy.base import Policy
from positronic.policy.executor import Executor
from positronic.utils import flatten_dict, frozen_view

# How long the harness waits between looks at a call it made.
WAIT_PERIOD_SEC = 0.01
# The shortest and the longest real-time round. The floor caps how fast a session drives the loop, the
# ceiling how late the harness reads ``done`` and the stop signal.
MIN_ROUND_SEC = 0.001
MAX_ROUND_SEC = 1.0


class _EpisodeInference:
    """One episode: the work the policy opens, the runtime that serves it, and the session that plays it."""

    def __init__(self, policy: Policy, context: dict[str, Any], charges_wall_time: bool, clock: pimm.Clock) -> None:
        self._charges_wall_time = charges_wall_time
        self._clock = clock
        # One instant on two clocks, so ``wait`` adds a wall duration to a world instant.
        self._t0_ns, self._wall_t0 = clock.now_ns(), time.monotonic()
        self._closing = contextlib.ExitStack()
        try:
            self._runtime = Executor(self._closing.enter_context(policy.episode(context)))
            self._closing.callback(self._runtime.close)
            self._session = policy.new_session(self._runtime)
            self._closing.callback(self._session.close)
        except BaseException:
            self._closing.close()
            raise

    @staticmethod
    def _owned(obs: dict[str, Any]) -> dict[str, Any]:
        """The observation with its arrays copied, so nothing rewrites what a function is still reading.

        A camera renders into the array behind the adapter it re-emits, and it keeps stepping while the
        function runs.
        """
        return {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in obs.items()}

    def __call__(self, obs: dict[str, Any]) -> tuple[Mapping[str, Any], int]:
        now_ns = self._clock.now_ns()
        # A call that joins work already in flight keeps its anchor, so the trial pays for that work one time.
        if not self._runtime.in_flight:
            self._t0_ns, self._wall_t0 = now_ns, time.monotonic()
        commands, resume_at_ns = self._session(frozen_view(self._owned(obs)), now_ns)
        assert resume_at_ns > now_ns, f'resume time must be in the future: {resume_at_ns} <= {now_ns}'
        return commands, resume_at_ns

    def wait(self, should_stop: pimm.SignalReceiver[bool]) -> None:
        """Wait for the function in flight, for as long as the trial charges the loop for it."""
        if self._charges_wall_time:
            # Wall time cannot be held still, so the loop waits out only the time the world is already ahead by.
            paid_through = self._t0_ns / 1e9 + (time.monotonic() - self._wall_t0)
            self._runtime.wait(max(self._clock.now() - paid_through, 0.0))
            return
        # A trial that pays nothing waits the function out. It waits in steps, because a model that never
        # answers must not also keep the world from coming down.
        while self._runtime.in_flight and not should_stop.value:
            self._runtime.wait(WAIT_PERIOD_SEC)

    def close(self) -> None:
        """End the session, then the runtime, then the work the episode opened.

        Until ``Executor.close`` returns, the function in flight still holds the episode's websocket or model.
        """
        self._closing.close()


class _EpisodeTelemetry:
    """The live rollout's episode span, with the index, step count and virtual start it is stamped with.
    Inert while telemetry is unbound, so the harness calls it unconditionally. The span stays anchored while
    open, so the rollout's phase spans parent to it rather than to the pass."""

    def __init__(self) -> None:
        self._span: Span | None = None
        self._index = -1
        self._steps = 0
        # ``None`` until the rollout starts, so the prepare is excluded and one that fails leaves it unstamped.
        self._virtual_start: float | None = None

    def begin(self, params: dict[str, Any]) -> None:
        """Open the episode span, stamped with the index and the trial's flat ``params``. Called before the
        scene reset, so the reset is timed inside the rollout it belongs to."""
        self._index += 1
        self._steps = 0
        self._virtual_start = None
        attrs: dict[str, Any] = {telemetry_keys.ATTR_EPISODE_INDEX: self._index}
        attrs.update({k: v for k, v in params.items() if isinstance(v, (bool, int, float, str))})
        self._span = telemetry.start_span(telemetry_keys.SPAN_EPISODE, **attrs)
        telemetry.push_anchor(self._span)

    def start_rollout(self, virtual_now: float) -> None:
        """Anchor the rollout's virtual duration at the instant the rig finished being readied."""
        self._virtual_start = virtual_now

    def step(self) -> None:
        """Count one control round: the harness read an observation, called the session and emitted its answer."""
        self._steps += 1

    def end(self, virtual_now: float) -> None:
        """Close the rollout, stamped with its step count and its virtual duration up to ``virtual_now`` —
        captured when the rollout ended, before the flush round advances the sim clock."""
        if self._span is None:
            return
        self._close(virtual_now)
        telemetry.force_flush()

    def seal(self, virtual_now: float) -> None:
        """Close a rollout abandoned mid-flight by a raising ``reset`` / ``new_session`` / session call, marked
        ``episode.partial`` so the reduce keeps it. Ending it is what exports it: the batch processor drops an
        unended span, orphaning the finished children and losing their phases."""
        if self._span is None:
            return
        telemetry.set_attrs(self._span, **{telemetry_keys.ATTR_EPISODE_PARTIAL: True})
        self.end(virtual_now)

    def _close(self, virtual_now: float) -> None:
        # A rollout that never started — a prepare that raised — has zero virtual duration.
        virtual_s = max(virtual_now - self._virtual_start, 0.0) if self._virtual_start is not None else 0.0
        assert self._span is not None
        attrs = {telemetry_keys.ATTR_EPISODE_STEPS: self._steps, telemetry_keys.ATTR_EPISODE_VIRTUAL_S: virtual_s}
        telemetry.set_attrs(self._span, **attrs)
        self._end_span()

    def _end_span(self) -> None:
        assert self._span is not None
        self._span.end()
        telemetry.pop_anchor(self._span)
        self._span = None


class Harness(pimm.ControlSystem):
    """Control system that runs the episode lifecycle and emits what the policy commands.

    The session owns the trajectory and answers the commands to run now; the harness emits them and comes
    back when the session asked. The session is called on the loop thread and answers inside the round; the
    functions it starts run on the runtime's own, so playing continues while the model runs. That work costs
    the trial either the wall time it took or nothing, with the world held still for it.

    An episode runs one ``Task``, asked for by a ``perform_task`` call and answered with the terminal
    payload it ended on. The task's ``timeout_sec`` bounds it and a truthy ``done`` within budget ends it
    early — ``eval.terminated`` records which; a task without one ends on ``done`` alone. Which tasks run,
    and in what order, belongs to whoever makes the calls.
    """

    def __init__(self, policy: Policy, embodiment: Embodiment, *, static_meta: dict[str, Any] | None = None):
        self._embodiment = embodiment
        self.policy: Policy = policy
        self._static_meta = static_meta or {}
        # This episode's session and the runtime serving it. ``None`` while no episode is live.
        self._inference: _EpisodeInference | None = None
        # The call this episode answers when it ends.
        self._call: pimm.calls.Call[Task, dict[str, Any]] | None = None
        # An inference let go of with a function still running.
        self._retired: _EpisodeInference | None = None
        # ``task.timeout_sec``, armed per episode; a task without one has no deadline and ends on ``done`` alone.
        self._deadline_ns: int | None = None
        # When the live session asked for its next call. ``None`` while no session has answered.
        self._resume_at_ns: int | None = None
        # Wall-clock telemetry for the live rollout, opened under ``--timing`` and inert otherwise.
        self._telemetry = _EpisodeTelemetry()

        self.observations = pimm.ReceiverDict(self, names=embodiment.observations)
        self.commands = pimm.EmitterDict(self, names=embodiment.commands)
        self.prepare = pimm.calls.CallerDict[Any, None](self, names=embodiment.prepare_handlers)

        # One episode per call, answered with the terminal payload it ended on.
        self.perform_task = pimm.calls.ControlSystemHandler[Task, dict[str, Any]](self)
        self.manual_command = pimm.ControlSystemReceiver(self)
        self.ds_command = pimm.ControlSystemEmitter[DsWriterCommand](self)
        # The instant on the world's clock the live episode ends at, ``None`` while no deadline stands.
        self.deadline_ns = pimm.ControlSystemEmitter[int | None](self)
        self.robot_meta_in = pimm.DefaultingReceiver(self, default={})
        # Stop-signal: a truthy payload within the trial's budget ends it.
        self.done = pimm.DefaultingReceiver[dict](self, default={})

    def _statics(self) -> dict[str, Any]:
        """What is known about the rig before the episode runs, live values winning."""
        return self._embodiment.static_meta | self._static_meta | self.robot_meta_in.value

    @property
    def _task(self) -> Task:
        """The live episode's task. An episode runs for the call that asked for it, so the call carries it."""
        assert self._call is not None, 'only a live episode has a task'
        return self._call.request

    @property
    def _charges_wall_time(self) -> bool:
        """Whether each model call costs the trial the wall time it took. A real rig has no other option."""
        return self._task.charge_inference_time or not self._embodiment.simulated

    def _build_episode_meta(self) -> dict[str, Any]:
        meta = self._statics()
        meta[keys.EVAL_UNIVERSE] = 'sim' if self._embodiment.simulated else 'real'
        meta[keys.EVAL_EMBODIMENT] = self._embodiment.descriptor
        meta[keys.EVAL_CHARGE_INFERENCE_TIME] = self._charges_wall_time
        if self._task.timeout_sec is not None:  # the recorder takes no nulls, and an unbounded episode has none
            meta[keys.EVAL_TIMEOUT] = self._task.timeout_sec
        for k, v in flatten_dict(self.policy.meta).items():
            meta[f'{keys.POLICY_META}.{k}'] = v
        meta.update(self._task.meta)
        meta[keys.TASK] = self._task.instruction
        return meta

    def _emit(self, action: dict[str, Any]) -> None:
        for name, value in action.items():
            self.commands[name].emit(value)

    def _ready(self, should_stop: pimm.SignalReceiver, args: dict[str, Any]) -> Generator[pimm.Command, None, None]:
        """Ask each device in ``args`` for the value it names, and come back once every one has answered."""
        unknown = sorted(set(args) - set(self.prepare))
        if unknown:
            rig = self._embodiment.descriptor or 'this rig'
            raise ValueError(f'{unknown} is not something {rig} readies; it readies {sorted(self.prepare)}')
        ready = pimm.calls.all_of([self.prepare[name](arg) for name, arg in args.items()])
        while not ready.done() and not should_stop.value:
            yield pimm.Yield() if self._embodiment.simulated else pimm.Sleep(WAIT_PERIOD_SEC)
        # An episode must not open on a rig that never got ready, and its asker must hear that rather than wait
        if not ready.done():
            raise RuntimeError('The world stopped before every device was ready')
        ready.result()

    def _pace(self, clock: pimm.Clock) -> pimm.Command:
        """The command that ends this round: a yield in sim, where the simulator's control-period sleep is
        the sole time-master, and a sleep to the moment the live session asked for on a real rig."""
        if self._embodiment.simulated:
            return pimm.Yield()
        if self._resume_at_ns is None:
            return pimm.Sleep(WAIT_PERIOD_SEC)
        until_ns = self._resume_at_ns if self._deadline_ns is None else min(self._resume_at_ns, self._deadline_ns)
        due_sec = (until_ns - clock.now_ns()) / 1e9
        return pimm.Sleep(min(max(due_sec, MIN_ROUND_SEC), MAX_ROUND_SEC))

    def _retire_inference(self) -> None:
        """Let go of this episode's inference, keeping it for ``_reap_inference``: ending an episode must not
        wait for a model that hangs."""
        if self._inference is not None:
            self._retired, self._inference = self._inference, None
        self._resume_at_ns = None

    def _reap_inference(self) -> None:
        if self._retired is not None:
            self._retired.close()
            self._retired = None

    def _finalize_recording(
        self, clock: pimm.Clock, payload: dict[str, Any] | None = None
    ) -> Generator[pimm.Command, None, None]:
        """Commit the live episode: end the session playing it, stop the recorder — stamping the
        episode's full static meta (plus any terminal payload) — then close its span."""
        self._set_deadline(None)
        # Stamped before the inference is retired: the meta overlays what its session reports.
        stop = DsWriterCommand.STOP({**self._build_episode_meta(), **(payload or {})})
        # Retiring the session ends the chunk it was playing; devices hold their last commanded position.
        self._retire_inference()
        self.ds_command.emit(stop)
        virtual_now = clock.now()  # before the round below, whose sim-clock advance belongs to no rollout
        # Give the recorder a round to commit the STOP before the next START (they share ``ds_command``, where
        # last-value-wins would drop one) and before the next trial's prepare, so the moves it asks for stay
        # out of the recording.
        yield self._pace(clock)
        # After that round, so the recorder's STOP-time record.io span still parents to the episode. Skew: a
        # producer stepping in that shared round charges ≤ one control period to the closing episode.
        self._telemetry.end(virtual_now)

    def _set_deadline(self, deadline_ns: int | None) -> None:
        """Arm the live episode's deadline and publish it, so the enforced one and the published one agree."""
        self._deadline_ns = deadline_ns
        self.deadline_ns.emit(deadline_ns)

    def _begin_episode(
        self, clock: pimm.Clock, should_stop: pimm.SignalReceiver, call: pimm.calls.Call[Task, dict[str, Any]]
    ) -> Generator[pimm.Command, None, None]:
        """Open a fresh episode: ready the rig and the scene, read the instruction, open the session, the
        recording and the first inference."""
        # Before anything that can raise, so an episode that fails to open still answers whoever asked for it.
        self._call = call
        # The episode span opens first, so the prepare and the rollout's other phase spans parent to it.
        self._telemetry.begin(self._task.meta)
        with telemetry.span(telemetry_keys.SPAN_RESET):
            yield from self._ready(should_stop, self._task.prepare_args)
        # The reap waits out the function the last episode left running. It runs after the rig is readied, so
        # a model that hangs does not leave the devices at the policy's last setpoint, and before the session
        # below, because an in-process policy is one model across every episode.
        self._reap_inference()
        # Read after the reset: an embodiment that learns its task from the scene reports it only once the
        # scene is set up.
        self._inference = _EpisodeInference(
            self.policy, {keys.TASK: self._task.instruction}, self._charges_wall_time, clock
        )
        budget = self._task.timeout_sec
        self._set_deadline(clock.now_ns() + round(budget * 1e9) if budget is not None else None)
        self._telemetry.start_rollout(clock.now())
        self.ds_command.emit(DsWriterCommand.START())
        # The fresh data is here, later round would read a frame the recording did not open on.
        self._infer(self._inference, clock, should_stop)

    def _end_episode(
        self, clock: pimm.Clock, should_stop: pimm.SignalReceiver, payload: dict[str, Any]
    ) -> Generator[pimm.Command, None, None]:
        """Close the live episode: finalize the recording, put the rig back, retire the session, hand the
        terminal back to whoever asked for the episode.

        The inference is retired rather than closed here, so the episode's websocket outlives the function
        still using it.
        """
        yield from self._finalize_recording(clock, payload)
        # A powered arm holds the policy's last setpoint until the next trial, so each device the trial placed
        # goes back where it put it — the trial's own args, not a fresh draw. The scene is a person's to set
        # up, and is not asked again. The terminal waits on the move: a scene the next trial draws rebuilds
        # the model an unfinished one is still travelling under, which nothing but its timeout would end.
        yield from self._ready(should_stop, {k: v for k, v in self._task.prepare_args.items() if k != keys.SCENE})
        assert self._call is not None, 'an episode exists only for the call that asked for it'
        self._call.set_result(payload)
        self._call = None

    def _fail_call(self, exc: BaseException) -> None:
        """Raise to whoever asked for the live episode, in place of the terminal it will never get."""
        if self._call is not None:
            self._call.set_exception(exc)
            self._call = None

    def _advance_episode(
        self,
        inference: _EpisodeInference,
        done: pimm.Message[dict] | None,
        clock: pimm.Clock,
        should_stop: pimm.SignalReceiver,
    ) -> Generator[pimm.Command, None, None]:
        """One round of the live episode: end it if it is out of budget or done, else run one inference round."""
        if (terminal := self._trial_terminal(done, clock)) is not None:
            yield from self._end_episode(clock, should_stop, terminal)
        else:
            self._infer(inference, clock, should_stop)

    def _build_obs(self, clock: pimm.Clock) -> dict[str, Any]:
        """Read every observation channel and assemble the policy input dict.

        Raises ``NoValueException`` if any channel has no value yet.
        """
        assert_default_frame(self._statics())
        inputs: dict[str, Any] = {}
        for name, obs in self._embodiment.observations.items():
            message = self.observations[name].read()
            if message is None:
                raise pimm.NoValueException
            value = message.data
            if obs.serializer is not None:
                value = obs.serializer(value)
            inputs.update({full: v for full, v in expand_suffixed(name, value) if v is not None})
        inputs[keys.TASK] = self._task.instruction
        inputs[keys.WALL_TIME_NS] = time.time_ns()
        inputs[keys.OBS_TIME_NS] = clock.now_ns()
        inputs[keys.DESCRIPTOR] = self._embodiment.descriptor
        return inputs

    def _infer(self, inference: _EpisodeInference, clock: pimm.Clock, should_stop: pimm.SignalReceiver[bool]) -> None:
        try:
            obs = self._build_obs(clock)
        except pimm.NoValueException:
            return  # no function is in flight yet, so this skips no wait
        commands, self._resume_at_ns = inference(obs)
        self._telemetry.step()
        # The world can reach the deadline while the session runs, and a command placed after the point the
        # trial advertises it stops at outlives the trial; ``_run`` finishes it next round.
        if self._deadline_ns is None or clock.now_ns() < self._deadline_ns:
            # The key-filtered demux: a command this rig declares no channel for reaches no driver.
            self._emit({name: commands[name] for name in self.commands if name in commands})
        inference.wait(should_stop)

    def _trial_terminal(self, done: pimm.Message[dict] | None, clock: pimm.Clock) -> dict[str, Any] | None:
        """The terminal static payload if the live trial has ended this round, else ``None``.

        The deadline is hard: a truthy ``done`` within budget records ``eval.terminated`` True plus its
        payload, the budget passing records False, and a terminal past the deadline is a timeout rather than
        a late success. A task without a timeout has no budget and ends on ``done`` alone. Only a truthy
        ``done`` counts, so a producer can clear a stale terminal off the wire with an empty payload.
        """
        deadline_ns = self._deadline_ns
        if done is not None and done.data and (deadline_ns is None or done.ts <= deadline_ns):
            return {**done.data, keys.EVAL_TERMINATED: True}
        if deadline_ns is not None and clock.now_ns() >= deadline_ns:
            return {keys.EVAL_TERMINATED: False}
        return None

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        try:
            yield from self._run(should_stop, clock)
        except BaseException as exc:
            # Seal the open span before the exception reaches ``bind``'s exit flush: an unended span never
            # exports, orphaning its finished children and charging the episode's wall to between_episodes.
            self._telemetry.seal(clock.now())
            self._fail_call(exc)
            raise
        finally:
            # A no-op once the block above has answered: the world coming down under a live episode is the
            # one way a call goes unanswered, and it is the caller's to hear about.
            self._fail_call(RuntimeError('The world stopped before the episode ended'))
            # An episode abandoned by a raise never reaches ``_finalize_recording``, and would leave a
            # deadline standing that nothing will ever meet.
            self._set_deadline(None)
            self._retire_inference()
            self._reap_inference()

    def _run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        while not should_stop.value:
            # One action per round: start the episode a call asks for, finish one that is out of budget or
            # done, or run one inference round.
            call = next(self.perform_task.incoming(), None)
            # Read every round so the flag clears mid-episode; a press during a trial is consumed, not replayed.
            manual = pimm.value_updated(self.manual_command)
            # Read every round for the same reason: a terminal landing between episodes belongs to none of
            # them, and left on the wire it would end the next one on its first round.
            done = pimm.read_updated(self.done)
            if self._inference is not None:
                if call is not None:  # the live episode is the one that finishes; a second ask is refused
                    call.set_exception(RuntimeError('An episode is already running'))
                yield from self._advance_episode(self._inference, done, clock, should_stop)
            elif call is not None:
                yield from self._begin_episode(clock, should_stop, call)
            elif manual is not None:
                self._emit(manual)
            yield self._pace(clock)

        if self._inference is not None:
            yield from self._finalize_recording(clock)
