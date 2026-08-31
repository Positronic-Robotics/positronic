import time
from collections import deque
from collections.abc import Generator, Iterator
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
from positronic.policy.base import Policy
from positronic.policy.executor import Executor
from positronic.utils import flatten_dict, frozen_view

# How far from now an action may be scheduled: past any real chunk, short of the decades a rig-side stack is
# off by when it leaves timestamps chunk-relative or anchors them twice.
MAX_ACTION_SKEW_SEC = 60.0

# How long a real-time round may last when no waypoint is due sooner. It bounds how late a call is noticed,
# and with it the granularity every command timestamp is quantized to.
POLL_PERIOD_SEC = 0.01


class Rollout:
    """What one ``perform_task`` call asks for: the trial to run, the session that runs it, the runtime that
    serves the policy's functions to that session, and the path the episode records into.

    Whoever asks opens it, and so decides which model runs the trial and where the recording lands. An
    ``output_path`` of ``None`` records nothing. The Harness closes it: the episode it ran, or the ask it refused.
    """

    def __init__(self, task: Task, policy: Policy, output_path: Path | None):
        self.task = task
        self.output_path = output_path
        # The Harness closes this runtime before the session, and charges the trial for the model's time
        # through it. TODO(#661): the framework takes over closing the chain, and only the charge keeps a
        # runtime here.
        self.rt = Executor(policy.functions)
        try:
            self.session = policy.new_session(rt=self.rt)
        except BaseException:
            self.rt.close()
            raise

    def close(self) -> None:
        """Close the runtime, then the session it was serving.

        Until ``Executor.close`` returns, the function in flight still holds the session's websocket or model.
        """
        self.rt.close()
        self.session.close()


class _EpisodeInference:
    """One episode's rollout, charged the way its trial asks and anchored on the world's clock."""

    def __init__(self, rollout: Rollout, charges_wall_time: bool, clock: pimm.Clock) -> None:
        self._rollout = rollout
        self._charges_wall_time = charges_wall_time
        self._clock = clock
        # One instant on two clocks, so ``wait`` adds a wall duration to a world instant.
        self._t0_ns, self._wall_t0 = clock.now_ns(), time.monotonic()

    @property
    def meta(self) -> dict[str, Any]:
        return self._rollout.session.meta

    @staticmethod
    def _owned(obs: dict[str, Any]) -> dict[str, Any]:
        """The observation with its arrays copied, so nothing rewrites what a function is still reading.

        A camera renders into the array behind the adapter it re-emits, and it keeps stepping while the
        function runs.
        """
        return {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in obs.items()}

    def __call__(self, obs: dict[str, Any]) -> list[dict[str, Any]] | None:
        now_ns = self._clock.now_ns()
        # A call that joins work already in flight keeps its anchor, so the trial pays for that work one time.
        if not self._rollout.rt.in_flight:
            self._t0_ns, self._wall_t0 = now_ns, time.monotonic()
        return self._rollout.session(frozen_view(self._owned(obs)), now_ns)

    def wait(self, should_stop: pimm.SignalReceiver[bool]) -> None:
        """Wait for the function in flight, for as long as the trial charges the loop for it."""
        if self._charges_wall_time:
            # Wall time cannot be held still, so the loop waits out only the time the world is already ahead by.
            paid_through = self._t0_ns / 1e9 + (time.monotonic() - self._wall_t0)
            self._rollout.rt.wait(max(self._clock.now() - paid_through, 0.0))
            return
        # A trial that pays nothing waits the function out. It waits in steps, because a model that never
        # answers must not also keep the world from coming down.
        while self._rollout.rt.in_flight and not should_stop.value:
            self._rollout.rt.wait(POLL_PERIOD_SEC)

    def close(self) -> None:
        self._rollout.close()


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
    """Control system that runs the episode lifecycle and plays the policy's trajectory to the drivers.

    The layer owns the trajectory, the harness plays it, one command per channel per round. The session is
    called on the loop thread and answers inside the round; the functions it starts run on the runtime's own,
    so playing continues while the model runs. That work costs the trial either the wall time it took or
    nothing, with the world held still for it.

    An episode runs one ``Rollout``, asked for by a ``perform_task`` call and answered with the terminal
    payload it ended on. The rollout's task ``timeout_sec`` bounds it and a truthy ``done`` within budget
    ends it early — ``eval.terminated`` records which; a task without one ends on ``done`` alone. Which
    tasks run, in what order, and on which model, belongs to whoever makes the calls.
    """

    def __init__(self, embodiment: Embodiment, *, static_meta: dict[str, Any] | None = None):
        self._embodiment = embodiment
        self._static_meta = static_meta or {}
        # This episode's rollout, on the clock and under the charge its trial asks for. ``None`` while no
        # episode is live.
        self._inference: _EpisodeInference | None = None
        # The call this episode answers when it ends.
        self._call: pimm.calls.Call[Rollout, dict[str, Any]] | None = None
        # An inference let go of with a function still running.
        self._retired: _EpisodeInference | None = None
        # ``task.timeout_sec``, armed per episode; a task without one has no deadline and ends on ``done`` alone.
        self._deadline_ns: int | None = None
        # Wall-clock telemetry for the live rollout, opened under ``--timing`` and inert otherwise.
        self._telemetry = _EpisodeTelemetry()

        self.observations = pimm.ReceiverDict(self, names=embodiment.observations)
        self.commands = pimm.EmitterDict(self, names=embodiment.commands)
        self.prepare = pimm.calls.CallerDict[Any, None](self, names=embodiment.prepare_handlers)
        # Each channel's waypoints not yet played, stamped with absolute clock ns and ascending.
        self._schedules: dict[str, deque[tuple[int, Any]]] = {name: deque() for name in embodiment.commands}

        # One episode per call, answered with the terminal payload it ended on.
        self.perform_task = pimm.calls.ControlSystemHandler[Rollout, dict[str, Any]](self)
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
        return self._call.request.task

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
        assert self._inference is not None, 'only a live episode has meta'
        for k, v in flatten_dict(self._inference.meta).items():
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
            yield pimm.Yield() if self._embodiment.simulated else pimm.Sleep(POLL_PERIOD_SEC)
        # An episode must not open on a rig that never got ready, and its asker must hear that rather than wait
        if not ready.done():
            raise RuntimeError('The world stopped before every device was ready')
        ready.result()

    def _pace(self, clock: pimm.Clock) -> pimm.Command:
        """Sim: yield, so the simulator's control-period sleep is the sole time-master and the policy reads
        each observation instantly. Real: sleep to the next waypoint, capped at the poll period, so a
        waypoint is emitted at its own time and a round rarely finds more than one due."""
        if self._embodiment.simulated:
            return pimm.Yield()
        due = min((sched[0][0] for sched in self._schedules.values() if sched), default=None)
        if due is None:
            return pimm.Sleep(POLL_PERIOD_SEC)
        return pimm.Sleep(min(POLL_PERIOD_SEC, max(due - clock.now_ns(), 1) / 1e9))

    def _retire_inference(self) -> None:
        """Let go of this episode's inference, keeping it for ``_reap_inference``: the recording stops and the
        rig goes back without waiting for a model that hangs."""
        if self._inference is not None:
            self._retired, self._inference = self._inference, None

    def _reap_inference(self) -> None:
        if self._retired is not None:
            self._retired.close()
            self._retired = None

    def _finalize_recording(
        self, clock: pimm.Clock, payload: dict[str, Any] | None = None
    ) -> Generator[pimm.Command, None, None]:
        """Commit the live episode: cancel the in-flight chunk, stop the recorder — stamping the
        episode's full static meta (plus any terminal payload) — then close its span."""
        self._set_deadline(None)
        # Stamped before the inference is retired: the meta overlays what its session reports.
        self.ds_command.emit(DsWriterCommand.STOP({**self._build_episode_meta(), **(payload or {})}))
        for schedule in self._schedules.values():  # devices hold their last commanded position
            schedule.clear()
        self._retire_inference()
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
        self, clock: pimm.Clock, should_stop: pimm.SignalReceiver, call: pimm.calls.Call[Rollout, dict[str, Any]]
    ) -> Generator[pimm.Command, None, None]:
        """Open a fresh episode: take the rollout, ready the rig and the scene, open the recording and the
        first inference."""
        # Before anything that can raise, so an episode that fails to open still answers whoever asked for
        # it, and still closes the session it was handed.
        self._call = call
        self._inference = _EpisodeInference(call.request, self._charges_wall_time, clock)
        # The episode span opens first, so the prepare and the rollout's other phase spans parent to it.
        self._telemetry.begin(self._task.meta)
        with telemetry.span(telemetry_keys.SPAN_RESET):
            # An empty ask answers at once, so the episode would open on a rig that no device moved.
            if self.prepare and not self._task.prepare_args:
                rig = self._embodiment.descriptor or 'this rig'
                raise ValueError(
                    f'The trial readies nothing on {rig}, which readies {sorted(self.prepare)}; '
                    'name what each one gets in prepare_args'
                )
            yield from self._ready(should_stop, self._task.prepare_args)
        budget = self._task.timeout_sec
        self._set_deadline(clock.now_ns() + round(budget * 1e9) if budget is not None else None)
        self._telemetry.start_rollout(clock.now())
        self.ds_command.emit(DsWriterCommand.START(call.request.output_path))
        # The fresh data is here, later round would read a frame the recording did not open on.
        self._infer(self._inference, clock, should_stop)

    def _end_episode(
        self, clock: pimm.Clock, should_stop: pimm.SignalReceiver, payload: dict[str, Any]
    ) -> Generator[pimm.Command, None, None]:
        """Close the live episode: finalize the recording, put the rig back, close the session, and hand the
        terminal back to whoever asked for the episode.

        The session is closed after the rig has moved, so a model still inside its function costs the
        recording and the move nothing.
        """
        yield from self._finalize_recording(clock, payload)
        # A powered arm holds the policy's last setpoint until the next trial, so each device the trial placed
        # goes back where it put it — the trial's own args, not a fresh draw. The scene is a person's to set
        # up, and is not asked again. The terminal waits on the move: a scene the next trial draws rebuilds
        # the model an unfinished one is still travelling under, which nothing but its timeout would end.
        yield from self._ready(should_stop, {k: v for k, v in self._task.prepare_args.items() if k != keys.SCENE})
        # The answer waits until the model is out of this episode's function. An in-process policy is one
        # model across every episode, so the session that the next ask opens must not overtake it. The move
        # back above gives the function that time.
        self._reap_inference()
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
        if (trajectory := inference(obs)) is not None:
            self._reschedule(trajectory, clock)
        inference.wait(should_stop)

    @staticmethod
    def _assert_anchored(trajectory: list[dict[str, Any]], now: float) -> None:
        """Reject a chunk whose timestamps are not times on the harness clock."""
        skew = max((abs(action[keys.ACTION_TIMESTAMP] - now) for action in trajectory), default=0.0)
        if skew > MAX_ACTION_SKEW_SEC:
            raise ValueError(
                f'Action scheduled {skew:.0f}s from now, over the {MAX_ACTION_SKEW_SEC:.0f}s bound: the '
                f'rig-side stack is not anchoring chunks to the harness clock'
            )

    def _reschedule(self, trajectory: list[dict[str, Any]], clock: pimm.Clock) -> None:
        """Replace the schedule being played with the session's trajectory. Every channel it names gets that
        channel's waypoints; one it omits is cleared and holds. The timestamps are already absolute, stamped
        by the scheduling layer against the harness clock.
        """
        if self._deadline_ns is not None and clock.now_ns() >= self._deadline_ns:
            # The world reached the deadline while the function was in flight, so its chunk is dropped rather
            # than placed past the point the trial advertises it stops at; ``_run`` finishes the trial next round.
            return
        self._assert_anchored(trajectory, clock.now())
        self._telemetry.step()
        # Layers time actions in float seconds; the schedules and every pimm channel are in ns.
        for name, schedule in self._schedules.items():
            schedule.clear()
            schedule.extend((int(a[keys.ACTION_TIMESTAMP] * 1e9), a[name]) for a in trajectory if name in a)

    def _play(self, clock: pimm.Clock) -> None:
        """Emit each channel's command due this round, and nothing on a channel with none.

        A channel with several waypoints due emits the last: exact for an absolute setpoint, lossy for a
        relative one. Pacing keeps one due per round wherever a round is shorter than the waypoint spacing.
        """
        now_ns = clock.now_ns()
        for name, schedule in self._schedules.items():
            value = None
            while schedule and schedule[0][0] <= now_ns:
                value = schedule.popleft()[1]
            if value is not None:
                self.commands[name].emit(value)

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
            # An ask this loop never reached still carries a live session, and the Harness closes what it is
            # handed. The world answers what is left queued, so these calls are answered here as well.
            for call in self.perform_task.incoming():
                call.request.close()
                call.set_exception(pimm.calls.HandlerStopped())

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
                    call.request.close()  # the session came with the ask, and nothing here will run it
                    call.set_exception(RuntimeError('An episode is already running'))
                yield from self._advance_episode(self._inference, done, clock, should_stop)
            elif call is not None:
                yield from self._begin_episode(clock, should_stop, call)
            elif manual is not None:
                self._emit(manual)
            self._play(clock)
            yield self._pace(clock)

        if self._inference is not None:
            yield from self._finalize_recording(clock)
