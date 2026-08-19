import concurrent.futures
import contextvars
import logging
import time
from collections import deque
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
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
from positronic.utils import flatten_dict, frozen_view

# How far from now an action may be scheduled: past any real chunk, short of the decades a rig-side stack is
# off by when it leaves timestamps chunk-relative or anchors them twice.
MAX_ACTION_SKEW_SEC = 60.0

# How long a real-time round may last when no waypoint is due sooner. It bounds how late a call is noticed,
# and with it the granularity every command timestamp is quantized to.
POLL_PERIOD_SEC = 0.01

# How long a submitted call may take and still resolve within its round: a wrapper that skips inference
# answers in microseconds, a real model call runs far past this and is throttled across rounds.
SKIP_REPLY_SEC = 0.001


class _InferenceWorker:
    """One episode's policy session, called one at a time on a thread of its own so the harness keeps
    playing while the model runs.

    ``charges_wall_time`` says whether a call costs the trial the wall time it really took or nothing —
    the loop is held for the call, which holds a virtual clock still.
    """

    def __init__(self, policy: Policy, context: dict[str, Any], charges_wall_time: bool, clock: pimm.Clock) -> None:
        self._charges_wall_time = charges_wall_time
        self._clock = clock
        # World clock and ``time.monotonic()`` at the in-flight call's submit, anchored at the episode's
        # start so ``effect_time`` reads a trial instant from the moment the session exists.
        self._t0_ns, self._wall_t0 = clock.now_ns(), time.monotonic()
        self._session = policy.new_session(context, self.effect_time)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='harness-session')
        self._call: Future[list[dict[str, Any]] | None] | None = None

    @property
    def meta(self) -> dict[str, Any]:
        return self._session.meta

    @property
    def idle(self) -> bool:
        return self._call is None

    @property
    def done(self) -> bool:
        """Whether the call in flight has returned."""
        return self._call is not None and self._call.done()

    def effect_time(self) -> float:
        """The trial instant the in-flight call's output takes effect: its submit, plus its wall duration so
        far when the trial pays wall time."""
        wall = time.monotonic() - self._wall_t0 if self._charges_wall_time else 0.0
        return self._t0_ns / 1e9 + wall

    @staticmethod
    def _owned(obs: dict[str, Any]) -> dict[str, Any]:
        """The observation with its arrays copied, so nothing rewrites what the worker is still reading.

        A producer may reuse one buffer for every sample it emits — a camera renders into the array behind
        the adapter it re-emits each frame — and it keeps stepping while a call charged in wall time runs.
        Copying at dispatch pays once per call rather than per round.
        """
        return {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in obs.items()}

    def submit(self, obs: dict[str, Any]) -> None:
        """Start a call on ``obs``. The moment's wait lets a wrapper that skips inference resolve in the round
        it was asked."""
        self._t0_ns, self._wall_t0 = self._clock.now_ns(), time.monotonic()
        # The call runs under a copy of the loop's context, so the telemetry it records anchors to the episode
        # that asked for it even when it outlives that episode's close.
        context = contextvars.copy_context()
        self._call = self._executor.submit(context.run, self._session, frozen_view(self._owned(obs)))
        concurrent.futures.wait([self._call], timeout=SKIP_REPLY_SEC)

    def throttle(self) -> None:
        """Slow the loop for the call in flight as the trial's mode requires: until the call returns when the
        world is held for it, else only while the world is ahead of the call's own wall clock."""
        assert self._call is not None
        # Wall time cannot be held still, so the world runs no further ahead of the call's start than it has.
        timeout = max(self._clock.now() - self.effect_time(), 0.0) if self._charges_wall_time else None
        concurrent.futures.wait([self._call], timeout=timeout)

    def result(self) -> list[dict[str, Any]] | None:
        """The returned call's trajectory — ``None`` when it had nothing to place — leaving the worker idle."""
        assert self._call is not None
        # Read on the loop thread, so a failing call still seals the episode.
        actions, self._call = self._call.result(), None
        return actions

    @staticmethod
    def _report_abandoned(future: Future[list[dict[str, Any]] | None]) -> None:
        """Report the failure of a call nobody is waiting for any more."""
        # rules-allow: swallowed-error — the call outlived the episode that asked for it, so there is no
        # caller left to raise to; the log is the only place its failure can go.
        if not future.cancelled() and (exc := future.exception()) is not None:
            logging.error(f'Inference failed after the episode that asked for it ended: {exc}')

    def abandon(self) -> None:
        """Let go of the call in flight: its answer lands nowhere and its failure only reaches the log."""
        if self._call is not None:
            self._call.add_done_callback(self._report_abandoned)
            self._call = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def join(self) -> None:
        """Wait out an abandoned call, then close the session it was inside.

        ``shutdown(cancel_futures=True)`` cancels only what is still queued, so until this returns the call
        still holds the session's resources: a ``RemoteSession``'s websocket, or the in-process model every
        session shares.
        """
        self._executor.shutdown(wait=True)
        self._session.close()


class _EpisodeTelemetry:
    """The live rollout's episode span, with the index, step count and virtual start it is stamped with.
    Inert while telemetry is unbound, so the harness calls it unconditionally. The span stays anchored while
    open, so the rollout's phase spans parent to it rather than to the pass."""

    def __init__(self) -> None:
        self._span: Span | None = None
        self._index = -1
        self._steps = 0
        # ``None`` until the rollout starts, so the reset is excluded and a failed reset leaves it unstamped.
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
        """Anchor the rollout's virtual duration at the instant its first observation landed."""
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
        # A rollout whose first observation never landed — a reset that raised, or a task already done before
        # it arrived — has zero virtual duration.
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

    The wrapper owns the trajectory, the harness plays it, one command per channel per round. The session call
    runs on a worker so playing continues while the model does. A call costs the trial either the wall time
    it took or nothing — the world held still for it — and the ``now`` handed to ``new_session`` reads the
    instant the call's output takes effect, so wrappers stamp for it without knowing the mode.

    An episode runs one ``Task``, asked for by a ``perform_task`` call and answered with the terminal
    payload it ended on. The task's ``timeout_sec`` bounds it and a truthy ``done`` within budget ends it
    early — ``eval.terminated`` records which; a task without one ends on ``done`` alone. Which tasks run,
    and in what order, belongs to whoever makes the calls.
    """

    def __init__(
        self,
        policy: Policy,
        embodiment: Embodiment,
        *,
        reset: Callable[[dict[str, Any]], None] | None = None,
        static_meta: dict[str, Any] | None = None,
    ):
        self._embodiment = embodiment
        # Re-randomizes the scene from the task's ``params``; ``None`` where reset is physical/human.
        self._reset = reset
        self.policy: Policy = policy
        self._static_meta = static_meta or {}
        # This episode's session and the thread it runs on. ``None`` while no episode is live: while one is,
        # stepping and recording happen together.
        self._worker: _InferenceWorker | None = None
        # The call this episode answers when it ends.
        self._call: pimm.calls.Call[Task, dict[str, Any]] | None = None
        # A worker let go of mid-call, kept until the join that makes closing its session safe.
        self._retiring: _InferenceWorker | None = None
        # ``task.timeout_sec``, armed per episode; a task without one has no deadline and ends on ``done`` alone.
        self._deadline: float | None = None
        # False until this episode's first observation lands; until then the deadline stands where the reset
        # put it, which bounds an episode that never gets one.
        self._rollout_started = False
        # Wall-clock telemetry for the live rollout, opened under ``--timing`` and inert otherwise.
        self._telemetry = _EpisodeTelemetry()
        # Channels that have not delivered since this episode began; the first inference waits until every one
        # has. A receiver latches its last value, so a producer silent between episodes would otherwise feed
        # the previous episode's final frame. Delivery is judged by ``updated``, not ``ts``: some producers
        # stamp ``ts`` on their own clock.
        self._awaiting_obs: set[str] = set()

        self.observations = pimm.ReceiverDict(self)
        self.commands = pimm.EmitterDict(self)
        for name in embodiment.observations:
            self.observations[name]  # touch to allocate the port
        for name in embodiment.commands:
            self.commands[name]
        # Each channel's waypoints not yet played, stamped with absolute clock ns and ascending.
        self._schedules: dict[str, deque[tuple[int, Any]]] = {name: deque() for name in embodiment.commands}

        # One episode per call, answered with the terminal payload it ended on.
        self.perform_task = pimm.calls.ControlSystemHandler[Task, dict[str, Any]](self)
        self.manual_command = pimm.ControlSystemReceiver(self)
        self.ds_command = pimm.ControlSystemEmitter[DsWriterCommand](self)
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
        # ``policy.meta`` is the static baseline; the session overlays per-episode specifics (e.g. the
        # sampled sub-policy) and wins on conflict.
        session_meta = self.policy.meta | (self._worker.meta if self._worker else {})
        for k, v in flatten_dict(session_meta).items():
            meta[f'{keys.POLICY_META}.{k}'] = v
        meta.update(self._task.params)
        meta[keys.TASK] = self._task.instruction
        return meta

    def _emit(self, action: dict[str, Any]) -> None:
        for name, value in action.items():
            self.commands[name].emit(value)

    def _home(self) -> None:
        self._emit(self._embodiment.home)

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

    def _cancel_session(self) -> None:
        """Drop everything the episode has going: the schedule being played, and the call on the worker.

        The call is let go of rather than waited for, so a model that hangs cannot hold up the recording's
        stop or the home. Devices hold their last commanded position; nothing downstream is buffered.
        """
        for schedule in self._schedules.values():
            schedule.clear()
        self._retire_worker()

    def _retire_worker(self) -> None:
        """Let go of this episode's worker and the call it is running, keeping it for ``_reap_worker``:
        ending an episode must not wait for a model that hangs."""
        if self._worker is not None:
            self._worker.abandon()
            self._retiring, self._worker = self._worker, None

    def _reap_worker(self) -> None:
        """Join the retired worker and close the session its abandoned call was inside."""
        if self._retiring is not None:
            self._retiring.join()
            self._retiring = None

    def _finalize_recording(
        self, clock: pimm.Clock, payload: dict[str, Any] | None = None
    ) -> Generator[pimm.Command, None, None]:
        """Commit the live episode: cancel the in-flight chunk, stop the recorder — stamping the
        episode's full static meta (plus any terminal payload) — then close its span."""
        # Stamped before the worker is retired: the meta overlays what its session reports.
        stop = DsWriterCommand.STOP({**self._build_episode_meta(), **(payload or {})})
        self._cancel_session()
        self.ds_command.emit(stop)
        virtual_now = clock.now()  # before the round below, whose sim-clock advance belongs to no rollout
        # Give the recorder a round to commit the STOP before the next START (they share ``ds_command``, where
        # last-value-wins would drop one) and before the home command, so homing stays out of the recording.
        yield self._pace(clock)
        # After that round, so the recorder's STOP-time record.io span still parents to the episode. Skew: a
        # producer stepping in that shared round charges ≤ one control period to the closing episode.
        self._telemetry.end(virtual_now)

    def _begin_episode(self, clock: pimm.Clock, call: pimm.calls.Call[Task, dict[str, Any]]) -> None:
        """Open a fresh episode: reset the scene, read the instruction, open the session and the recording.

        ``reset`` only arms the producer; the first observation lands a later round. The recorder drains its
        channels the turn it opens, so the pre-reset frame and the inter-episode home command drop out. The
        deadline is armed here and moved to that first observation once it lands.
        """
        # Before anything that can raise, so an episode that fails to open still answers whoever asked for it.
        self._call = call
        # Before the span opens, so the wait for a call the last episode abandoned is inter-episode wall
        # rather than overhead the timing reducer attributes to this one.
        self._reap_worker()
        self._awaiting_obs = set(self._embodiment.observations)
        self._rollout_started = False
        # Before the reset, so the reset and the rollout's other phase spans parent to the episode span.
        self._telemetry.begin(self._task.params)
        if self._reset is not None:
            with telemetry.span(telemetry_keys.SPAN_RESET):
                self._reset(self._task.params)
        # Read after the reset: an embodiment that learns its task from the scene reports it only once the
        # scene is set up.
        self._worker = _InferenceWorker(
            self.policy, {keys.TASK: self._task.instruction}, self._charges_wall_time, clock
        )
        budget = self._task.timeout_sec
        self._deadline = clock.now() + budget if budget is not None else None
        self.ds_command.emit(DsWriterCommand.START())

    def _end_episode(self, clock: pimm.Clock, payload: dict[str, Any]) -> Generator[pimm.Command, None, None]:
        """Close the live episode: finalize the recording, retire the session, home devices, hand the terminal
        back to whoever asked for the episode.

        The worker is retired rather than joined here, so a ``RemoteSession``'s websocket outlives the call
        still using it.
        """
        yield from self._finalize_recording(clock, payload)
        self._home()
        assert self._call is not None, 'an episode exists only for the call that asked for it'
        self._call.set_result(payload)
        self._call = None

    def _fail_call(self, exc: BaseException) -> None:
        """Raise to whoever asked for the live episode, in place of the terminal it will never get."""
        if self._call is not None:
            self._call.set_exception(exc)
            self._call = None

    def _advance_episode(
        self, worker: _InferenceWorker, done: pimm.Message[dict] | None, clock: pimm.Clock
    ) -> Generator[pimm.Command, None, None]:
        """One round of the live episode: end it if it is out of budget or done, else step the policy."""
        if (terminal := self._trial_terminal(done, clock)) is not None:
            yield from self._end_episode(clock, terminal)
        else:
            try:
                self._step(worker, clock)
            except pimm.NoValueException:
                pass

    def _build_obs(self, clock: pimm.Clock) -> dict[str, Any] | None:
        """Read every observation channel and assemble the policy input dict.

        Raises ``NoValueException`` if any channel has no value yet, and returns ``None`` while a channel
        still holds a pre-reset value, rather than feed a stale obs.
        """
        # Against the live model, not the one known at episode start: a remote env publishes its ``robot_meta``
        # a turn after the reset that produced it, so at episode start there is no model to check.
        assert_default_frame(self._statics())
        inputs: dict[str, Any] = {}
        for name, obs in self._embodiment.observations.items():
            message = self.observations[name].read()
            if message is None:
                raise pimm.NoValueException
            if message.updated:
                self._awaiting_obs.discard(name)
            value = message.data
            if obs.serializer is not None:
                value = obs.serializer(value)
            inputs.update({full: v for full, v in expand_suffixed(name, value) if v is not None})
        if self._awaiting_obs:
            return None
        inputs[keys.TASK] = self._task.instruction
        inputs[keys.WALL_TIME_NS] = time.time_ns()
        inputs[keys.OBS_TIME_NS] = clock.now_ns()
        inputs[keys.DESCRIPTOR] = self._embodiment.descriptor
        if not self._rollout_started:
            # The rollout begins at its first observation, not when the reset returned: the turns spent
            # delivering the scene are neither the trial's budget nor its duration.
            self._rollout_started = True
            self._telemetry.start_rollout(clock.now())
            if self._task.timeout_sec is not None:
                self._deadline = clock.now() + self._task.timeout_sec
        return inputs

    def _step(self, worker: _InferenceWorker, clock: pimm.Clock) -> None:
        """One round of inference: throttle the loop for the call in flight as the trial's mode requires and
        reschedule on what it returns; with no call in flight, submit one on a fresh observation and give it
        the rest of the round to return."""
        if not worker.idle:
            self._throttle_and_reschedule(worker, clock)
        if worker.idle:
            obs = self._build_obs(clock)
            if obs is not None:
                worker.submit(obs)
                self._throttle_and_reschedule(worker, clock)

    def _throttle_and_reschedule(self, worker: _InferenceWorker, clock: pimm.Clock) -> None:
        worker.throttle()
        if worker.done and (trajectory := worker.result()) is not None:
            self._reschedule(trajectory, clock)

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
        by the scheduling wrapper for the instant the call's output takes effect.
        """
        if self._deadline is not None and clock.now() >= self._deadline:
            # The world reached the deadline while the call was in flight, so its chunk is dropped rather than
            # placed past the point the trial advertises it stops at; ``_run`` finishes the trial next round.
            return
        self._assert_anchored(trajectory, clock.now())
        self._telemetry.step()
        # The single explicit seconds->ns seam: wrappers time actions in float seconds, the schedules and
        # every pimm channel are in ns.
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
        deadline = self._deadline
        if done is not None and done.data and (deadline is None or done.ts <= deadline * 1e9):
            return {**done.data, keys.EVAL_TERMINATED: True}
        if deadline is not None and clock.now() >= deadline:
            return {keys.EVAL_TERMINATED: False}
        return None

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        # Home the embodiment before the first episode; each ``_end_episode`` re-homes for the next one, so
        # every episode begins from the home pose (a real arm gets the inter-episode gap to reach it).
        self._home()

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
            self._retire_worker()
            self._reap_worker()

    def _run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        while not should_stop.value:
            # One action per round, mutually exclusive: start the episode a call asks for, finish one that
            # is out of budget or done, or step the policy. Starting takes its own round, so inference waits
            # for the producer's post-reset observation.
            call = next(self.perform_task.incoming(), None)
            # Read every round so the flag clears mid-episode; a press during a trial is consumed, not replayed.
            manual = pimm.value_updated(self.manual_command)
            # Read every round for the same reason: a terminal landing between episodes belongs to none of
            # them, and left on the wire it would end the next one on its first round.
            done = pimm.read_updated(self.done)
            if self._worker is not None:
                if call is not None:  # the live episode is the one that finishes; a second ask is refused
                    call.set_exception(RuntimeError('An episode is already running'))
                yield from self._advance_episode(self._worker, done, clock)
            elif call is not None:
                self._begin_episode(clock, call)
            elif manual is not None:
                self._emit(manual)
            self._play(clock)
            yield self._pace(clock)

        if self._worker is not None:
            yield from self._finalize_recording(clock)
