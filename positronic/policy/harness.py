import concurrent.futures
import logging
import time
from collections import deque
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from opentelemetry.trace import Span

import pimm
from positronic import keys, telemetry, telemetry_keys
from positronic.dataset.ds_writer_agent import DsWriterCommand
from positronic.dataset.serializers import expand_suffixed
from positronic.drivers import roboarm
from positronic.drivers.roboarm.ik import assert_default_frame
from positronic.eval import Embodiment, Task
from positronic.policy.base import Policy
from positronic.utils import flatten_dict, frozen_view

# How far from now an action may be scheduled: past any real chunk, short of the decades a rig-side stack is
# off by when it leaves timestamps chunk-relative or anchors them twice.
MAX_ACTION_SKEW_SEC = 60.0

# How long a real-time round may last when no waypoint is due sooner. It bounds how late a directive is
# noticed, and with it the granularity every command timestamp is quantized to.
POLL_PERIOD_SEC = 0.01

# How long a submitted call may take and still resolve within its round: a wrapper that skips inference
# answers in microseconds, a real model call runs far past this and is throttled across rounds.
SKIP_REPLY_SEC = 0.001


class DirectiveType(Enum):
    RUN = 'run'
    FINISH = 'finish'
    ABORT = 'abort'


@dataclass
class Directive:
    """Directive from the orchestrator to the harness."""

    type: DirectiveType
    payload: Any | None = None

    @classmethod
    def RUN(cls, **kwargs) -> 'Directive':
        """Begin running the policy with the given context."""
        return cls(DirectiveType.RUN, kwargs)

    @classmethod
    def FINISH(cls, **kwargs) -> 'Directive':
        """Finalize the recording with optional eval data, then home devices."""
        return cls(DirectiveType.FINISH, kwargs)

    @classmethod
    def ABORT(cls) -> 'Directive':
        """Discard the live recording and home the devices."""
        return cls(DirectiveType.ABORT)


def _report_abandoned(future: Future[list[dict[str, Any]] | None]) -> None:
    """Report the failure of a call nobody is waiting for any more."""
    # rules-allow: swallowed-error — the call outlived the episode that asked for it, so there is no
    # caller left to raise to; the log is the only place its failure can go.
    if not future.cancelled() and (exc := future.exception()) is not None:
        logging.error(f'Inference failed after the episode that asked for it ended: {exc}')


def _owned(obs: dict[str, Any]) -> dict[str, Any]:
    """The observation with its arrays copied, so nothing rewrites what the worker is still reading.

    A producer may reuse one buffer for every sample it emits — a camera renders into the array behind
    the adapter it re-emits each frame — and it keeps stepping while a call charged in wall time runs.
    Copying at dispatch pays once per call rather than per round.
    """
    return {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in obs.items()}


class _InferenceWorker:
    """One episode's policy session, called one at a time on a thread of its own so the harness keeps
    playing while the model runs. Ending an episode ``abandon``s the call in flight rather than waiting for
    it, so the next episode never queues behind a model that hangs.

    ``charge_wall`` is what a call costs the trial: the wall time it took, or nothing — the loop is held
    for the call, which holds a virtual clock still.
    """

    def __init__(self, policy: Policy, context: dict[str, Any], charge_wall: bool) -> None:
        self._charge_wall = charge_wall
        self._session = policy.new_session(context, self.effect_time)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='harness-session')
        self._call: Future[list[dict[str, Any]] | None] | None = None
        self._t0_ns = 0  # world clock at the in-flight call's submit
        self._wall_t0 = 0.0  # ``time.monotonic()`` at that submit

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
        wall = time.monotonic() - self._wall_t0 if self._charge_wall else 0.0
        return self._t0_ns / 1e9 + wall

    def submit(self, obs: dict[str, Any], clock: pimm.Clock) -> None:
        """Start a call on ``obs``. The moment's wait lets a wrapper that skips inference resolve in the round
        it was asked."""
        self._t0_ns, self._wall_t0 = clock.now_ns(), time.monotonic()
        self._call = self._executor.submit(self._session, frozen_view(_owned(obs)))
        concurrent.futures.wait([self._call], timeout=SKIP_REPLY_SEC)

    def throttle(self, clock: pimm.Clock) -> None:
        """Slow the loop for the call in flight as the trial's mode requires: until the call returns when the
        world is held for it, else only while the world is ahead of the call's own wall clock."""
        assert self._call is not None
        # Wall time cannot be held still, so the world runs no further ahead of the call's start than it has.
        timeout = max(clock.now() - self.effect_time(), 0.0) if self._charge_wall else None
        concurrent.futures.wait([self._call], timeout=timeout)

    def result(self) -> list[dict[str, Any]] | None:
        """The returned call's trajectory — ``None`` when it had nothing to place — leaving the worker idle."""
        assert self._call is not None
        # Read on the loop thread, so a failing call still seals the episode.
        actions, self._call = self._call.result(), None
        return actions

    def abandon(self) -> None:
        """Let go of the call in flight: its answer lands nowhere and its failure only reaches the log."""
        if self._call is not None:
            self._call.add_done_callback(_report_abandoned)
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

    def begin(self, context: dict[str, Any]) -> None:
        """Open the episode span, stamped with the index and the flat trial-context keys. Called before the
        scene reset, so the reset is timed inside the rollout it belongs to."""
        self._index += 1
        self._steps = 0
        self._virtual_start = None
        attrs: dict[str, Any] = {telemetry_keys.ATTR_EPISODE_INDEX: self._index}
        attrs.update({k: v for k, v in context.items() if isinstance(v, (bool, int, float, str))})
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

    def abort(self) -> None:
        """Close an aborted rollout, marked ``episode.aborted`` so the reduce drops it."""
        if self._span is None:
            return
        telemetry.set_attrs(self._span, **{telemetry_keys.ATTR_EPISODE_ABORTED: True})
        self._end_span()

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

    The wrapper owns the plan, the harness plays it, one command per channel per round. The session call
    runs on a worker so playing continues while the model does. A call costs the trial either the wall time
    it took or nothing — the world held still for it — and the ``now`` handed to ``new_session`` reads the
    instant the call's output takes effect, so wrappers stamp for it without knowing the mode.

    A ``trials`` plan makes the harness self-driving: it starts the next trial whenever idle and returns
    once the plan is exhausted. A task's ``timeout`` bounds every trial either way, and a truthy privileged
    ``done`` ends one early — ``eval.terminated`` records which. A task-less session ends only on directives.
    """

    def __init__(
        self,
        policy: Policy,
        embodiment: Embodiment,
        *,
        task: Task | None = None,
        trials: Iterable[dict[str, Any]] | None = None,
        static_meta: dict[str, Any] | None = None,
    ):
        assert trials is None or task is not None, 'A trial plan needs a task: its timeout bounds each trial'
        self._embodiment = embodiment
        self._task = task
        # Each entry is a RUN context; when None, directives are the only lifecycle source.
        self._trials = iter(trials) if trials is not None else None
        self.policy: Policy = policy
        self.context: dict[str, Any] = {}
        self._static_meta = static_meta or {}
        # This episode's session and the thread it runs on. ``None`` while no episode is live: between RUN
        # and FINISH/ABORT, stepping and recording happen together.
        self._worker: _InferenceWorker | None = None
        # A worker let go of mid-call, kept until the join that makes closing its session safe.
        self._retiring: _InferenceWorker | None = None
        # ``task.timeout``, set per episode; a task-less session has no deadline and ends on directives.
        self._deadline: float | None = None
        # False until this episode's first observation lands; until then the deadline stands where the reset
        # put it, which bounds an episode that never gets one.
        self._rollout_started = False
        # Wall-clock telemetry for the live rollout, opened under ``--timing`` and inert otherwise.
        self._telemetry = _EpisodeTelemetry()
        # Channels that have not delivered since this episode's reset. A receiver latches its last value, so
        # emptying this set is what keeps the first inference off the previous episode's final frame.
        self._awaiting_obs: set[str] = set()

        self.observations = pimm.ReceiverDict(self)
        self.commands = pimm.EmitterDict(self)
        for name in embodiment.observations:
            self.observations[name]  # touch to allocate the port
        for name in embodiment.commands:
            self.commands[name]
        # Each channel's waypoints not yet played, stamped with absolute clock ns and ascending.
        self._schedules: dict[str, deque[tuple[int, Any]]] = {name: deque() for name in embodiment.commands}

        self.directive = pimm.ControlSystemReceiver[Directive](self, default=None, maxsize=3)
        self.manual_command = pimm.ControlSystemReceiver(self, default=None)
        self.ds_command = pimm.ControlSystemEmitter[DsWriterCommand](self)
        self.robot_meta_in = pimm.ControlSystemReceiver(self, default={})
        # Privileged stop-signal: a truthy value within the trial's budget ends it.
        self.done = pimm.ControlSystemReceiver[dict](self, default={})

    def _statics(self) -> dict[str, Any]:
        """What is known about the rig before the episode runs, live values winning."""
        return self._embodiment.static_meta | self._static_meta | self.robot_meta_in.value

    def _build_episode_meta(self, context: dict[str, Any]) -> dict[str, Any]:
        meta = self._statics()
        if self._task is not None:
            # TODO: also stamp the eval's catalog name and its resolved config — both need configuronic
            # introspection that does not exist yet.
            meta['eval.universe'] = 'sim' if self._embodiment.simulated else 'real'
            meta['eval.embodiment'] = self._embodiment.descriptor
            meta['eval.timeout'] = self._task.timeout
        # ``policy.meta`` is the static baseline; the session overlays per-episode specifics (e.g. the
        # sampled sub-policy) and wins on conflict.
        session_meta = self.policy.meta | (self._worker.meta if self._worker else {})
        for k, v in flatten_dict(session_meta).items():
            meta[f'{keys.POLICY_META}.{k}'] = v
        meta.update(context)
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
        stop = DsWriterCommand.STOP({**self._build_episode_meta(self.context), **(payload or {})})
        self._cancel_session()
        self.ds_command.emit(stop)
        virtual_now = clock.now()  # before the round below, whose sim-clock advance belongs to no rollout
        # Give the recorder a round to commit the STOP before the next START (they share ``ds_command``, where
        # last-value-wins would drop one) and before the home command, so homing stays out of the recording.
        yield self._pace(clock)
        # After that round, so the recorder's STOP-time record.io span still parents to the episode. Skew: a
        # producer stepping in that shared round charges ≤ one control period to the closing episode.
        self._telemetry.end(virtual_now)

    def _begin_episode(self, context: dict[str, Any], clock: pimm.Clock) -> None:
        """Open a fresh episode: reset the scene, fix the task context and session, and open the recording.

        A resettable task's ``reset`` only arms the producer; the first observation lands a later round. The
        recorder drains its channels the turn it opens, so the pre-reset frame and the inter-episode home
        command drop out. The deadline is armed here and moved to that first observation once it lands.
        """
        # Before the span opens, so the wait for a call the last episode abandoned is inter-episode wall
        # rather than overhead the timing reducer attributes to this one.
        self._reap_worker()
        self.context = context
        latency = self.context.get(keys.INFERENCE_LATENCY, False)
        if not isinstance(latency, bool):
            raise ValueError(f'{keys.INFERENCE_LATENCY} is a flag, got {latency!r}')
        # A real rig pays wall time whatever the trial asks: the knob is sim-only, and the eval CLI writes it
        # into every trial.
        charge_wall = latency or not self._embodiment.simulated
        self._awaiting_obs = set(self._embodiment.observations)
        self._rollout_started = False
        # Before the reset, so the reset and the rollout's other phase spans parent to the episode span.
        self._telemetry.begin(context)
        # Reset before opening the session: a resettable task only learns its instruction on reset (a remote
        # env reports it then), so the session context — and the sampling it drives — must read it here.
        if self._task is not None and self._task.reset is not None:
            with telemetry.span(telemetry_keys.SPAN_RESET):
                self._task.reset(self.context)
        if self._task is not None:
            self.context = {**self.context, keys.TASK: self._task.instruction}
        self._worker = _InferenceWorker(self.policy, self.context, charge_wall)
        self._deadline = clock.now() + self._task.timeout if self._task is not None else None
        self.ds_command.emit(DsWriterCommand.START())

    def _end_episode(
        self, clock: pimm.Clock, payload: dict[str, Any] | None = None, *, abort: bool = False
    ) -> Generator[pimm.Command, None, None]:
        """Close the live episode: finalize (or abort) the recording, retire the session, home devices.

        The worker is retired rather than joined here, so a ``RemoteSession``'s websocket outlives the call
        still using it.
        """
        if self._worker is not None:
            if abort:
                self._cancel_session()  # abort has no finalize to do it — stop the episode before the home
                self.ds_command.emit(DsWriterCommand.ABORT())
                yield self._pace(clock)  # the settling round a finalize also takes, before the home command
                self._telemetry.abort()
            else:
                yield from self._finalize_recording(clock, payload)
        self._home()

    def _handle_directive(self, directive: Directive, clock: pimm.Clock) -> Generator[pimm.Command, None, None]:
        """Dispatch a directive to the episode lifecycle."""
        match directive.type:
            case DirectiveType.RUN:
                if self._worker is None:  # a RUN mid-trial is ignored — the operator finishes before starting anew
                    self._begin_episode(directive.payload or {}, clock)
            case DirectiveType.FINISH:
                if self._worker is not None:  # a FINISH while idle is ignored — nothing to finalize
                    yield from self._end_episode(clock, directive.payload)
            case DirectiveType.ABORT:
                yield from self._end_episode(clock, abort=True)
            case _:
                raise ValueError(f'Unknown directive type: {directive.type}')

    @staticmethod
    def _is_faulted(value: Any) -> bool:
        """Whether a raw observation is an arm reporting a fault. Every other not-ready sample is simply absent."""
        return isinstance(value, roboarm.State) and value.status is roboarm.RobotStatus.ERROR

    def _build_obs(self, clock: pimm.Clock) -> dict[str, Any] | None:
        """Read every observation channel and assemble the policy input dict.

        Raises ``NoValueException`` if any channel has no value yet. Returns ``None`` while a sample is not
        ready (``robot_state`` during a ``RESETTING`` arm) or a channel still holds a pre-reset value, rather
        than feed a partial or stale obs. A faulted arm is the exception: the plan being played was made for
        an arm that is now somewhere else, so its observation reaches the stack carrying ``keys.ROBOT_FAULT``
        and without the arm's own entries.
        """
        # Against the live model, not the one known at episode start: a remote env publishes its ``robot_meta``
        # a turn after the reset that produced it, so at episode start there is no model to check.
        assert_default_frame(self._statics())
        inputs: dict[str, Any] = {}
        faulted = False
        not_ready = False
        for name, obs in self._embodiment.observations.items():
            message = self.observations[name].read()
            if message is None:
                raise pimm.NoValueException
            if message.updated:
                self._awaiting_obs.discard(name)
            value = message.data
            if obs.serializer is not None:
                value = obs.serializer(value)
                if value is None:  # no sample to give: a resetting or faulted arm alike
                    # HACK(#619): a serializer answers `None` for a resetting arm and a faulted one alike, so
                    # the fault is recovered from the raw sample and stapled on as `keys.ROBOT_FAULT` — a name
                    # already claiming to be part of `robot_state`. Emit it from the serializer and this
                    # branch, the raw-type check and the flag all go, and the fault reaches the recording too.
                    faulted = faulted or self._is_faulted(message.data)
                    not_ready = True
                    continue
            inputs.update({full: v for full, v in expand_suffixed(name, value) if v is not None})
        # Every channel is read before this decision, so a bimanual rig cannot hide one arm's fault behind
        # another arm's not-ready sample: whichever channel comes first, the fault still reaches the stack.
        if (not_ready and not faulted) or self._awaiting_obs:
            return None
        # The trial's context goes under what the harness read this round, never over it: a context carries
        # whatever keys the RUN directive puts in it, and a ``robot_state.fault`` among them must not tell
        # ``StopOnFault`` that a faulted arm is sound.
        inputs = {**self.context, **inputs}
        inputs[keys.ROBOT_FAULT] = faulted
        inputs[keys.WALL_TIME_NS] = time.time_ns()
        inputs[keys.OBS_TIME_NS] = clock.now_ns()
        inputs['descriptor'] = self._embodiment.descriptor
        if not self._rollout_started:
            # The rollout begins at its first observation, not when the reset returned: the turns spent
            # delivering the scene are neither the trial's budget nor its duration.
            self._rollout_started = True
            self._telemetry.start_rollout(clock.now())
            if self._task is not None:
                self._deadline = clock.now() + self._task.timeout
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
                worker.submit(obs, clock)
                self._throttle_and_reschedule(worker, clock)

    def _throttle_and_reschedule(self, worker: _InferenceWorker, clock: pimm.Clock) -> None:
        worker.throttle(clock)
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

    def _trial_terminal(self, clock: pimm.Clock) -> dict[str, Any] | None:
        """The terminal static payload if a self-driven trial has ended this round, else ``None``.

        The deadline is hard: a truthy ``done`` within budget records ``eval.terminated`` True plus its
        payload, the budget passing records False, and a terminal past the deadline is a timeout rather than
        a late success. Only a freshly delivered ``done`` counts, or the receiver's latched value would
        re-fire a prior trial's terminal.
        """
        deadline = self._deadline
        if deadline is None:  # a task-less session has no budget and ends on directives alone
            return None
        done_msg = self.done.read()
        assert done_msg is not None  # the receiver carries a default, so ``read`` always yields a message
        if done_msg.updated and done_msg.data and done_msg.ts <= deadline * 1e9:
            return {**done_msg.data, keys.EVAL_TERMINATED: True}
        if clock.now() >= deadline:
            return {keys.EVAL_TERMINATED: False}
        return None

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        # Home the embodiment before the first episode; each ``_end_episode`` re-homes for the next one, so
        # every episode begins from the home pose (a real arm gets the inter-episode gap to reach it).
        self._home()

        try:
            yield from self._run(should_stop, clock)
        except BaseException:
            # Seal the open span before the exception reaches ``bind``'s exit flush: an unended span never
            # exports, orphaning its finished children and charging the episode's wall to between_episodes.
            self._telemetry.seal(clock.now())
            raise
        finally:
            # A call still in flight runs to completion and its result is dropped. The join is not deferred:
            # no later episode will do it, and the policy the call holds outlives this harness.
            self._retire_worker()
            self._reap_worker()

    def _run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        while not should_stop.value:
            # One action per round, mutually exclusive: handle a directive, start the next trial (or exit
            # when the plan is done), finish one that is out of budget or done, or step the policy. Starting
            # takes its own round, so inference waits for the producer's post-reset observation.
            directive_msg = self.directive.read()
            # Read every round so the flag clears mid-episode; a press during a trial is consumed, not replayed.
            manual_msg = self.manual_command.read()
            # Both receivers carry a default, so ``read`` always yields a message.
            assert directive_msg is not None and manual_msg is not None
            if directive_msg.updated:
                yield from self._handle_directive(directive_msg.data, clock)
            elif self._worker is None:
                if manual_msg.updated and manual_msg.data is not None:
                    self._emit(manual_msg.data)
                elif self._trials is not None:
                    trial = next(self._trials, None)
                    if trial is None:  # plan exhausted — let the recorder commit the final episode, then exit
                        yield pimm.Sleep(0.5)
                        break
                    self._begin_episode(trial, clock)
            elif (terminal := self._trial_terminal(clock)) is not None:
                yield from self._end_episode(clock, terminal)
            else:
                try:
                    self._step(self._worker, clock)
                except pimm.NoValueException:
                    pass
            self._play(clock)
            yield self._pace(clock)

        if self._worker is not None:
            yield from self._finalize_recording(clock)
