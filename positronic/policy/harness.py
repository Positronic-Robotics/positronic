import time
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from opentelemetry.trace import Span

import pimm
from positronic import keys, telemetry, telemetry_keys
from positronic.dataset.ds_writer_agent import DsWriterCommand
from positronic.dataset.serializers import expand_suffixed
from positronic.drivers.roboarm.ik import assert_default_frame
from positronic.eval import Embodiment, Task
from positronic.policy.base import Policy, Session
from positronic.policy.wrappers import ChunkedSchedule
from positronic.utils import flatten_dict, frozen_view


class DirectiveType(Enum):
    """Directive types for the harness."""

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


class _EpisodeTelemetry:
    """The live rollout's wall-clock telemetry: the episode span, its 0-based index, its control-step count
    and the virtual instant its rollout began. Every method is inert while telemetry is unbound (a normal eval
    binds nothing), so the harness calls them unconditionally.

    The span is anchored while it is open, so the phase spans the rollout's control systems emit (reset,
    env.step, policy.infer, record.io) parent to it rather than to the pass.
    """

    def __init__(self) -> None:
        self._span: Span | None = None
        self._index = -1
        self._steps = 0
        # The virtual instant the rollout began — ``None`` until it starts, so the reset is excluded and a
        # reset that fails leaves it unstamped.
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
        """Anchor the rollout's virtual duration at the first control cycle that has an observation.

        A simulated producer's ``reset`` only arms frame zero, which it publishes on its next turn, and that
        turn advances the virtual clock by a control period without stepping the environment. Anchoring when
        the reset returns would charge that period to the rollout while its wall sits under reset, inflating
        the real-time factor by one control period per episode. Called every cycle; only the first lands.
        """
        if self._virtual_start is None:
            self._virtual_start = virtual_now

    def step(self) -> None:
        self._steps += 1

    def end(self, virtual_now: float) -> None:
        """Close a finished rollout, stamped with its step count and its virtual duration up to
        ``virtual_now`` — captured when the rollout ended, before the flush round advances the sim clock."""
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
        """Close a rollout abandoned by a failure mid-flight (a raising ``reset`` / ``new_session`` / session
        call), stamped like a clean end and marked ``episode.partial``. Ending it is what exports it: the batch
        processor never emits an unended span, so the abandoned rollout's finished children would orphan and
        the reduce would lose their phases. Marked (not aborted) so the reduce keeps it — its finished phases
        attribute — while flagging that it did not run to completion. Inert when no span is open (telemetry
        off, or the failure fell outside a rollout)."""
        if self._span is None:
            return
        telemetry.set_attrs(self._span, **{telemetry_keys.ATTR_EPISODE_PARTIAL: True})
        self._close(virtual_now)
        telemetry.force_flush()

    def _close(self, virtual_now: float) -> None:
        # A rollout that never reached its first observation — a reset that raised, or a task already done
        # before frame zero landed — has zero virtual duration; only an anchored rollout measures from its
        # start.
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
    """Control system that manages episode lifecycle and forwards trajectories to drivers.

    The harness handles directives (RUN/FINISH/ABORT) and dataset recording. All inference
    intelligence (scheduling, error recovery, blending, absolute time stamping) lives in the
    policy/session layer — the harness just calls the session, demuxes the action dicts into
    per-channel trajectories, and emits.

    ``RUN`` may carry ``inference_latency`` (sim-only inference-cost simulation) in its context; the whole
    context is handed to the task's scene reset, which reads the per-trial keys it needs (e.g. ``eval.seed``).
    A ``trials`` plan (a sequence of RUN contexts) makes the harness self-driving: whenever it is
    idle it starts the next trial itself and exits once the plan is exhausted, so the unattended
    path needs no driver at all. A task's ``timeout`` bounds every trial it is given to — self-driven
    or operator-driven alike — so an attended episode still terminates at the deadline even if the
    operator never sends FINISH; a task-less attended session has no deadline and ends only on directives.

    A deadline-bounded trial also ends early when the privileged ``done`` signal is delivered within
    its budget: it records ``eval.terminated`` True and the delivered payload in its static data, a
    timed-out one False. A task-less session has no budget, so ``done`` does not terminate it.

    The ``Embodiment`` provides the observation serializers (which own the canonical key names),
    the command channels, and the home action; the harness reads them to assemble inputs and demux
    actions, treating every channel alike.

    The scheduling wrapper (``ChunkedSchedule``, or a swap-in like RTC) anchors the chunk's
    relative timestamps to absolute time, reading the clock the harness passes to ``new_session``.
    The policy owns its wrapper stack (declared by the server or composed in its pipeline); the
    harness runs the policy it is given.
    """

    def __init__(
        self,
        policy: Policy,
        embodiment: Embodiment,
        *,
        task: Task | None = None,
        trials: Iterable[dict[str, Any]] | None = None,
        static_meta: dict[str, Any] | None = None,
        on_episode_complete: Callable[[Session, dict[str, Any]], None] | None = None,
    ):
        assert trials is None or task is not None, 'A trial plan needs a task: its timeout bounds each trial'
        self._embodiment = embodiment
        self._task = task
        # The unattended trial plan: each entry is a RUN context. When set, the run loop starts the
        # next trial whenever it is idle and returns once the plan is exhausted; when None,
        # directives are the only lifecycle source.
        self._trials = iter(trials) if trials is not None else None
        # Called with (session, context) when an episode completes successfully (clean
        # FINISH or auto-finalize), never on abort. Used to feed completion bookkeeping
        # like a ``SampledPolicy``'s episode counter, with no sampling knowledge in the harness.
        self._on_complete = on_episode_complete or (lambda session, context: None)
        self.policy: Policy = policy
        self.context: dict[str, Any] = {}
        self._static_meta = static_meta or {}
        self._policy_session: Session | None = None
        # True between RUN and FINISH/ABORT: the trial is live — stepping and recording happen together.
        self._running = False
        # ``inference_latency`` is delivered on the RUN context (sim-only): ``True`` advances the
        # (sim) clock by the wall-clock cost of the inference call; a float is a fixed deterministic
        # delay (used by the reproducible golden). Sleep is yielded BEFORE ``ChunkedSchedule`` reads
        # ``clock.now()`` so the trajectory is anchored to inference-finish, not inference-start.
        self._inference_latency: bool | float = False
        # A trial with a task is bounded by ``task.timeout``, set per episode; a task-less attended
        # session has no deadline and is ended by directives.
        self._deadline: float | None = None
        # Wall-clock telemetry for the live rollout, opened under ``--timing`` and inert otherwise.
        self._telemetry = _EpisodeTelemetry()
        # Observation channels that have not delivered since this episode's reset. A receiver latches its
        # last value, so an empty set is what makes the first inference of an episode read the post-reset
        # scene rather than the previous episode's final frame.
        self._awaiting_obs: set[str] = set()

        self._descriptor = embodiment.descriptor
        self.observations = pimm.ReceiverDict(self)
        self.commands = pimm.EmitterDict(self)
        for name in embodiment.observations:
            self.observations[name]  # touch to allocate the port
        for name in embodiment.commands:
            self.commands[name]

        self.directive = pimm.ControlSystemReceiver[Directive](self, default=None, maxsize=3)
        self.manual_command = pimm.ControlSystemReceiver(self, default=None)
        self.ds_command = pimm.ControlSystemEmitter[DsWriterCommand](self)
        self.robot_meta_in = pimm.ControlSystemReceiver(self, default={})
        # Privileged stop-signal: a truthy value within a trial's time budget ends it,
        # recording ``eval.terminated`` True plus that dict in the episode's static data.
        self.done = pimm.ControlSystemReceiver[dict](self, default={})

    def _statics(self) -> dict[str, Any]:
        """What is known about the rig before the episode runs, live values winning."""
        return self._embodiment.static_meta | self._static_meta | self.robot_meta_in.value

    def _build_episode_meta(self, context: dict[str, Any]) -> dict[str, Any]:
        meta = self._statics()
        if self._task is not None:
            # The eval-identity block: which eval produced this episode.
            # TODO: also stamp the eval's catalog name and its resolved config — both need
            # configuronic introspection that does not exist yet.
            meta['eval.universe'] = 'sim' if self._embodiment.simulated else 'real'
            meta['eval.embodiment'] = self._embodiment.descriptor
            meta['eval.timeout'] = self._task.timeout
        # ``policy.meta`` is the static baseline (the wrapped policy aggregates model +
        # codec meta); the session overlays per-episode specifics (e.g. the sampled
        # sub-policy) and wins on conflict.
        session_meta = self.policy.meta | (self._policy_session.meta if self._policy_session else {})
        for k, v in flatten_dict(session_meta).items():
            meta[f'inference.policy.{k}'] = v
        meta.update(context)
        return meta

    def _home(self, clock):
        now = clock.now_ns()
        for name, value in self._embodiment.home.items():
            self.commands[name].emit([(now, value)])

    def _apply_manual(self, action: dict[str, Any], clock: pimm.Clock) -> None:
        now = clock.now_ns()
        for name, value in action.items():
            self.commands[name].emit([(now, value)])

    def _pace(self) -> pimm.Command:
        """Sim mode: yield so the simulator's control-period sleep is the sole time-master — the policy
        reads each observation instantly, matching the gym contract. Real mode: sleep the poll period to
        hold wall-clock rate."""
        return pimm.Yield() if self._embodiment.simulated else pimm.Sleep(0.01)

    def _bump_schedule_end(self, delta_sec: float) -> None:
        """Shift the active ``ChunkedSchedule._Session`` ``_trajectory_end`` by ``delta_sec``.

        Used by ``inference_latency``: the session anchored the chunk pre-sleep, then we slept and
        post-shifted the emitted timestamps. The scheduling wrapper's internal end-of-chunk gate
        must move forward too, or it will re-infer before the driver has actually played the (shifted)
        trajectory.
        """
        s = self._policy_session
        while s is not None:
            if isinstance(s, ChunkedSchedule._Session) and s._trajectory_end is not None:
                s._trajectory_end += delta_sec
                return
            s = getattr(s, '_inner', None)

    def _cancel_trajectories(self) -> None:
        """Drop any in-flight chunk from drivers and from the recording's tail.

        Emits ``[]`` on every command channel so each driver's
        ``TrajectoryPlayer`` clears its buffer (devices hold position) and
        ``TrajectoryOverrideSerializer`` drops its uncommitted tail. Must
        precede ``STOP_EPISODE``, which ``flush()``​es the recording's
        serializers and would otherwise commit canceled waypoints. Also
        cancels the active session's scheduling state so the next inference
        is not held back by stale trajectory_end.
        """
        self._emit_commands([])
        if self._policy_session is not None:
            self._policy_session.cancel()

    def _finalize_recording(self, payload: dict[str, Any] | None = None) -> None:
        """Commit the live episode: tally completion, cancel the in-flight chunk, stop the recorder —
        stamping the episode's full static meta (plus any terminal payload) at finalize."""
        if self._policy_session:
            self._on_complete(self._policy_session, self.context)
        self._cancel_trajectories()
        self.ds_command.emit(DsWriterCommand.STOP({**self._build_episode_meta(self.context), **(payload or {})}))

    def _begin_episode(self, context: dict[str, Any], clock: pimm.Clock) -> None:
        """Open a fresh episode: reset the scene, fix the task context and session, and open the recording.

        A resettable task's ``reset`` only arms the producer, which publishes frame-0 after the harness
        (last in the round). The recorder drains its channels the turn it opens, so the pre-reset frame and
        the inter-episode home command — lingering there from before START — drop out and its first sample
        is the post-reset scene, which the harness infers on once it lands. The trial deadline (a task's
        ``timeout``, bounding policy- and operator-driven trials alike) is armed here; a task-less attended
        session has no deadline and ends only on a directive.
        """
        self.context = context
        # ``inference_latency`` rides the RUN context (and lands in episode meta with it).
        self._inference_latency = self.context.get('inference_latency', False)
        self._awaiting_obs = set(self._embodiment.observations)
        # Open the episode span before the reset, so the phase spans (reset, env.step, policy.infer,
        # record.io) parent to it.
        self._telemetry.begin(context)
        # Reset the scene before opening the session: a resettable task only learns its instruction on reset
        # (a remote env reports it then), so the session context — and the task-grouped sampling/counting it
        # drives — must read the instruction here, once it is known.
        if self._task is not None and self._task.reset is not None:
            with telemetry.span(telemetry_keys.SPAN_RESET):
                self._task.reset(self.context)
        if self._task is not None:
            self.context = {**self.context, keys.TASK: self._task.instruction}
            # The timeout is a safety net, not the benchmark deadline: a sim that reports its own horizon (bench
            # env) owns the episode's end, so a timeout that is not strictly longer would silently truncate a valid
            # episode. Reject it loudly at the trial it would first bite, rather than mis-scoring the run.
            horizon = self._task.horizon
            if horizon is not None and self._task.timeout <= horizon:
                raise ValueError(
                    f'eval timeout {self._task.timeout}s must be strictly longer than the sim-enforced episode '
                    f'horizon {horizon}s (it is only a safety net) — raise the timeout above the benchmark horizon'
                )
        self._policy_session = self.policy.new_session(self.context, clock.now)
        self._running = True
        self._deadline = clock.now() + self._task.timeout if self._task is not None else None
        self.ds_command.emit(DsWriterCommand.START())

    def _end_episode(
        self, clock: pimm.Clock, payload: dict[str, Any] | None = None, *, abort: bool = False
    ) -> Generator[pimm.Command, None, None]:
        """Close the live episode: finalize (or abort) the recording, release the session, home devices.

        Releasing the session here (not only at shutdown) closes a ``RemoteSession``'s websocket
        promptly, so the offboard server's per-session cleanup (active-session decrement, idle watchdog)
        runs now.
        """
        if self._running:
            if abort:
                self._cancel_trajectories()  # abort has no finalize to do it — stop drivers before the home
                self.ds_command.emit(DsWriterCommand.ABORT())
            else:
                self._finalize_recording(payload)
            # The rollout's virtual duration ends here — the flush round below advances the sim clock, and that
            # advance belongs to no rollout.
            virtual_now = clock.now()
            # Let the recorder commit the STOP/ABORT before the next START (they share ``ds_command`` —
            # without a round between, last-value-wins would drop one) and before the home command, so
            # homing stays out of the recording.
            yield self._pace()
            # End the episode span after that round, so the recorder's STOP-time record.io span (which parents
            # to the episode) is captured while it is still in flight. Accepted skew: a producer that also
            # steps during that shared round charges one more span (≤ one control period per episode) to the
            # closing episode — the cooperative scheduler cannot give the recorder a turn alone.
            if abort:
                self._telemetry.abort()
            else:
                self._telemetry.end(virtual_now)
        if self._policy_session:
            self._policy_session.close()
            self._policy_session = None
        self._home(clock)
        self._running = False

    def _handle_directive(self, directive: Directive, clock: pimm.Clock) -> Generator[pimm.Command, None, None]:
        """Dispatch a directive to the episode lifecycle; updates ``_running``."""
        match directive.type:
            case DirectiveType.RUN:
                if not self._running:  # a RUN mid-trial is ignored — the operator finishes before starting anew
                    self._begin_episode(directive.payload or {}, clock)
            case DirectiveType.FINISH:
                if self._running:  # a FINISH while idle is ignored — nothing to finalize
                    yield from self._end_episode(clock, directive.payload)
            case DirectiveType.ABORT:
                yield from self._end_episode(clock, abort=True)
            case _:
                raise ValueError(f'Unknown directive type: {directive.type}')

    def _build_obs(self, clock: pimm.Clock) -> dict[str, Any] | None:
        """Read every observation channel and assemble the policy input dict.

        Raises ``NoValueException`` (caught by ``run``) if any channel has no value
        yet — so inference waits for a complete set of inputs. Returns ``None`` if a
        serializer reports a sample is not ready (e.g. ``robot_state`` while the arm is
        ``RESETTING``), or while a channel is still holding a value delivered before this
        episode's reset, so the harness skips inference rather than feeding a partial or
        stale obs.
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
                if value is None:
                    return None
            for full_name, v in expand_suffixed(name, value):
                if v is not None:
                    inputs[full_name] = v
        if self._awaiting_obs:
            return None
        inputs['wall_time_ns'] = time.time_ns()
        inputs['obs_time_ns'] = clock.now_ns()
        inputs.update(self.context)
        inputs['descriptor'] = self._descriptor  # last, so a context key can't shadow it
        return inputs

    def _emit_commands(self, actions: list[dict[str, Any]]) -> None:
        """Republish-all demux: emit every command channel from this action chunk.

        Each channel emits the ``(ts_ns, value)`` waypoints the chunk carries for
        it; a channel an action omits gets ``[]`` — overwriting its last-value-wins
        signal, so the driver holds. An empty ``actions`` therefore cancels every
        channel.
        """
        for name, emitter in self.commands.items():
            # Wrappers do action-timing math in float seconds (codecs are fps-based);
            # clients on every pimm channel (driver TrajectoryPlayer, dataset writer)
            # expect ns. This is the single explicit seconds->ns seam.
            traj = [(int(a['timestamp'] * 1e9), a[name]) for a in actions if name in a]
            emitter.emit(traj)

    def _inference_delay(self, wall_start: float) -> float:
        """The inference cost to simulate: measured wall time (``True``), a fixed float, or 0 (``False``)."""
        if self._inference_latency is True:  # bool is an int subclass — check identity first
            return time.monotonic() - wall_start
        return float(self._inference_latency)

    def _step(self, clock: pimm.Clock) -> Generator[pimm.Sleep, None, None]:
        """Build obs, call session, demux trajectories into per-channel emissions.

        The session output already carries absolute timestamps (stamped by the
        outermost scheduling wrapper). The harness only demuxes by channel.
        """
        obs = self._build_obs(clock)
        if obs is None:
            return
        self._telemetry.start_rollout(clock.now())

        # Advance the (sim) clock by the inference cost so rollouts feel the model's latency. We only
        # sleep on cycles where inference actually ran (session returned a chunk) — otherwise blocked
        # cycles would slow the harness's directive-handling loop. The trajectory was anchored
        # pre-sleep, so we post-shift it and also bump the scheduling wrapper's internal
        # ``_trajectory_end`` to stay consistent.
        wall_start = time.monotonic()
        actions = self._policy_session(frozen_view(obs))
        if actions is None:
            return
        delay = self._inference_delay(wall_start)
        if delay > 0.0:
            yield pimm.Sleep(delay)
            actions = [{**a, 'timestamp': a['timestamp'] + delay} for a in actions]
            self._bump_schedule_end(delay)

        # Recheck the deadline: the latency sleep (or a slow inference call on a real clock) may have
        # crossed it. Drop the chunk rather than emit past the advertised self-termination point —
        # the run loop fires the timeout FINISH on its next cycle.
        if self._deadline is not None and clock.now() >= self._deadline:
            return

        self._telemetry.step()
        self._emit_commands(actions)

    def _trial_terminal(self, clock: pimm.Clock) -> dict[str, Any] | None:
        """The terminal static payload if a self-driven trial has ended this round, else ``None``.

        The deadline is hard: a truthy ``done`` whose terminal lands within budget records
        ``eval.terminated`` True plus that payload; the budget passing records only False; a terminal past
        the deadline is a timeout, not a late success. Reached only for a task with a deadline — a task-less
        attended episode ends solely on directives, so ``done`` never terminates (or leaks across) it.

        Only a freshly delivered ``done`` terminates: the receiver latches its last value, so a prior
        trial's terminal — whose timestamp precedes this trial's later deadline — would otherwise re-fire.
        Gating on delivery clears it without relying on the producer to republish, so a task with no scene
        ``reset`` (a real embodiment) is handled too.
        """
        done_msg = self.done.read()
        if done_msg.updated and done_msg.data and done_msg.ts <= self._deadline * 1e9:
            return {**done_msg.data, 'eval.terminated': True}
        if clock.now() >= self._deadline:
            return {'eval.terminated': False}
        return None

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        # Home the embodiment before the first episode; each ``_end_episode`` re-homes for the next one, so
        # every episode begins from the home pose (a real arm gets the inter-episode gap to reach it).
        self._home(clock)

        try:
            yield from self._run(should_stop, clock)
        except BaseException:
            # A failure mid-rollout (task.reset / new_session / a session call raising after the episode span
            # was opened) unwinds past the normal span close. Seal the open span here, before the exception
            # reaches ``bind``'s exit flush, or it never exports and its finished children orphan — losing
            # their phases and charging the episode's wall to between_episodes.
            self._telemetry.seal(clock.now())
            raise

    def _run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:  # noqa: C901
        while not should_stop.value:
            # One action per round, mutually exclusive: handle a directive, start the next trial (or exit
            # when the plan is done), finish a self-driven trial that is out of budget or done, or step the
            # policy. Starting takes its own round so a begin never shares a round with a step — inference
            # waits for the producer's post-reset frame-0, which the recorder logs once its open-turn drain
            # has cleared the channels of the pre-reset frame.
            directive_msg = self.directive.read()
            # Read every round so the updated flag clears even mid-episode; a press arriving during a
            # trial is consumed here and never replayed once idle.
            manual_msg = self.manual_command.read()
            if directive_msg.updated:
                yield from self._handle_directive(directive_msg.data, clock)
            elif not self._running:
                if manual_msg.updated and manual_msg.data is not None:
                    self._apply_manual(manual_msg.data, clock)
                elif self._trials is not None:
                    trial = next(self._trials, None)
                    if trial is None:  # plan exhausted — let the recorder commit the final episode, then exit
                        yield pimm.Sleep(0.5)
                        break
                    self._begin_episode(trial, clock)
            elif self._deadline is not None and (terminal := self._trial_terminal(clock)) is not None:
                yield from self._end_episode(clock, terminal)
            else:
                try:
                    yield from self._step(clock)
                except pimm.NoValueException:
                    pass
            yield self._pace()

        if self._running:
            self._finalize_recording()
            virtual_now = clock.now()  # the flush round's clock advance belongs to no rollout
            # Let the recorder commit the queued STOP while the episode span is still open — the same close
            # order as ``_end_episode`` — so its shutdown-flush record.io span parents to the episode, not
            # the pass.
            yield self._pace()
            self._telemetry.end(virtual_now)
        if self._policy_session:
            self._policy_session.close()
        # The harness does not own the policy's lifetime: the caller may run several harnesses over
        # one policy (a multi-eval sweep), so it closes the policy once, after the last run.
