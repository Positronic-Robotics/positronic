import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from opentelemetry.trace import Span

import pimm
from positronic import keys, telemetry, telemetry_keys
from positronic.dataset.ds_writer_agent import DsWriterCommand
from positronic.dataset.serializers import expand_suffixed
from positronic.drivers.roboarm.ik import assert_default_frame
from positronic.eval import Embodiment, Rollout
from positronic.policy.base import Policy, Session
from positronic.policy.wrappers import ChunkedSchedule
from positronic.utils import flatten_dict, frozen_view

# How far from now an action may be scheduled. A chunk spans seconds, so this is loose enough that no real
# trajectory approaches it, and tight enough to catch a rig-side stack that left timestamps relative to the
# chunk (decades behind) or anchored them twice (decades ahead).
MAX_ACTION_SKEW_SEC = 60.0


def _assert_anchored(actions: list[dict[str, Any]], now: float) -> None:
    """Reject a chunk whose timestamps are not times on the harness clock."""
    skew = max((abs(action[keys.ACTION_TIMESTAMP] - now) for action in actions), default=0.0)
    if skew > MAX_ACTION_SKEW_SEC:
        raise ValueError(
            f'Action scheduled {skew:.0f}s from now, over the {MAX_ACTION_SKEW_SEC:.0f}s bound: the rig-side '
            f'stack is not anchoring chunks to the harness clock'
        )


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


# The rollout's budget in a ``Directive.RUN`` payload; its instruction rides ``keys.TASK``.
RUN_TIMEOUT = 'timeout'


class _EpisodeTelemetry:
    """The live rollout's wall-clock telemetry: the episode span, its index, its step count and the virtual
    instant it began. Inert while telemetry is unbound, so the harness calls it unconditionally.

    The span stays anchored while open, so the rollout's phase spans (reset, env.step, policy.infer,
    record.io) parent to it rather than to the pass.
    """

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
        """Close a rollout abandoned mid-flight by a raising ``reset`` / ``new_session`` / session call, stamped
        like a clean end and marked ``episode.partial``. Ending it is what exports it — the batch processor drops
        an unended span, orphaning the finished children and losing their phases. Partial rather than aborted so
        the reduce keeps it. Inert when no span is open."""
        if self._span is None:
            return
        telemetry.set_attrs(self._span, **{telemetry_keys.ATTR_EPISODE_PARTIAL: True})
        self._close(virtual_now)
        telemetry.force_flush()

    def _close(self, virtual_now: float) -> None:
        # A rollout whose first observation never landed — a reset that raised, or a rollout already done before
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
    """Control system that runs the episode lifecycle and forwards trajectories to drivers.

    Handles directives (RUN/FINISH/ABORT) and dataset recording. Inference intelligence — scheduling,
    error recovery, blending, absolute time stamping — lives in the policy/session layer; the harness
    calls the session, demuxes the action dicts into per-channel trajectories, and emits.

    Every episode runs one ``Rollout``. A ``rollouts`` plan makes the harness self-driving: it starts the
    next one whenever idle and returns once the plan is exhausted, so the unattended path needs no driver.
    Without a plan a RUN directive carries the rollout instead, and the two are the same episode to
    everything downstream. Only the rollout's ``instruction`` reaches an observation — its scene goes to
    ``reset``, to the recording, and to ``new_session``, where a sampling policy groups by it, so no model
    is shown the ground truth its rollout is scored against.

    A rollout's ``timeout`` bounds it, self-driven or operator-driven, so an attended episode given a
    budget still terminates at the deadline if the operator never sends FINISH. A bounded rollout also ends
    early on a truthy privileged ``done``, recording ``eval.terminated`` True and the delivered payload in
    its static data; a timeout records False. A rollout without a budget has no deadline, so ``done`` does not
    terminate it and only directives end it. ``inference_latency`` applies to every episode of the run.

    The ``Embodiment`` supplies the observation serializers (which own the canonical key names), the command
    channels and the home action. The policy owns its wrapper stack; the harness runs what it is given,
    passing ``new_session`` the clock the scheduling wrapper anchors chunks to.
    """

    def __init__(
        self,
        policy: Policy,
        embodiment: Embodiment,
        *,
        rollouts: list[Rollout] | None = None,
        reset: Callable[[dict[str, Any]], None] | None = None,
        static_meta: dict[str, Any] | None = None,
        inference_latency: bool | float = False,
    ):
        self._embodiment = embodiment
        # The unattended rollout plan; when None, directives are the only lifecycle source.
        self._plan = enumerate(rollouts) if rollouts is not None else None
        self._rollout_count = len(rollouts) if rollouts is not None else None
        self._reset = reset
        self.policy: Policy = policy
        self.context: dict[str, Any] = {}
        # The rollout's model-facing half, assembled per episode: the instruction and nothing else. The
        # scene, seed and plan position are recorded and reach ``new_session``, which a sampling policy
        # groups episodes by, but they never enter an observation and so never reach a model.
        self._policy_inputs: dict[str, Any] = {}
        self._static_meta = static_meta or {}
        self._policy_session: Session | None = None
        # True between RUN and FINISH/ABORT: the trial is live — stepping and recording happen together.
        self._running = False
        # Sim-only: ``True`` advances the sim clock by the measured wall cost of the inference call, a float
        # by a fixed deterministic amount (the reproducible golden). The chunk is anchored before the sleep,
        # so ``_step`` post-shifts it.
        self._inference_latency: bool | float = inference_latency
        # The live rollout's ``timeout`` and the deadline it arms; both ``None`` for an unbounded episode, which
        # ends on directives.
        self._timeout: float | None = None
        self._deadline: float | None = None
        # Whether this episode's first observation has landed. Until it does the deadline stands where the
        # reset put it, which bounds an episode whose first observation never arrives.
        self._rollout_started = False
        # Wall-clock telemetry for the live rollout, opened under ``--timing`` and inert otherwise.
        self._telemetry = _EpisodeTelemetry()
        # Channels that have not delivered since this episode's reset. A receiver latches its last value, so
        # emptying this set is what keeps the first inference off the previous episode's final frame.
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
        # Privileged stop-signal: a truthy value within the trial's budget ends it.
        self.done = pimm.ControlSystemReceiver[dict](self, default={})

    def _statics(self) -> dict[str, Any]:
        """What is known about the rig before the episode runs, live values winning."""
        return self._embodiment.static_meta | self._static_meta | self.robot_meta_in.value

    def _build_episode_meta(self) -> dict[str, Any]:
        meta = self._statics()
        # TODO: also stamp the eval's catalog name and its resolved config — both need configuronic
        # introspection that does not exist yet.
        meta[keys.EVAL_UNIVERSE] = 'sim' if self._embodiment.simulated else 'real'
        meta[keys.EVAL_EMBODIMENT] = self._embodiment.descriptor
        meta[keys.INFERENCE_LATENCY] = self._inference_latency
        # ``policy.meta`` is the static baseline; the session overlays per-episode specifics (e.g. the
        # sampled sub-policy) and wins on conflict.
        session_meta = self.policy.meta | (self._policy_session.meta if self._policy_session else {})
        for k, v in flatten_dict(session_meta).items():
            meta[f'{keys.POLICY_META}.{k}'] = v
        meta.update(self.context)
        return meta

    def _emit_now(self, action: dict[str, Any], clock: pimm.Clock) -> None:
        now = clock.now_ns()
        for name, value in action.items():
            self.commands[name].emit([(now, value)])

    def _home(self, clock: pimm.Clock) -> None:
        self._emit_now(self._embodiment.home, clock)

    def _pace(self) -> pimm.Command:
        """Sim: yield, so the simulator's control-period sleep is the sole time-master and the policy reads
        each observation instantly. Real: sleep the poll period to hold wall-clock rate."""
        return pimm.Yield() if self._embodiment.simulated else pimm.Sleep(0.01)

    def _bump_schedule_end(self, delta_sec: float) -> None:
        """Shift the active ``ChunkedSchedule._Session`` ``_trajectory_end`` by ``delta_sec``.

        ``_step`` post-shifts a chunk's timestamps to pay for ``inference_latency``; the wrapper's
        end-of-chunk gate must follow, or it re-infers before the driver has played the shifted trajectory.

        TODO: reaching the wrapper by attribute-name guessing and writing its private state breaks on any rename.
        """
        s = self._policy_session
        while s is not None:
            if isinstance(s, ChunkedSchedule._Session) and s._trajectory_end is not None:
                s._trajectory_end += delta_sec
                return
            s = getattr(s, '_inner', None)

    def _cancel_trajectories(self) -> None:
        """Drop any in-flight chunk from drivers and from the recording's tail.

        Emits ``[]`` on every command channel, so each driver's ``TrajectoryPlayer`` clears its buffer
        (devices hold position) and ``TrajectoryOverrideSerializer`` drops its uncommitted tail. Must precede
        ``STOP_EPISODE``, which flushes the recording's serializers and would otherwise commit canceled
        waypoints. Also cancels the session's scheduling state so the next inference is not held back.
        """
        self._emit_commands([])
        if self._policy_session is not None:
            self._policy_session.cancel()

    def _finalize_recording(
        self, clock: pimm.Clock, payload: dict[str, Any] | None = None
    ) -> Generator[pimm.Command, None, None]:
        """Commit the live episode: cancel the in-flight chunk, stop the recorder — stamping the
        episode's full static meta (plus any terminal payload) — then close its span."""
        self._cancel_trajectories()
        self.ds_command.emit(DsWriterCommand.STOP({**self._build_episode_meta(), **(payload or {})}))
        virtual_now = clock.now()  # before the round below, whose sim-clock advance belongs to no rollout
        # Give the recorder a round to commit the STOP before the next START (they share ``ds_command``, where
        # last-value-wins would drop one) and before the home command, so homing stays out of the recording.
        yield self._pace()
        # After that round, so the recorder's STOP-time record.io span is still in flight and parents to the
        # episode. Accepted skew: a producer stepping in that shared round charges one span (≤ one control
        # period) to the closing episode — the cooperative scheduler cannot give the recorder a turn alone.
        self._telemetry.end(virtual_now)

    def _begin_episode(self, rollout: Rollout, clock: pimm.Clock, position: dict[str, Any]) -> None:
        """Open a fresh episode: stage the scene, fix the episode context and session, open the recording.

        ``position`` is where this rollout sits in the plan, and is empty for one a directive carried, which
        sits in no plan.

        ``reset`` only arms the producer, which publishes the first observation on a later round. The
        recorder drains its channels the turn it opens, so the pre-reset frame and the inter-episode home
        command drop out and its first sample is the post-reset scene. The deadline is armed here and moved
        to that first observation once it lands, so an episode that never gets one is still bounded.
        """
        self._awaiting_obs = set(self._embodiment.observations)
        self._rollout_started = False
        self._timeout = rollout.timeout
        context: dict[str, Any] = {**rollout.scene, **position}
        if rollout.timeout is not None:
            context[keys.EVAL_TIMEOUT] = rollout.timeout
        # Before the reset, so the reset and the rollout's other phase spans parent to the episode span.
        self._telemetry.begin(context)
        # Reset before opening the session: an embodiment that only learns its goal on reset (a remote env
        # reports it then) resolves its instruction here, so the session context — and the sampling it
        # drives — reads the instruction once it is known.
        if self._reset is not None:
            with telemetry.span(telemetry_keys.SPAN_RESET):
                self._reset(rollout.scene)
        instruction = rollout.instruction
        self._policy_inputs = {keys.TASK: instruction} if instruction is not None else {}
        self.context = context | self._policy_inputs
        self._policy_session = self.policy.new_session(self.context, clock.now)
        self._running = True
        self._deadline = clock.now() + rollout.timeout if rollout.timeout is not None else None
        self.ds_command.emit(DsWriterCommand.START())

    def _end_episode(
        self, clock: pimm.Clock, payload: dict[str, Any] | None = None, *, abort: bool = False
    ) -> Generator[pimm.Command, None, None]:
        """Close the live episode: finalize (or abort) the recording, release the session, home devices.

        Releasing the session here, not only at shutdown, closes a ``RemoteSession``'s websocket promptly, so
        the offboard server's per-session cleanup (active-session decrement, idle watchdog) runs now.
        """
        if self._running:
            if abort:
                self._cancel_trajectories()  # abort has no finalize to do it — stop drivers before the home
                self.ds_command.emit(DsWriterCommand.ABORT())
                yield self._pace()  # the settling round a finalize also takes, before the home command
                self._telemetry.abort()
            else:
                yield from self._finalize_recording(clock, payload)
        if self._policy_session:
            self._policy_session.close()
            self._policy_session = None
        self._home(clock)
        self._running = False

    @staticmethod
    def _directive_rollout(payload: dict[str, Any]) -> Rollout:
        """A RUN payload read as one rollout.

        An operator names the goal and, where the rollout is to end on its own, a budget; every other key
        describes the scene they staged, and is recorded like any other scene.
        """
        scene = {k: v for k, v in payload.items() if k not in (keys.TASK, RUN_TIMEOUT)}
        return Rollout(instruction=payload.get(keys.TASK), timeout=payload.get(RUN_TIMEOUT), scene=scene)

    def _handle_directive(self, directive: Directive, clock: pimm.Clock) -> Generator[pimm.Command, None, None]:
        """Dispatch a directive to the episode lifecycle; updates ``_running``."""
        match directive.type:
            case DirectiveType.RUN:
                if not self._running:  # a RUN mid-trial is ignored — the operator finishes before starting anew
                    self._begin_episode(self._directive_rollout(directive.payload or {}), clock, {})
            case DirectiveType.FINISH:
                if self._running:  # a FINISH while idle is ignored — nothing to finalize
                    yield from self._end_episode(clock, directive.payload)
            case DirectiveType.ABORT:
                yield from self._end_episode(clock, abort=True)
            case _:
                raise ValueError(f'Unknown directive type: {directive.type}')

    def _build_obs(self, clock: pimm.Clock) -> dict[str, Any] | None:
        """Read every observation channel and assemble the policy input dict.

        Raises ``NoValueException`` if any channel has no value yet. Returns ``None`` while a serializer
        reports a sample is not ready (``robot_state`` during a ``RESETTING`` arm) or a channel still holds a
        pre-reset value — either way the harness skips inference rather than feed a partial or stale obs.
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
        inputs[keys.WALL_TIME_NS] = time.time_ns()
        inputs[keys.OBS_TIME_NS] = clock.now_ns()
        inputs.update(self._policy_inputs)
        inputs['descriptor'] = self._descriptor
        return inputs

    def _emit_commands(self, actions: list[dict[str, Any]]) -> None:
        """Republish-all demux: emit every command channel from this action chunk.

        Each channel gets the ``(ts_ns, value)`` waypoints the chunk carries for it; a channel the chunk
        omits gets ``[]``, overwriting its last-value-wins signal so the driver holds. An empty chunk
        therefore cancels every channel.
        """
        for name, emitter in self.commands.items():
            # Wrappers do action-timing math in float seconds; every pimm channel client expects ns. This is
            # the single explicit seconds->ns seam.
            traj = [(int(a[keys.ACTION_TIMESTAMP] * 1e9), a[name]) for a in actions if name in a]
            emitter.emit(traj)

    def _inference_delay(self, wall_start: float) -> float:
        """The inference cost to simulate: measured wall time (``True``), a fixed float, or 0 (``False``)."""
        if self._inference_latency is True:  # bool is an int subclass — check identity first
            return time.monotonic() - wall_start
        return float(self._inference_latency)

    def _step(self, clock: pimm.Clock) -> Generator[pimm.Sleep, None, None]:
        """Build obs, call the session, demux its chunk into per-channel emissions.

        The chunk already carries absolute timestamps, stamped by the outermost scheduling wrapper.
        """
        obs = self._build_obs(clock)
        if obs is None:
            return
        if not self._rollout_started:
            # The rollout begins at its first observation, not when the reset returned: a reset only asks the
            # producer for a scene, and the turns spent delivering it are neither the trial's budget nor its
            # duration.
            self._rollout_started = True
            self._telemetry.start_rollout(clock.now())
            if self._timeout is not None:
                self._deadline = clock.now() + self._timeout

        # Advance the sim clock by the inference cost so rollouts feel the model's latency, but only on
        # cycles where inference ran — sleeping on blocked cycles would slow directive handling. The chunk
        # was anchored pre-sleep, so it is post-shifted and the wrapper's gate bumped to match.
        wall_start = time.monotonic()
        actions = self._policy_session(frozen_view(obs))
        if actions is None:
            return
        delay = self._inference_delay(wall_start)
        if delay > 0.0:
            yield pimm.Sleep(delay)
            actions = [{**a, keys.ACTION_TIMESTAMP: a[keys.ACTION_TIMESTAMP] + delay} for a in actions]
            self._bump_schedule_end(delay)

        # The latency sleep (or a slow inference call on a real clock) may have crossed the deadline. Drop
        # the chunk rather than emit past the advertised self-termination point; the run loop fires FINISH
        # next cycle.
        if self._deadline is not None and clock.now() >= self._deadline:
            return

        self._telemetry.step()
        _assert_anchored(actions, clock.now())
        self._emit_commands(actions)

    def _rollout_terminal(self, clock: pimm.Clock) -> dict[str, Any] | None:
        """The terminal static payload if a self-driven rollout has ended this round, else ``None``.

        The deadline is hard: a truthy ``done`` delivered within budget records ``eval.terminated`` True plus
        its payload, the budget passing records False, and a terminal past the deadline is a timeout rather
        than a late success. Only a freshly delivered ``done`` counts — the receiver latches its last value,
        so a prior rollout's terminal would otherwise re-fire; gating on delivery clears it without asking the
        producer to republish. Reached only for a rollout with a deadline.
        """
        done_msg = self.done.read()
        if done_msg.updated and done_msg.data and done_msg.ts <= self._deadline * 1e9:
            return {**done_msg.data, keys.EVAL_TERMINATED: True}
        if clock.now() >= self._deadline:
            return {keys.EVAL_TERMINATED: False}
        return None

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        # Home the embodiment before the first episode; each ``_end_episode`` re-homes for the next one, so
        # every episode begins from the home pose (a real arm gets the inter-episode gap to reach it).
        self._home(clock)

        try:
            yield from self._run(should_stop, clock)
        except BaseException:
            # A failure mid-rollout unwinds past the normal span close. Seal the open span before the
            # exception reaches ``bind``'s exit flush, or it never exports and its finished children orphan,
            # losing their phases and charging the episode's wall to between_episodes.
            self._telemetry.seal(clock.now())
            raise

    def _run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:  # noqa: C901
        while not should_stop.value:
            # One action per round, mutually exclusive: handle a directive, start the next trial (or exit
            # when the plan is done), finish a self-driven trial that is out of budget or done, or step the
            # policy. Starting takes its own round so a begin never shares one with a step — inference waits
            # for the producer's post-reset observation, which the recorder logs once its open-turn drain has
            # cleared the channels.
            directive_msg = self.directive.read()
            # Read every round so the flag clears mid-episode; a press during a trial is consumed, not replayed.
            manual_msg = self.manual_command.read()
            # Both receivers carry a default, so ``read`` always yields a message.
            assert directive_msg is not None and manual_msg is not None
            if directive_msg.updated:
                yield from self._handle_directive(directive_msg.data, clock)
            elif not self._running:
                if manual_msg.updated and manual_msg.data is not None:
                    self._emit_now(manual_msg.data, clock)
                elif self._plan is not None:
                    entry = next(self._plan, None)
                    if entry is None:  # plan exhausted — let the recorder commit the final episode, then exit
                        yield pimm.Sleep(0.5)
                        break
                    index, rollout = entry
                    position = {keys.EVAL_ROLLOUT_INDEX: index, keys.EVAL_ROLLOUT_COUNT: self._rollout_count}
                    self._begin_episode(rollout, clock, position)
            elif self._deadline is not None and (terminal := self._rollout_terminal(clock)) is not None:
                yield from self._end_episode(clock, terminal)
            else:
                try:
                    yield from self._step(clock)
                except pimm.NoValueException:
                    pass
            yield self._pace()

        if self._running:
            yield from self._finalize_recording(clock)
        if self._policy_session:
            self._policy_session.close()
        # The harness does not own the policy's lifetime: the caller may run several harnesses over
        # one policy (a multi-eval sweep), so it closes the policy once, after the last run.
