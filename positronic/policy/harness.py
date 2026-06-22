import time
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pimm
from positronic.dataset.ds_writer_agent import DsWriterCommand
from positronic.dataset.serializers import expand_suffixed
from positronic.eval import Embodiment, Task
from positronic.policy.base import Policy, PolicyWrapper, Session
from positronic.policy.wrappers import ChunkedSchedule, default_wrappers
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


class Harness(pimm.ControlSystem):
    """Control system that manages episode lifecycle and forwards trajectories to drivers.

    The harness handles directives (RUN/FINISH/ABORT) and dataset recording. All inference
    intelligence (scheduling, error recovery, blending, absolute time stamping) lives in the
    policy/session layer — the harness just calls the session, demuxes the action dicts into
    per-channel trajectories, and emits.

    ``RUN`` may carry ``inference_latency`` (sim-only inference-cost simulation) and ``eval.seed``
    (handed to the task's scene reset) in its context.
    A ``trials`` plan (a sequence of RUN contexts) makes the harness self-driving: whenever it is
    idle it starts the next trial itself — bounded by the task's ``timeout`` — and exits once the
    plan is exhausted, so the unattended path needs no driver at all. Attended drivers own episode
    termination themselves; directive-driven trials get no deadline.

    A self-driven trial also ends early when the privileged ``done`` signal is delivered within its
    budget: it records ``eval.terminated`` True and the delivered payload in its static data, a
    timed-out one False. Attended episodes ignore ``done`` — the operator's directives end them.

    The ``Embodiment`` provides the observation serializers (which own the canonical key names),
    the command channels, and the home action; the harness reads them to assemble inputs and demux
    actions, treating every channel alike.

    The scheduling wrapper (``ChunkedSchedule``, or a swap-in like RTC) anchors the chunk's
    relative timestamps to absolute time, reading the clock the harness binds in at ``wrap``.

    By default, wraps the given policy with ``ErrorRecovery | ChunkedSchedule``. Pass ``wrap=None``
    to skip auto-wrapping (if you've already composed your own pipeline).
    """

    def __init__(
        self,
        policy: Policy,
        embodiment: Embodiment,
        *,
        task: Task | None = None,
        trials: Iterable[dict[str, Any]] | None = None,
        static_meta: dict[str, Any] | None = None,
        wrap: PolicyWrapper | None = default_wrappers,
        on_episode_complete: Callable[[Session, dict[str, Any]], None] | None = None,
    ):
        assert trials is None or task is not None, 'A trial plan needs a task: its timeout bounds each trial'
        self._raw_policy = policy
        self._embodiment = embodiment
        self._task = task
        # The unattended trial plan: each entry is a RUN context. When set, the run loop starts the
        # next trial whenever it is idle and returns once the plan is exhausted; when None,
        # directives are the only lifecycle source.
        self._trials = iter(trials) if trials is not None else None
        self._wrap = wrap
        # Called with (session, context) when an episode completes successfully (clean
        # FINISH or auto-finalize), never on abort. Used to feed completion bookkeeping
        # like a ``SampledPolicy``'s episode counter, with no sampling knowledge in the harness.
        self._on_complete = on_episode_complete or (lambda session, context: None)
        # Wrapping happens in ``run()`` once we have the clock — ``wrap`` binds it into the sessions
        # it builds (e.g. ``ChunkedSchedule``). Until then ``self.policy`` mirrors the raw policy.
        self.policy: Policy = policy
        self.context: dict[str, Any] = {}
        self._static_meta = static_meta or {}
        self._session: Session | None = None
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

        self._descriptor = embodiment.descriptor
        self.observations = pimm.ReceiverDict(self)
        self.commands = pimm.EmitterDict(self)
        for name in embodiment.observations:
            self.observations[name]  # touch to allocate the port
        for name in embodiment.commands:
            self.commands[name]

        self.directive = pimm.ControlSystemReceiver[Directive](self, default=None, maxsize=3)
        self.ds_command = pimm.ControlSystemEmitter[DsWriterCommand](self)
        self.robot_meta_in = pimm.ControlSystemReceiver(self, default={})
        # Privileged stop-signal: a truthy value within a trial's time budget ends it,
        # recording ``eval.terminated`` True plus that dict in the episode's static data.
        self.done = pimm.ControlSystemReceiver[dict](self, default={})

    def _build_episode_meta(self, context: dict[str, Any]) -> dict[str, Any]:
        meta = dict(self._embodiment.static_meta)
        meta.update(self._static_meta)
        meta.update(self.robot_meta_in.value)
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
        session_meta = self.policy.meta | (self._session.meta if self._session else {})
        for k, v in flatten_dict(session_meta).items():
            meta[f'inference.policy.{k}'] = v
        meta.update(context)
        return meta

    def _home(self, clock):
        now = clock.now_ns()
        for name, value in self._embodiment.home.items():
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
        s = self._session
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
        if self._session is not None:
            self._session.cancel()

    def _finalize_recording(self, payload: dict[str, Any] | None = None) -> None:
        """Commit the live episode: tally completion, cancel the in-flight chunk, stop the recorder —
        stamping the episode's full static meta (plus any terminal payload) at finalize."""
        if self._session:
            self._on_complete(self._session, self.context)
        self._cancel_trajectories()
        self.ds_command.emit(DsWriterCommand.STOP({**self._build_episode_meta(self.context), **(payload or {})}))

    def _begin_episode(self, context: dict[str, Any], clock: pimm.Clock) -> None:
        """Open a fresh episode: set context and session, arm the scene reset, and open the recording.

        A resettable task's ``reset`` only arms the producer, which publishes frame-0 after the harness
        (last in the round). The recorder drains its channels the turn it opens, so the pre-reset frame and
        the inter-episode home command — lingering there from before START — drop out and its first sample
        is the post-reset scene, which the harness infers on once it lands. The trial deadline (a task's
        ``timeout``, bounding policy- and operator-driven trials alike) is armed here; a task-less attended
        session has no deadline and ends only on a directive.
        """
        self.context = context
        if self._task is not None:
            self.context = {**self.context, 'task': self._task.instruction}
        # ``inference_latency`` rides the RUN context (and lands in episode meta with it).
        self._inference_latency = self.context.get('inference_latency', False)
        self._session = self.policy.new_session(self.context)
        self._running = True
        if self._task is not None and self._task.reset is not None:
            self._task.reset(self.context.get('eval.seed'))
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
            # Let the recorder commit the STOP/ABORT before the next START (they share ``ds_command`` —
            # without a tick between, last-value-wins would drop one) and before the home command, so
            # homing stays out of the recording.
            yield self._pace()
        if self._session:
            self._session.close()
            self._session = None
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
        ``RESETTING``), so the harness skips inference rather than feeding a partial obs.
        """
        inputs: dict[str, Any] = {}
        for name, obs in self._embodiment.observations.items():
            value = self.observations[name].value
            if obs.serializer is not None:
                value = obs.serializer(value)
                if value is None:
                    return None
            for full_name, v in expand_suffixed(name, value):
                if v is not None:
                    inputs[full_name] = v
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

        # Advance the (sim) clock by the inference cost so rollouts feel the model's latency. We only
        # sleep on cycles where inference actually ran (session returned a chunk) — otherwise blocked
        # cycles would slow the harness's directive-handling loop. The trajectory was anchored
        # pre-sleep, so we post-shift it and also bump the scheduling wrapper's internal
        # ``_trajectory_end`` to stay consistent.
        wall_start = time.monotonic()
        actions = self._session(frozen_view(obs))
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

        self._emit_commands(actions)

    def _trial_terminal(self, clock: pimm.Clock) -> dict[str, Any] | None:
        """The terminal static payload if a self-driven trial has ended this round, else ``None``.

        The deadline is hard: a truthy ``done`` whose terminal lands within budget records
        ``eval.terminated`` True plus that payload; the budget passing records only False; a terminal past
        the deadline is a timeout, not a late success. Reached only for a task with a deadline — a task-less
        attended episode ends solely on directives, so ``done`` never terminates (or leaks across) it.
        """
        done_msg = self.done.read()
        if done_msg.data and done_msg.ts <= self._deadline * 1e9:
            return {**done_msg.data, 'eval.terminated': True}
        if clock.now() >= self._deadline:
            return {'eval.terminated': False}
        return None

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        # Resolve wrap now that we have the clock — ``wrap`` binds it into the sessions it builds.
        if self._wrap is None:
            self.policy = self._raw_policy
        else:
            self.policy = self._wrap.wrap(self._raw_policy, clock.now)

        # Home the embodiment before the first episode; each ``_end_episode`` re-homes for the next one, so
        # every episode begins from the home pose (a real arm gets the inter-episode gap to reach it).
        self._home(clock)

        while not should_stop.value:
            # One action per round, mutually exclusive: handle a directive, start the next trial (or exit
            # when the plan is done), finish a self-driven trial that is out of budget or done, or step the
            # policy. Starting takes its own round so a begin never shares a round with a step — inference
            # waits for the producer's post-reset frame-0, which the recorder logs once its open-turn drain
            # has cleared the channels of the pre-reset frame.
            directive_msg = self.directive.read()
            if directive_msg.updated:
                yield from self._handle_directive(directive_msg.data, clock)
            elif not self._running:
                if self._trials is not None:
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
        if self._session:
            self._session.close()
        self.policy.close()
