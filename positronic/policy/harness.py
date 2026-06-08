import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pimm
from positronic.dataset.ds_writer_agent import DsWriterCommand
from positronic.dataset.serializers import expand_suffixed
from positronic.drivers import roboarm
from positronic.embodiment import Embodiment
from positronic.policy.base import DelegatingSession, Policy, PolicyWrapper, Session
from positronic.utils import flatten_dict, frozen_view


class DirectiveType(Enum):
    """Directive types for the harness."""

    RUN = 'run'
    STOP = 'stop'
    FINISH = 'finish'
    HOME = 'home'


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
    def STOP(cls) -> 'Directive':
        """Stop running the policy; devices hold position, recording suspended."""
        return cls(DirectiveType.STOP, None)

    @classmethod
    def FINISH(cls, **kwargs) -> 'Directive':
        """Finalize the recording with optional eval data, then home devices."""
        return cls(DirectiveType.FINISH, kwargs)

    @classmethod
    def HOME(cls, preset: str = 'home') -> 'Directive':
        """Abort recording and send devices to a named safe state."""
        return cls(DirectiveType.HOME, preset)


# ---------------------------------------------------------------------------
# Policy wrappers — composable concerns extracted from the harness
# ---------------------------------------------------------------------------


class ChunkedSchedule(PolicyWrapper):
    """Wait for current trajectory to finish before calling inner policy again.

    Owns relative→absolute time conversion: inner layers (codecs, models) emit
    relative timestamps; this wrapper anchors them to ``clock.now()`` *after*
    inner inference returns, so execution aligns to inference-finish (not
    inference-start). Other scheduling strategies (RTC, temporal ensembling)
    will plug in here with their own timing policies.

    Returns ``None`` (meaning "keep executing current trajectory") until the
    last action's timestamp has been reached, then calls the inner policy.

    Composable via ``|``::

        pipeline = ErrorRecovery() | ChunkedSchedule(clock) | codec
        wrapped = pipeline.wrap(model)
    """

    class _Session(DelegatingSession):
        """Skips inner calls while the current trajectory plays; stamps absolute on emit."""

        def __init__(self, inner: Session, clock: pimm.Clock):
            super().__init__(inner)
            self._clock = clock
            self._trajectory_end: float | None = None

        def __call__(self, obs):
            if self._trajectory_end is not None and self._clock.now() < self._trajectory_end:
                return None
            result = self._inner(obs)
            if result is not None:
                # A single-action session may return a bare dict, and a no-codec path may
                # omit ``timestamp`` (servers can stamp/truncate themselves); normalize both
                # so an immediate action executes instead of raising.
                if isinstance(result, dict):
                    result = [result]
                # Anchor to post-inference time so execution starts when inference *finished*.
                # Copy dicts so we don't mutate caller-owned data (sessions may reuse templates).
                now = self._clock.now()
                result = [{**r, 'timestamp': now + r.get('timestamp', 0.0)} for r in result]
                self._trajectory_end = result[-1]['timestamp'] if result else None
            return result

        def cancel(self):
            self._trajectory_end = None
            super().cancel()

    def __init__(self, clock: pimm.Clock):
        self._clock = clock

    def wrap_session(self, inner: Session, context):
        return ChunkedSchedule._Session(inner, self._clock)


class ErrorRecovery(PolicyWrapper):
    """Wraps a policy to handle robot errors by emitting Recover commands.

    On error: emits a single Recover trajectory, then returns None until
    the robot recovers. On recovery: resumes normal inference.

    Composable via ``|``::

        pipeline = ErrorRecovery(clock) | ChunkedSchedule(clock) | codec
        wrapped = pipeline.wrap(model)

    TODO: this wrapper is not name-free. It hard-codes the ``robot_state.error``
    observation and the ``robot_command`` channel (with a Franka ``Recover``), so it
    only fits Franka-named embodiments; others must disable ``default_wrappers``. How
    an embodiment should declare its error signal and recovery action is still open.
    """

    class _Session(DelegatingSession):
        """Emits Recover trajectory on robot error, delegates otherwise."""

        def __init__(self, inner: Session, clock: pimm.Clock):
            super().__init__(inner)
            self._clock = clock
            self._in_error = False

        def __call__(self, obs):
            was_ok = not self._in_error
            self._in_error = obs['robot_state.error'] == 1

            if self._in_error:
                if was_ok:
                    # Reset any inner scheduling state so post-recovery doesn't stall
                    # on a stale trajectory_end from the pre-error chunk.
                    self._inner.cancel()
                    return [{'robot_command': roboarm.command.Recover(), 'timestamp': self._clock.now()}]
                return None

            return self._inner(obs)

    def __init__(self, clock: pimm.Clock):
        self._clock = clock

    def wrap_session(self, inner: Session, context):
        return ErrorRecovery._Session(inner, self._clock)


def default_wrappers(clock: pimm.Clock) -> PolicyWrapper:
    """Default wrapper pipeline: error recovery + chunked scheduling bound to the harness clock."""
    return ErrorRecovery(clock) | ChunkedSchedule(clock)


class Harness(pimm.ControlSystem):
    """Control system that manages episode lifecycle and forwards trajectories to drivers.

    The harness handles directives (RUN/STOP/FINISH/HOME) and dataset recording.
    All inference intelligence (scheduling, error recovery, blending, absolute
    time stamping) lives in the policy/session layer — the harness just calls
    the session, demuxes the action dicts into per-channel trajectories, and
    emits.

    The ``Embodiment`` provides the observation serializers (which own the
    canonical key names), the command channels, and the home action; the harness
    reads them to assemble inputs and demux actions, treating every channel alike.

    The outermost wrapper (typically ``ChunkedSchedule`` or a swap-in alternative
    like RTC) is responsible for producing absolute timestamps.

    By default, wraps the given policy with ``ErrorRecovery | ChunkedSchedule``.
    Pass ``wrap=None`` to skip auto-wrapping (if you've already composed your
    own pipeline).
    """

    def __init__(
        self,
        policy: Policy,
        embodiment: Embodiment,
        *,
        static_meta: dict[str, Any] | None = None,
        wrap: PolicyWrapper | Callable[[pimm.Clock], PolicyWrapper] | None = default_wrappers,
        simulate_inference: bool | float = False,
        on_episode_complete: Callable[[Session, dict[str, Any]], None] | None = None,
    ):
        self._raw_policy = policy
        self._embodiment = embodiment
        self._wrap = wrap
        # Called with (session, context) when an episode completes successfully (clean
        # STOP/FINISH), never on abort. Used to feed completion bookkeeping like a
        # ``SampledPolicy``'s episode counter, with no sampling knowledge in the harness.
        self._on_complete = on_episode_complete or (lambda session, context: None)
        # Wrapping happens in ``run()`` once we have the clock — some wrappers (e.g.
        # ``ChunkedSchedule``) need it. Until then ``self.policy`` mirrors the raw policy.
        self.policy: Policy = policy
        self.context: dict[str, Any] = {}
        self._static_meta = static_meta or {}
        self._session: Session | None = None
        # ``True`` advances the (sim) clock by the wall-clock cost of the inference
        # call; a float is a fixed deterministic delay (used by the reproducible
        # golden). Sleep is yielded BEFORE ``ChunkedSchedule`` reads ``clock.now()``
        # so the trajectory is anchored to inference-finish, not inference-start.
        self.simulate_inference = simulate_inference

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

    def _build_episode_meta(self, context: dict[str, Any]) -> dict[str, Any]:
        meta = dict(self._embodiment.static_meta)
        meta.update(self._static_meta)
        meta.update(self.robot_meta_in.value)
        meta['inference.simulate_inference'] = self.simulate_inference
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

    def _bump_schedule_end(self, delta_sec: float) -> None:
        """Shift the active ``ChunkedSchedule._Session`` ``_trajectory_end`` by ``delta_sec``.

        Used by ``simulate_inference``: the session anchored the chunk pre-sleep,
        then we slept and post-shifted the emitted timestamps. The scheduling
        wrapper's internal end-of-chunk gate must move forward too, or it will
        re-infer before the driver has actually played the (shifted) trajectory.
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

    def _handle_directive(
        self, directive: Directive, clock: pimm.Clock, recording: bool
    ) -> Generator[pimm.Sleep, None, tuple[bool, bool]]:
        """Handle a directive, yielding any necessary pauses. Returns (running, recording)."""
        match directive.type:
            case DirectiveType.RUN:
                if recording:
                    if self._session:
                        self._on_complete(self._session, self.context)
                    self._cancel_trajectories()
                    self.ds_command.emit(DsWriterCommand.STOP())
                    self._home(clock)
                    yield pimm.Yield()
                self.context = directive.payload or {}
                if self._session:
                    self._session.close()
                self._session = self.policy.new_session(self.context)
                self.ds_command.emit(DsWriterCommand.START(self._build_episode_meta(self.context)))
                return True, True
            case DirectiveType.STOP:
                # SUSPEND before the cancel: the writer flushes the due trajectory
                # prefix (dropping the future tail) when it handles SUSPEND, then skips
                # inputs — so the `[]` cancel that follows is only acted on by the robot.
                if recording:
                    self.ds_command.emit(DsWriterCommand.SUSPEND())
                self._cancel_trajectories()
                return False, recording
            case DirectiveType.FINISH:
                if recording:
                    if self._session:
                        self._on_complete(self._session, self.context)
                    self._cancel_trajectories()
                    self.ds_command.emit(DsWriterCommand.STOP(directive.payload or {}))
                    recording = False
                # End the per-episode session here (not just at RUN/shutdown) so a
                # ``RemoteSession``'s websocket closes promptly and the offboard server's
                # per-session cleanup (active-session decrement, idle watchdog) runs now.
                if self._session:
                    self._session.close()
                    self._session = None
                self._home(clock)
                yield pimm.Yield()
                return False, recording
            case DirectiveType.HOME:
                if recording:
                    self.ds_command.emit(DsWriterCommand.ABORT())
                    recording = False
                if self._session:  # HOME aborts the episode; release the session like FINISH
                    self._session.close()
                    self._session = None
                self._home(clock)
                yield pimm.Yield()
                return False, recording
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
        inputs['inference_time_ns'] = clock.now_ns()
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

    def _step(self, clock: pimm.Clock) -> Generator[pimm.Sleep, None, None]:
        """Build obs, call session, demux trajectories into per-channel emissions.

        The session output already carries absolute timestamps (stamped by the
        outermost scheduling wrapper). The harness only demuxes by channel.
        """
        obs = self._build_obs(clock)
        if obs is None:
            return

        # Advance the (sim) clock by the inference cost so rollouts feel the
        # model's latency. ``True`` measures real wall time around the session
        # call; a float is a fixed deterministic delay (used by the golden).
        # We only sleep on cycles where inference actually ran (session
        # returned a chunk) — otherwise blocked cycles would slow the
        # harness's directive-handling loop. The trajectory was anchored
        # pre-sleep, so we post-shift it and also bump the scheduling
        # wrapper's internal ``_trajectory_end`` to stay consistent.
        wall_start = time.monotonic()
        actions = self._session(frozen_view(obs))
        if actions is None:
            return
        if self.simulate_inference is True:  # bool is an int subclass — check identity first
            delay = time.monotonic() - wall_start
        elif self.simulate_inference:
            delay = float(self.simulate_inference)
        else:
            delay = 0.0
        if delay > 0.0:
            yield pimm.Sleep(delay)
            actions = [{**a, 'timestamp': a['timestamp'] + delay} for a in actions]
            self._bump_schedule_end(delay)

        self._emit_commands(actions)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        # Resolve wrap now that we have the clock — some wrappers (e.g. ChunkedSchedule) need it.
        if self._wrap is None:
            self.policy = self._raw_policy
        elif isinstance(self._wrap, PolicyWrapper):
            self.policy = self._wrap.wrap(self._raw_policy)
        else:  # factory: Callable[[Clock], PolicyWrapper]
            self.policy = self._wrap(clock).wrap(self._raw_policy)

        running = False
        recording = False

        while not should_stop.value:
            directive_msg = self.directive.read()
            if directive_msg.updated:
                running, recording = yield from self._handle_directive(directive_msg.data, clock, recording)

            try:
                if running:
                    yield from self._step(clock)
            except pimm.NoValueException:
                pass
            finally:
                yield pimm.Sleep(0.01)

        if recording:
            if self._session:
                self._on_complete(self._session, self.context)
            # Stop the live drivers before finalizing (matches FINISH/RUN). The
            # recording's unexecuted chunk tail is dropped by the serializer flush
            # cutoff at STOP, not by this cancel.
            self._cancel_trajectories()
            self.ds_command.emit(DsWriterCommand.STOP())
        if self._session:
            self._session.close()
        self.policy.close()
