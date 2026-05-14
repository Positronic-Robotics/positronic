import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pimm
from positronic.dataset.ds_writer_agent import DsWriterCommand, Serializers
from positronic.drivers import roboarm
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
                # Anchor to post-inference time so execution starts when inference *finished*.
                # Copy dicts so we don't mutate caller-owned data (sessions may reuse templates).
                now = self._clock.now()
                result = [{**r, 'timestamp': now + r['timestamp']} for r in result]
                self._trajectory_end = result[-1]['timestamp'] if result else None
            return result

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
    """

    class _Session(DelegatingSession):
        """Emits Recover trajectory on robot error, delegates otherwise."""

        def __init__(self, inner: Session, clock: pimm.Clock):
            super().__init__(inner)
            self._clock = clock
            self._in_error = False

        def __call__(self, obs):
            was_ok = not self._in_error
            self._in_error = obs['robot_state.status'] == roboarm.RobotStatus.ERROR

            if self._in_error:
                if was_ok:
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

    The outermost wrapper (typically ``ChunkedSchedule`` or a swap-in alternative
    like RTC) is responsible for producing absolute timestamps.

    By default, wraps the given policy with ``ErrorRecovery | ChunkedSchedule``.
    Pass ``wrap=None`` to skip auto-wrapping (if you've already composed your
    own pipeline).
    """

    def __init__(
        self,
        policy: Policy,
        *,
        static_meta: dict[str, Any] | None = None,
        wrap: PolicyWrapper | Callable[[pimm.Clock], PolicyWrapper] | None = default_wrappers,
    ):
        self._raw_policy = policy
        self._wrap = wrap
        # Wrapping happens in ``run()`` once we have the clock — some wrappers (e.g.
        # ``ChunkedSchedule``) need it. Until then ``self.policy`` mirrors the raw policy.
        self.policy: Policy = policy
        self.context: dict[str, Any] = {}
        self._static_meta = static_meta or {}
        self._session: Session | None = None

        self.frames = pimm.ReceiverDict(self)
        self.robot_state = pimm.ControlSystemReceiver(self)
        self.gripper_state = pimm.ControlSystemReceiver(self)
        self.robot_commands = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemEmitter(self)

        self.directive = pimm.ControlSystemReceiver[Directive](self, default=None, maxsize=3)
        self.ds_command = pimm.ControlSystemEmitter[DsWriterCommand](self)
        self.robot_meta_in = pimm.ControlSystemReceiver(self, default={})

    def _build_episode_meta(self, context: dict[str, Any]) -> dict[str, Any]:
        meta = dict(self._static_meta)
        meta.update(self.robot_meta_in.value)
        session_meta = self._session.meta if self._session else self.policy.meta
        for k, v in flatten_dict(session_meta).items():
            meta[f'inference.policy.{k}'] = v
        meta.update(context)
        return meta

    def _home(self, clock):
        now = clock.now()
        self.robot_commands.emit([(now, roboarm.command.Reset())])
        self.target_grip.emit([(now, 0.0)])

    def _handle_directive(
        self, directive: Directive, clock: pimm.Clock, recording: bool
    ) -> Generator[pimm.Sleep, None, tuple[bool, bool]]:
        """Handle a directive, yielding any necessary pauses. Returns (running, recording)."""
        match directive.type:
            case DirectiveType.RUN:
                if recording:
                    if self._session:
                        self._session.on_episode_complete()
                    self.ds_command.emit(DsWriterCommand.STOP())
                    self._home(clock)
                    yield pimm.Pass()
                self.context = directive.payload or {}
                if self._session:
                    self._session.close()
                self._session = self.policy.new_session(self.context)
                self.ds_command.emit(DsWriterCommand.START(self._build_episode_meta(self.context)))
                return True, True
            case DirectiveType.STOP:
                if recording:
                    self.ds_command.emit(DsWriterCommand.SUSPEND())
                return False, recording
            case DirectiveType.FINISH:
                if recording:
                    if self._session:
                        self._session.on_episode_complete()
                    self.ds_command.emit(DsWriterCommand.STOP(directive.payload or {}))
                    recording = False
                self._home(clock)
                yield pimm.Pass()
                return False, recording
            case DirectiveType.HOME:
                if recording:
                    self.ds_command.emit(DsWriterCommand.ABORT())
                    recording = False
                self._home(clock)
                yield pimm.Pass()
                return False, recording
            case _:
                raise ValueError(f'Unknown directive type: {directive.type}')

    def _build_obs(self, clock: pimm.Clock) -> dict[str, Any] | None:
        """Read sensors and build observation dict. Returns None if not ready."""
        robot_state = self.robot_state.value
        inputs = {
            'robot_state.q': robot_state.q,
            'robot_state.dq': robot_state.dq,
            'robot_state.ee_pose': Serializers.transform_3d(robot_state.ee_pose),
            'robot_state.status': robot_state.status,
            'grip': self.gripper_state.value,
        }
        frame_messages = {k: v.value for k, v in self.frames.items()}
        images = {k: v.array for k, v in frame_messages.items()}
        if len(images) != len(self.frames):
            return None
        inputs.update(images)
        inputs['wall_time_ns'] = time.time_ns()
        inputs['inference_time_ns'] = clock.now_ns()
        inputs.update(self.context)
        return inputs

    def _step(self, clock: pimm.Clock) -> None:
        """Build obs, call session, demux trajectories into per-channel emissions.

        The session output already carries absolute timestamps (stamped by the
        outermost scheduling wrapper). The harness only demuxes by channel.
        """
        obs = self._build_obs(clock)
        if obs is None:
            return

        actions = self._session(frozen_view(obs))
        if actions is None:
            return

        robot_traj = [(cmd['timestamp'], cmd['robot_command']) for cmd in actions]
        grip_traj = [(cmd['timestamp'], cmd['target_grip']) for cmd in actions if 'target_grip' in cmd]

        self.robot_commands.emit(robot_traj)
        self.target_grip.emit(grip_traj)

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
                    self._step(clock)
            except pimm.NoValueException:
                pass
            finally:
                yield pimm.Sleep(0.01)

        if recording:
            if self._session:
                self._session.on_episode_complete()
            self.ds_command.emit(DsWriterCommand.STOP())
        if self._session:
            self._session.close()
        self.policy.close()
