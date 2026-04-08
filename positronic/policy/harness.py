import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pimm
from positronic.dataset.ds_writer_agent import DsWriterCommand, Serializers
from positronic.drivers import roboarm
from positronic.policy.base import DelegatingPolicy, DelegatingSession, Policy, PolicyWrapper, Session
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


class _ChunkedSession(DelegatingSession):
    """Calls inner session only when the previous trajectory has been consumed."""

    def __init__(self, inner: Session):
        super().__init__(inner)
        self._trajectory_end: float | None = None

    def __call__(self, obs):
        now = obs.get('inference_time_ns', 0) / 1e9
        if self._trajectory_end is not None and now < self._trajectory_end:
            return None
        result = self._inner(obs)
        if result is not None and isinstance(result, list) and result:
            self._trajectory_end = result[-1].get('timestamp')
        return result


class ChunkedSchedule(PolicyWrapper):
    """Wait for current trajectory to finish before calling inner policy again.

    Returns ``None`` (meaning "keep executing current trajectory") until the
    last action's timestamp has been reached, then calls the inner policy.

    Composable via ``|``::

        pipeline = ErrorRecovery() | ChunkedSchedule() | codec
        wrapped = pipeline.wrap(model)
    """

    def wrap(self, policy: Policy) -> Policy:
        return _ChunkedSchedulePolicy(policy)


class _ChunkedSchedulePolicy(DelegatingPolicy):
    def new_session(self, context=None):
        return _ChunkedSession(self._inner.new_session(context))


class _ErrorRecoverySession(DelegatingSession):
    """Emits Recover trajectory on robot error, delegates otherwise."""

    def __init__(self, inner: Session):
        super().__init__(inner)
        self._in_error = False

    def __call__(self, obs):
        status = obs.get('robot_state.status', roboarm.RobotStatus.AVAILABLE)
        was_ok = not self._in_error
        self._in_error = status == roboarm.RobotStatus.ERROR

        if self._in_error:
            if was_ok:
                return [{'robot_command': roboarm.command.to_wire(roboarm.command.Recover())}]
            return None

        return self._inner(obs)


class ErrorRecovery(PolicyWrapper):
    """Wraps a policy to handle robot errors by emitting Recover commands.

    On error: emits a single Recover trajectory, then returns None until
    the robot recovers. On recovery: resumes normal inference.

    Composable via ``|``::

        pipeline = ErrorRecovery() | ChunkedSchedule() | codec
        wrapped = pipeline.wrap(model)
    """

    def wrap(self, policy: Policy) -> Policy:
        return _ErrorRecoveryPolicy(policy)


class _ErrorRecoveryPolicy(DelegatingPolicy):
    def new_session(self, context=None):
        return _ErrorRecoverySession(self._inner.new_session(context))


# ---------------------------------------------------------------------------
# Harness — truly stupid: directives + session call + emit
# ---------------------------------------------------------------------------


class Harness(pimm.ControlSystem):
    """Control system that manages episode lifecycle and forwards trajectories to drivers.

    The harness handles directives (RUN/STOP/FINISH/HOME) and dataset recording.
    All inference intelligence (scheduling, error recovery, blending) lives in
    the policy/session layer — the harness just calls the session and emits
    whatever comes back.
    """

    def __init__(self, policy, *, static_meta: dict[str, Any] | None = None):
        self.policy = policy
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

    def _infer(self, clock: pimm.Clock) -> list[dict[str, Any]] | None:
        """Read sensors, call session, return commands or None."""
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
        result = self._session(frozen_view(inputs))
        if result is None:
            return None
        return result if isinstance(result, list) else [result]

    def _step(self, clock: pimm.Clock) -> None:
        """Call session and emit trajectory. That's it."""
        commands = self._infer(clock)
        if commands is None:
            return

        robot_traj = [(cmd.get('timestamp', 0.0), roboarm.command.from_wire(cmd['robot_command'])) for cmd in commands]
        grip_traj = [(cmd.get('timestamp', 0.0), cmd['target_grip']) for cmd in commands if 'target_grip' in cmd]
        self.robot_commands.emit(robot_traj)
        if grip_traj:
            self.target_grip.emit(grip_traj)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
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
