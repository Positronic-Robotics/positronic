import logging
import time
from collections import deque
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pimm
from positronic.dataset.ds_writer_agent import DsWriterCommand, Serializers
from positronic.drivers import roboarm
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


class Harness(pimm.ControlSystem):
    """
    Control system that manages device commands, episode recording, and policy execution.

    The harness is the single authority for device commands and episode lifecycle.
    It translates high-level directives into device commands and dataset recording
    commands. Both the policy (during episodes) and the orchestrator (between
    episodes) express intent through directives.

    Supposed to be run in foreground of a World.
    """

    def __init__(self, policy, *, static_meta: dict[str, Any] | None = None, simulate_timeout: bool = False):
        self.policy = policy
        self.context: dict[str, Any] = {}
        self.simulate_timeout = simulate_timeout
        self._static_meta = static_meta or {}

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
        meta['inference.simulate_timeout'] = self.simulate_timeout
        for k, v in flatten_dict(self.policy.meta).items():
            meta[f'inference.policy.{k}'] = v
        meta.update(context)
        return meta

    def _home(self):
        self.robot_commands.emit(roboarm.command.Reset())
        self.target_grip.emit(0.0)

    def _handle_directive(
        self, directive: Directive, recording: bool
    ) -> Generator[pimm.Sleep, None, tuple[bool, bool]]:
        """Handle a directive, yielding any necessary pauses. Returns (running, recording)."""
        match directive.type:
            case DirectiveType.RUN:
                if recording:
                    self.ds_command.emit(DsWriterCommand.STOP())
                    self._home()
                    yield pimm.Pass()
                self.context = directive.payload or {}
                self.policy.reset(self.context)
                self.ds_command.emit(DsWriterCommand.START(self._build_episode_meta(self.context)))
                return True, True
            case DirectiveType.STOP:
                if recording:
                    self.ds_command.emit(DsWriterCommand.SUSPEND())
                return False, recording
            case DirectiveType.FINISH:
                if recording:
                    self.ds_command.emit(DsWriterCommand.STOP(directive.payload or {}))
                    recording = False
                self._home()
                yield pimm.Pass()
                return False, recording
            case DirectiveType.HOME:
                if recording:
                    self.ds_command.emit(DsWriterCommand.ABORT())
                    recording = False
                self._home()
                yield pimm.Pass()
                return False, recording
            case _:
                raise ValueError(f'Unknown directive type: {directive.type}')

    def _infer(self, clock: pimm.Clock) -> list[dict[str, Any]] | None:
        """Read sensors and run policy inference. Returns commands or None if not ready."""
        robot_state = self.robot_state.value
        inputs = {
            'robot_state.q': robot_state.q,
            'robot_state.dq': robot_state.dq,
            'robot_state.ee_pose': Serializers.transform_3d(robot_state.ee_pose),
            'grip': self.gripper_state.value,
        }
        frame_messages = {k: v.value for k, v in self.frames.items()}
        images = {k: v.array for k, v in frame_messages.items()}
        if len(images) != len(self.frames):
            return None
        inputs.update(images)
        inputs['__wall_time_ns__'] = time.time_ns()
        inputs['__inference_time_ns__'] = clock.now_ns()
        inputs.update(self.context)
        commands = self.policy.select_action(frozen_view(inputs))
        return commands if isinstance(commands, list) else [commands]

    def _step(
        self, clock: pimm.Clock, commands_queue: deque, in_error: bool
    ) -> Generator[pimm.Sleep, None, bool | None]:
        """Execute one inference step. Returns in_error state, or None to signal exit."""
        was_ok = not in_error
        in_error = self.robot_state.value.status == roboarm.RobotStatus.ERROR
        if in_error and was_ok:
            commands_queue.clear()
            self.robot_commands.emit(roboarm.command.Recover())
        if in_error:
            return True

        if not commands_queue:
            wall_start = time.monotonic()
            commands = self._infer(clock)
            if commands is None:
                return False
            if self.simulate_timeout:
                yield pimm.Sleep(time.monotonic() - wall_start)
            prediction_time = clock.now()
            for cmd in commands:
                commands_queue.append((
                    roboarm.command.from_wire(cmd['robot_command']),
                    cmd['target_grip'],
                    prediction_time + cmd.get('timestamp', 0.0),
                ))

        if not commands_queue:
            logging.error('Policy returned no commands, exiting harness')
            return None

        roboarm_cmd, target_grip, scheduled_time = commands_queue.popleft()
        yield pimm.Sleep(max(0.0, scheduled_time - clock.now()))
        self.robot_commands.emit(roboarm_cmd)
        self.target_grip.emit(target_grip)
        return in_error

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        running = False
        recording = False
        in_error = False
        commands_queue = deque()

        while not should_stop.value:
            directive_msg = self.directive.read()
            if directive_msg.updated:
                commands_queue.clear()
                in_error = False
                running, recording = yield from self._handle_directive(directive_msg.data, recording)

            try:
                if running:
                    in_error = yield from self._step(clock, commands_queue, in_error)
                    if in_error is None:
                        return
            except pimm.NoValueException:
                pass
            finally:
                yield pimm.Sleep(0.01)

        if recording:
            self.ds_command.emit(DsWriterCommand.STOP())
        self.policy.close()
