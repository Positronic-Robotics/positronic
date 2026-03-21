import logging
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pimm
from positronic.dataset.ds_writer_agent import Serializers
from positronic.drivers import roboarm
from positronic.utils import flatten_dict, frozen_view


class DirectiveType(Enum):
    """Directive types for the harness."""

    RUN = 'run'
    STOP = 'stop'
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
        """Stop running the policy; devices hold position."""
        return cls(DirectiveType.STOP, None)

    @classmethod
    def HOME(cls, preset: str = 'home') -> 'Directive':
        """Stop and send devices to a named safe state."""
        return cls(DirectiveType.HOME, preset)


class Harness(pimm.ControlSystem):
    """
    Control system that manages device command authority, runs the policy, and emits
    actions according to their embedded timestamps.

    The harness is the single authority for device commands. Both the policy (during
    episodes) and the orchestrator (between episodes, via HOME) express intent through
    the harness, which translates to device-specific commands.

    Supposed to be run in foreground of a World.
    """

    def __init__(self, policy, simulate_timeout: bool = False):
        self.policy = policy
        self.context: dict[str, Any] = {}
        self.simulate_timeout = simulate_timeout

        self.frames = pimm.ReceiverDict(self)
        self.robot_state = pimm.ControlSystemReceiver(self)
        self.gripper_state = pimm.ControlSystemReceiver(self)
        self.robot_commands = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemEmitter(self)

        self.directive = pimm.ControlSystemReceiver[Directive](self, default=None, maxsize=3)

    def meta(self) -> dict[str, Any]:
        result = {'inference.simulate_timeout': self.simulate_timeout}
        for k, v in flatten_dict(self.context).items():
            result[f'inference.context.{k}'] = v
        for k, v in flatten_dict(self.policy.meta).items():
            result[f'inference.policy.{k}'] = v
        return result

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        running = False
        in_error = False
        commands_queue = deque()

        while not should_stop.value:
            directive_msg = self.directive.read()
            if directive_msg.updated:
                commands_queue.clear()
                in_error = False
                match directive_msg.data.type:
                    case DirectiveType.RUN:
                        self.context = directive_msg.data.payload or {}
                        self.policy.reset(self.context)
                        running = True
                    case DirectiveType.STOP:
                        running = False
                    case DirectiveType.HOME:
                        self.robot_commands.emit(roboarm.command.Reset())
                        self.target_grip.emit(0.0)
                        running = False
                        yield pimm.Pass()

            try:
                if not running:
                    continue

                was_ok = not in_error
                in_error = self.robot_state.value.status == roboarm.RobotStatus.ERROR
                if in_error and was_ok:
                    commands_queue.clear()
                    self.robot_commands.emit(roboarm.command.Recover())
                if in_error:
                    continue

                if not commands_queue:
                    robot_state = self.robot_state.value
                    inputs = {
                        'robot_state.q': robot_state.q,
                        'robot_state.dq': robot_state.dq,
                        'robot_state.ee_pose': Serializers.transform_3d(robot_state.ee_pose),
                        'grip': self.gripper_state.value,
                    }
                    frame_messages = {k: v.value for k, v in self.frames.items()}
                    # Extract array from NumpySMAdapter
                    images = {k: v.array for k, v in frame_messages.items()}
                    if len(images) != len(self.frames):
                        continue
                    inputs.update(images)

                    wall_start = time.monotonic()
                    inputs['__wall_time_ns__'] = time.time_ns()
                    inputs['__inference_time_ns__'] = clock.now_ns()
                    inputs.update(self.context)
                    commands = self.policy.select_action(frozen_view(inputs))
                    if not isinstance(commands, list):
                        commands = [commands]

                    if self.simulate_timeout:
                        yield pimm.Sleep(time.monotonic() - wall_start)

                    prediction_time = clock.now()
                    for cmd in commands:
                        roboarm_cmd = roboarm.command.from_wire(cmd['robot_command'])
                        target_grip = cmd['target_grip']
                        timestamp = cmd.get('timestamp', 0.0)
                        commands_queue.append((roboarm_cmd, target_grip, prediction_time + timestamp))

                if not commands_queue:
                    logging.error('Policy returned no commands, exiting harness')
                    return

                roboarm_cmd, target_grip, scheduled_time = commands_queue.popleft()
                yield pimm.Sleep(max(0.0, scheduled_time - clock.now()))

                self.robot_commands.emit(roboarm_cmd)
                self.target_grip.emit(target_grip)
            except pimm.NoValueException:
                pass
            finally:
                yield pimm.Sleep(0.01)

        self.policy.close()
