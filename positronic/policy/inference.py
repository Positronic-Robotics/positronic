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


class InferenceCommandType(Enum):
    """Commands for the inference."""

    START = 'start'
    STOP = 'stop'
    RESET = 'reset'


@dataclass
class InferenceCommand:
    """Command for the inference."""

    type: InferenceCommandType
    payload: Any | None = None

    @classmethod
    def START(cls, **kwargs) -> 'InferenceCommand':
        """Convenience method for creating a START command."""
        return cls(InferenceCommandType.START, kwargs)

    @classmethod
    def STOP(cls) -> 'InferenceCommand':
        """Convenience method for creating a STOP command."""
        return cls(InferenceCommandType.STOP, None)

    @classmethod
    def RESET(cls) -> 'InferenceCommand':
        """Convenience method for creating a RESET command."""
        return cls(InferenceCommandType.RESET, None)


class Inference(pimm.ControlSystem):
    """
    Control system that handles start/stop/reset commands, runs the policy, and emits
    actions according to their embedded timestamps.

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

        self.command = pimm.ControlSystemReceiver[InferenceCommand](self, default=None, maxsize=3)

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
            command_msg = self.command.read()
            if command_msg.updated:
                commands_queue.clear()
                in_error = False
                match command_msg.data.type:
                    case InferenceCommandType.START:
                        running = True
                        self.context = command_msg.data.payload or {}
                    case InferenceCommandType.STOP:
                        running = False
                    case InferenceCommandType.RESET:
                        self.robot_commands.emit(roboarm.command.Reset())
                        self.target_grip.emit(0.0)
                        self.policy.reset()
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
                    logging.error('Policy returned no commands, exiting inference')
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
