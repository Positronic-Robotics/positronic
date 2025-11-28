import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pimm
from positronic.dataset.ds_writer_agent import Serializers
from positronic.drivers import roboarm
from positronic.policy.action import ActionDecoder
from positronic.policy.observation import ObservationEncoder
from positronic.utils import flatten_dict


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
    def __init__(
        self,
        observation_encoder: ObservationEncoder,
        action_decoder: ActionDecoder,
        policy,
        inference_fps: int = 30,
        simulate_timeout: bool = False,
    ):
        self.observation_encoder = observation_encoder
        self.action_decoder = action_decoder
        self.policy = policy
        self.inference_fps = inference_fps
        self.context: dict[str, Any] = {}
        self.simulate_timeout = simulate_timeout

        self.frames = pimm.ReceiverDict(self)
        self.robot_state = pimm.ControlSystemReceiver(self)
        self.gripper_state = pimm.ControlSystemReceiver(self)
        self.robot_commands = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemEmitter(self)

        self.command = pimm.ControlSystemReceiver[InferenceCommand](self, default=None, maxsize=3)

    def meta(self) -> dict[str, Any]:
        result = {'inference.policy_fps': self.inference_fps, 'inference.simulate_timeout': self.simulate_timeout}
        for k, v in flatten_dict(self.context).items():
            result[f'inference.context.{k}'] = v
        for k, v in flatten_dict(self.observation_encoder.meta).items():
            result[f'inference.observation.{k}'] = v
        for k, v in flatten_dict(self.action_decoder.meta).items():
            result[f'inference.action.{k}'] = v
        for k, v in flatten_dict(self.policy.meta).items():
            result[f'inference.policy.{k}'] = v
        return result

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        running = False

        # TODO: We should emit new commands per frame, not per inference fps
        rate_limiter = pimm.RateLimiter(clock, hz=self.inference_fps)

        while not should_stop.value:
            command_msg = self.command.read()
            if command_msg.updated:
                match command_msg.data.type:
                    case InferenceCommandType.START:
                        running = True
                        self.context = command_msg.data.payload or {}
                        rate_limiter.reset()
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

                obs = self.observation_encoder.encode(inputs)
                obs.update(self.context)

                start = time.monotonic()
                action = self.policy.select_action(obs)
                roboarm_command, target_grip = self.action_decoder.decode(action, inputs)

                duration = time.monotonic() - start
                if self.simulate_timeout:
                    yield pimm.Sleep(duration)

                self.robot_commands.emit(roboarm_command)
                self.target_grip.emit(target_grip)
            except pimm.NoValueException:
                continue
            finally:
                yield pimm.Sleep(rate_limiter.wait_time())
