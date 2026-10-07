"""A stop–look–move policy whose model calls run on the rig's worker runtime."""

import io
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Annotated, Any

import configuronic as cfn
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from positronic import geom, keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import Policy, PolicyRun, Runtime, Step
from positronic.policy.processors import PauseOnUnavailable
from positronic.policy.sequential import Sequential

from .client import Endpoint
from .motion import Motion, MoveTo


class Images(StrEnum):
    ALWAYS = 'always'
    ON_DEMAND = 'on_demand'


class Tool(StrEnum):
    MOVE_TO = 'move_to'
    REVEAL_FRAMES = 'reveal_frames'
    DONE = 'done'
    GIVE_UP = 'give_up'


class Finish(BaseModel):
    """Stop issuing actions for this episode, giving a reason and hindsight for the recorded transcript."""

    model_config = ConfigDict(extra='forbid', strict=True)
    reason: Annotated[str, Field(min_length=1)]
    hindsight: Annotated[str, Field(min_length=1)]


class RevealFrames(BaseModel):
    """Reveal selected camera frames from this observation. An empty camera list selects every camera."""

    model_config = ConfigDict(extra='forbid', strict=True)
    cameras: list[str] = Field(default_factory=list)
    note: Annotated[str, Field(min_length=1)]


@dataclass
class _Observation:
    pose: geom.Transform3D
    grip: float
    task: str
    time_ns: int
    images: dict[str, np.ndarray]

    @staticmethod
    def measured_pose(obs: Mapping[str, Any]) -> geom.Transform3D:
        pose = np.asarray(obs[keys.EE_POSE], dtype=float)
        if (
            pose.shape != (7,)
            or not np.all(np.isfinite(pose))
            or not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-3)
        ):
            raise ValueError('Measured hand pose must contain xyz and a finite unit quaternion in wxyz order')
        return geom.Transform3D.from_vector(pose, geom.Rotation.Representation.QUAT)

    @classmethod
    def read(cls, obs: Mapping[str, Any], camera_keys: tuple[str, ...], time_ns: int) -> '_Observation':
        pose = cls.measured_pose(obs)
        grip = float(obs[keys.GRIP])
        if not np.isfinite(grip) or not 0 <= grip <= 1:
            raise ValueError('Measured gripper position must be in [0, 1]')
        images = {}
        for name in camera_keys:
            frame = np.asarray(obs[name])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f'{name} must be an RGB uint8 image')
            images[name] = frame
        return cls(pose, grip, str(obs[keys.TASK]), time_ns, images)

    def state(self, target: MoveTo | None) -> dict[str, Any]:
        state = {
            'task': self.task,
            policy_keys.OBS_TIME_NS: self.time_ns,
            'position_m': self.pose.translation.tolist(),
            'roll_pitch_yaw_rad': self.pose.rotation.as_euler.tolist(),
            'gripper': self.grip,
            'cameras': list(self.images),
        }
        if target is not None:
            state['remaining_translation_m'] = (target.pose.translation - self.pose.translation).tolist()
            state['remaining_gripper'] = target.gripper - self.grip
            state['previous_target'] = target.model_dump()
        return state

    def frames(self, cameras: list[str], image_size: int) -> ModelRequest:
        """One image prompt, tagged by observation until its PNG payloads are pruned."""
        content: list[str | BinaryContent] = []
        for camera in cameras:
            image = Image.fromarray(self.images[camera])
            image.thumbnail((image_size, image_size))
            data = io.BytesIO()
            image.save(data, format='PNG')
            content.extend([
                f'Camera {camera}, observation {self.time_ns} ns:',
                BinaryContent(data.getvalue(), media_type='image/png'),
            ])
        return ModelRequest([UserPromptPart(content)], metadata={policy_keys.OBS_TIME_NS: self.time_ns})


_SCHEMAS: dict[Tool, type[BaseModel]] = {
    Tool.MOVE_TO: MoveTo,
    Tool.REVEAL_FRAMES: RevealFrames,
    Tool.DONE: Finish,
    Tool.GIVE_UP: Finish,
}


class LLMPolicy(Policy):
    """An API-backed, single-arm Cartesian policy with independent episode conversations."""

    class _Conversation:
        def __init__(self, policy: 'LLMPolicy'):
            self._policy = policy
            self._transcript: list[dict[str, Any]] = []
            self._messages: list[ModelMessage] = []
            self._obs: _Observation | None = None
            self._pictures: list[str] = []
            self._revealed: set[str] = set()
            self._failures = 0
            self._calls = 0
            self._target: MoveTo | None = None

        def _initial(self) -> ModelRequest:
            prompt = (
                'Control one robot arm through exactly one tool call per response. '
                'Use absolute hand targets in the coordinate frame of the measured pose. '
                'Positions are metres; roll/pitch/yaw are radians with R=Rz(yaw)Ry(pitch)Rx(roll). '
                'The hand frame is the same frame as the measured hand pose. Gripper 0 is open and 1 is closed. '
                'Moves play before the next observation; actual arrival must be checked from the measured state. '
                'Oversized moves are clamped from the latest measured pose: translation keeps its direction and '
                'rotation follows the shortest turn, each limited independently. '
                'The tool result reports the clamped target; check it before planning the next move. '
                'Camera frames remain fixed during a decision. '
                f'{Tool.REVEAL_FRAMES.value} only reveals a frame; it does not move a camera. '
                'A note should briefly describe what you see and why you chose the motion. '
                'Use done when you believe the task is complete, or give_up when you cannot continue. '
                'Both stop further actions and model calls; '
                'the episode continues until external completion or timeout. '
                'Both require hindsight for inspection; no advice is carried into another episode. '
                f'Motion limits: {json.dumps(asdict(self._policy.motion))}. '
                f'API call budget, including corrections and pictures: {self._policy.max_calls}.'
            )
            self._transcript.append({
                'event': 'instructions',
                'prompt': prompt,
                'tools': {tool.name: tool.parameters_json_schema for tool in self._policy._tools},
            })
            return ModelRequest([SystemPromptPart(prompt)])

        def _observe(self, raw: Mapping[str, Any], time_ns: int) -> None:
            self._obs = _Observation.read(raw, self._policy.camera_keys, time_ns)
            if not self._messages:
                self._messages.append(self._initial())
            state = self._obs.state(self._target)
            self._transcript.append({'event': 'observation', **state})
            self._messages.append(ModelRequest.user_text_prompt(json.dumps(state, allow_nan=False)))
            self._pictures = list(self._obs.images) if self._policy.images is Images.ALWAYS else []
            self._revealed = set(self._pictures)
            self._failures = 0

        def _select_cameras(self, requested: list[str]) -> list[str]:
            assert self._obs is not None
            cameras = requested if requested else list(self._obs.images)
            selected = set(cameras)
            if len(selected) != len(cameras) or selected - self._obs.images.keys() or selected & self._revealed:
                raise ValueError('Choose available cameras not already revealed in this observation')
            return cameras

        def _tool(self, call: ToolCallPart) -> MoveTo | Finish | None:
            tool = Tool(call.tool_name)
            data = _SCHEMAS[tool].model_validate_json(call.args_as_json_str())
            match data:
                case MoveTo() | Finish():
                    return data
                case RevealFrames():
                    if self._policy.images is not Images.ON_DEMAND:
                        raise ValueError(f'{tool.value} is available only with images=on_demand')
                    cameras = self._select_cameras(data.cameras)
                    self._revealed.update(cameras)
                    self._pictures = cameras
                    self._messages.append(
                        ModelRequest([ToolReturnPart(tool.value, 'Frames attached.', tool_call_id=call.tool_call_id)])
                    )
                    return None
            raise ValueError(f'Unsupported tool: {tool}')

        def _reject(self, calls: list[ToolCallPart], error: str) -> None:
            parts = [
                ToolReturnPart(call.tool_name, f'Rejected: {error}', tool_call_id=call.tool_call_id) for call in calls
            ]
            self._messages.append(
                ModelRequest(parts if parts else [UserPromptPart(f'Rejected: {error}. Make one tool call.')])
            )

        def _respond(self, response: ModelResponse) -> tuple[ToolCallPart, MoveTo | Finish] | None:
            self._transcript.append({
                'event': 'response',
                'call': self._calls,
                'model': response.model_name,
                'finish_reason': response.finish_reason,
                'usage': asdict(response.usage),
                'tools': [
                    {'name': part.tool_name, 'arguments': part.args, 'id': part.tool_call_id}
                    for part in response.parts
                    if isinstance(part, ToolCallPart)
                ],
                'text': [part.content for part in response.parts if isinstance(part, TextPart)],
            })
            self._messages.append(response)
            calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
            try:
                if len(calls) != 1 or response.finish_reason in ('length', 'content_filter', 'error'):
                    raise ValueError('Expected exactly one complete tool call')
                data = self._tool(calls[0])
            except ValueError as exc:
                self._reject(calls, str(exc))
                self._transcript.append({'event': 'rejected', 'call': self._calls, 'reason': str(exc)})
                self._failures += 1
                if self._failures >= self._policy.max_invalid:
                    raise RuntimeError(f'Model produced {self._failures} consecutive invalid replies') from exc
                return None
            self._failures = 0
            return (calls[0], data) if data is not None else None

        def _prune_images(self) -> None:
            observations = list(
                dict.fromkeys(
                    message.metadata[policy_keys.OBS_TIME_NS]
                    for message in self._messages
                    if isinstance(message, ModelRequest) and message.metadata is not None
                )
            )
            retained = set(observations[-self._policy.image_horizon :])
            for index, message in enumerate(self._messages):
                if (
                    not isinstance(message, ModelRequest)
                    or message.metadata is None
                    or message.metadata[policy_keys.OBS_TIME_NS] in retained
                ):
                    continue
                (part,) = message.parts
                assert isinstance(part, UserPromptPart) and not isinstance(part.content, str)
                content = [
                    '[older camera frame omitted]' if isinstance(item, BinaryContent) else item for item in part.content
                ]
                self._messages[index] = replace(message, parts=[replace(part, content=content)], metadata=None)

        def _request_messages(self) -> list[ModelMessage]:
            assert self._obs is not None
            if self._pictures:
                self._messages.append(self._obs.frames(self._pictures, self._policy.image_size))
                self._pictures = []
                self._prune_images()
            self._calls += 1
            self._transcript.append({
                'event': 'request',
                'call': self._calls,
                'cameras': sorted(self._revealed),
                policy_keys.OBS_TIME_NS: self._obs.time_ns,
            })
            return list(self._messages)

        def _accept(
            self, call: ToolCallPart, data: MoveTo | Finish, obs: Mapping[str, Any], time_ns: int
        ) -> list[dict]:
            self._obs = None
            if isinstance(data, Finish):
                self._messages.append(
                    ModelRequest([
                        ToolReturnPart(call.tool_name, 'No further actions.', tool_call_id=call.tool_call_id)
                    ])
                )
                self._transcript.append({
                    'event': 'accepted',
                    'call': self._calls,
                    'time_ns': time_ns,
                    'stop_reason': call.tool_name,
                })
                return []
            start = _Observation.measured_pose(obs)
            self._target, trajectory = self._policy.motion.trajectory(start, data)
            result = {
                'target': self._target.model_dump(),
                'clamped': self._target != data,
                'duration_s': (len(trajectory) + 1) / self._policy.motion.fps,
                'feedback': 'Target scheduled. Check the next measured observation for actual arrival.',
            }
            self._messages.append(
                ModelRequest([ToolReturnPart(call.tool_name, result, tool_call_id=call.tool_call_id)])
            )
            self._transcript.append({'event': 'accepted', 'call': self._calls, 'time_ns': time_ns, **result})
            return trajectory

    def __init__(
        self,
        endpoint: Endpoint,
        motion: Motion,
        *,
        images: Images = Images.ALWAYS,
        camera_keys: tuple[str, ...] = (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE),
        image_size: int = 640,
        image_horizon: int = 2,
        max_calls: int = 100,
        max_invalid: int = 3,
    ):
        if min(image_size, image_horizon, max_calls, max_invalid) < 1:
            raise ValueError('Image size, history horizon, call budget and invalid-reply limit must be positive')
        if not camera_keys or len(set(camera_keys)) != len(camera_keys):
            raise ValueError('camera_keys must name at least one camera, without duplicates')
        self.endpoint, self.motion, self.images = endpoint, motion, images
        self.camera_keys, self.image_size, self.image_horizon = camera_keys, image_size, image_horizon
        self.max_calls, self.max_invalid = max_calls, max_invalid
        self._tools = [
            ToolDefinition(
                name=tool.value, description=schema.__doc__ or '', parameters_json_schema=schema.model_json_schema()
            )
            for tool, schema in _SCHEMAS.items()
            if tool is not Tool.REVEAL_FRAMES or images is Images.ON_DEMAND
        ]

    def meta(self) -> dict[str, Any]:
        return {
            policy_keys.TYPE: 'llm',
            'model': self.endpoint.model.model_id,
            'settings': dict(self.endpoint.settings),
            'motion': asdict(self.motion),
            'max_calls': self.max_calls,
            'max_invalid': self.max_invalid,
            'timeout': self.endpoint.timeout,
            'images': self.images.value,
            'camera_keys': list(self.camera_keys),
            'image_size': self.image_size,
            'image_horizon': self.image_horizon,
        }

    def run(self, runtime: Runtime) -> PolicyRun:
        conversation = self._Conversation(self)
        runtime.metadata['transcript'] = conversation._transcript
        obs = yield
        while True:
            if conversation._calls >= self.max_calls:
                runtime.metadata['stop_reason'] = 'call_budget'
                conversation._transcript.extend([
                    {'event': 'budget_exhausted', 'calls': conversation._calls},
                    {
                        'event': 'accepted',
                        'call': conversation._calls,
                        'time_ns': runtime.time_ns,
                        'stop_reason': 'call_budget',
                    },
                ])
                break
            if conversation._obs is None:
                conversation._observe(obs, runtime.time_ns)
            messages = conversation._request_messages()
            answer = runtime.submit(self.endpoint.request, messages, self._tools)
            obs = yield Step({}, runtime.time_ns)
            while not answer.done():
                obs = yield Step({}, runtime.time_ns)
            decision = conversation._respond(answer.result())
            if decision is None:
                continue
            call, data = decision
            chunk = conversation._accept(call, data, obs, runtime.time_ns)
            if isinstance(data, Finish):
                runtime.metadata.update(stop_reason=call.tool_name, hindsight=data.hindsight)
                break
            started_ns, index = runtime.time_ns, 0
            end_ns = started_ns + round((len(chunk) + 1) / self.motion.fps * 1e9)
            while index < len(chunk) or runtime.time_ns < end_ns:
                due_ns = started_ns + round((index + 1) / self.motion.fps * 1e9)
                commands = {}
                while index < len(chunk) and runtime.time_ns >= due_ns:
                    commands.update(chunk[index])
                    index += 1
                    due_ns = started_ns + round((index + 1) / self.motion.fps * 1e9)
                obs = yield Step(commands, due_ns)
        while True:
            yield Step({}, runtime.time_ns + 1_000_000_000)


@cfn.config(
    timeout=120.0,
    settings={},
    motion=cfn.Config(Motion),
    images=Images.ALWAYS.value,
    camera_keys=(keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE),
    image_size=640,
    image_horizon=2,
    max_calls=100,
    max_invalid=3,
)
def llm(
    model: str | Model,
    timeout: float,
    settings: ModelSettings,
    motion: Motion,
    images: str,
    camera_keys: tuple[str, ...],
    image_size: int,
    image_horizon: int,
    max_calls: int,
    max_invalid: int,
) -> Policy:
    """Direct API policy, with fault handling and trajectory scheduling on the rig."""
    policy = LLMPolicy(
        Endpoint(model, timeout=timeout, settings=settings),
        motion,
        images=Images(images),
        camera_keys=tuple(camera_keys),
        image_size=image_size,
        image_horizon=image_horizon,
        max_calls=max_calls,
        max_invalid=max_invalid,
    )
    return Sequential(PauseOnUnavailable(), policy)
