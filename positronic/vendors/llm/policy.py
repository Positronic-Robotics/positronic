"""A stop–look–move policy that calls a model API.

A ``Conversation`` holds the policy logic, and a driver runs it: ``LLMPolicy.run`` on the rig's runtime, and
``positronic.vendors.llm.server`` once per session. A driver calls ``request``, sends the messages to the model, and
gives the reply to ``respond`` until it returns a decision. ``accept`` turns that decision into a trajectory or a
``PolicyEnd``.
"""

import io
import itertools
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Annotated, Any

import configuronic as cfn
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
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
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame, ee_frames_entry
from positronic.policy import keys as policy_keys
from positronic.policy.base import Policy, PolicyEnd, PolicyRun, Runtime, Step
from positronic.utils import flatten_dict

from .client import Endpoint
from .motion import Motion, MovePath, Waypoint


class Images(StrEnum):
    ALWAYS = 'always'
    ON_DEMAND = 'on_demand'


class Tool(StrEnum):
    MOVE_PATH = 'move_path'
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

    def state(self, target: Waypoint | None) -> dict[str, Any]:
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
        """One image prompt with the labelled frames of the selected cameras."""
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
        return ModelRequest([UserPromptPart(content)])


_SCHEMAS: dict[Tool, type[BaseModel]] = {
    Tool.MOVE_PATH: MovePath,
    Tool.REVEAL_FRAMES: RevealFrames,
    Tool.DONE: Finish,
    Tool.GIVE_UP: Finish,
}


class Conversation:
    """One episode's conversation with the model: prompts, tool replies, call budget and transcript.

    ``set_metadata`` receives the episode values.
    """

    def __init__(self, policy: 'LLMPolicy', set_metadata: Callable[[str, Any], None]):
        self._policy = policy
        self._set_metadata = set_metadata
        self._transcript: list[dict[str, Any]] = []
        self._set_metadata(policy_keys.TRANSCRIPT, self._transcript)
        self._messages: list[ModelMessage] = []
        self._frames: list[list[int]] = []  # per observation, the indices of its frame messages
        self._anchor: int | None = None  # the newest frame message that holds stubs
        self._obs: _Observation | None = None
        self._pictures: list[str] = []
        self._revealed: set[str] = set()
        self._failures = 0
        self._calls = 0
        self._target: Waypoint | None = None

    def _record(self, event: dict[str, Any]) -> None:
        self._transcript.append(event)
        self._set_metadata(policy_keys.TRANSCRIPT, self._transcript)

    def _initial(self) -> ModelRequest:
        prompt = (
            'Control one robot arm through exactly one tool call per response. '
            'Use absolute hand targets in the coordinate frame of the measured pose. '
            'Positions are metres; roll/pitch/yaw are radians with R=Rz(yaw)Ry(pitch)Rx(roll). '
            'The hand frame is the same frame as the measured hand pose. Gripper 0 is open and 1 is closed. '
            'A path plays its waypoints in order before the next observation; '
            'actual arrival must be checked from the measured state. '
            'Oversized segments are clamped from the previous clamped waypoint, and the first from the latest '
            'measured pose: translation keeps its direction and rotation follows the shortest turn, each limited '
            'independently. A waypoint sets its gripper at the start of its segment. When a waypoint changes the '
            'gripper, the hand holds still at it for grip_hold seconds. '
            'The tool result reports the clamped waypoints; check them before planning the next path. '
            'Put several waypoints in one path when the next motions do not depend on what you will see. '
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
        self._record({
            'event': 'instructions',
            'prompt': prompt,
            'tools': {tool.name: tool.parameters_json_schema for tool in self._policy.tools},
        })
        return ModelRequest([SystemPromptPart(prompt)])

    def _observe(self, raw: Mapping[str, Any], time_ns: int) -> None:
        self._obs = _Observation.read(raw, self._policy.camera_keys, time_ns)
        if not self._messages:
            self._messages.append(self._initial())
        state = self._obs.state(self._target)
        self._record({'event': 'observation', **state})
        self._messages.append(ModelRequest.user_text_prompt(json.dumps(state, allow_nan=False)))
        self._frames.append([])
        if len(self._frames) > self._policy.image_horizon:
            self._stub(self._frames[-self._policy.image_horizon - 1])
        self._pictures = list(self._obs.images) if self._policy.images is Images.ALWAYS else []
        self._revealed = set(self._pictures)
        self._failures = 0

    def _stub(self, indices: list[int]) -> None:
        for index in indices:
            message = self._messages[index]
            assert isinstance(message, ModelRequest)
            (part,) = message.parts
            assert isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            content = [
                '[older camera frame omitted]' if isinstance(item, BinaryContent) else item for item in part.content
            ]
            self._messages[index] = replace(message, parts=[replace(part, content=content)])
            self._anchor = index

    def _select_cameras(self, requested: list[str]) -> list[str]:
        assert self._obs is not None
        cameras = requested if requested else list(self._obs.images)
        selected = set(cameras)
        if len(selected) != len(cameras) or selected - self._obs.images.keys() or selected & self._revealed:
            raise ValueError('Choose available cameras not already revealed in this observation')
        return cameras

    def _tool(self, call: ToolCallPart) -> MovePath | Finish | None:
        tool = Tool(call.tool_name)
        data = _SCHEMAS[tool].model_validate_json(call.args_as_json_str())
        match data:
            case MovePath() if len(data.waypoints) > self._policy.motion.max_waypoints:
                raise ValueError(f'A path has at most {self._policy.motion.max_waypoints} waypoints')
            case MovePath() | Finish():
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
        parts = [ToolReturnPart(call.tool_name, f'Rejected: {error}', tool_call_id=call.tool_call_id) for call in calls]
        self._messages.append(
            ModelRequest(parts if parts else [UserPromptPart(f'Rejected: {error}. Make one tool call.')])
        )

    def respond(self, response: ModelResponse) -> tuple[ToolCallPart, MovePath | Finish] | None:
        """The decision in ``response``, or ``None`` when the model must be asked again about the same observation."""
        self._record({
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
            self._record({'event': 'rejected', 'call': self._calls, 'reason': str(exc)})
            self._failures += 1
            if self._failures >= self._policy.max_invalid:
                raise RuntimeError(f'Model produced {self._failures} consecutive invalid replies') from exc
            return None
        self._failures = 0
        return (calls[0], data) if data is not None else None

    @staticmethod
    def _ends_with_prompt(message: ModelMessage) -> bool:
        return isinstance(message, ModelRequest) and isinstance(message.parts[-1], UserPromptPart)

    @staticmethod
    def _with_cache_point(message: ModelMessage) -> ModelMessage:
        assert isinstance(message, ModelRequest)
        *parts, part = message.parts
        assert isinstance(part, UserPromptPart)
        content = [part.content] if isinstance(part.content, str) else part.content
        return replace(message, parts=[*parts, replace(part, content=[*content, CachePoint()])])

    @property
    def budget_spent(self) -> bool:
        return self._calls >= self._policy.max_calls

    def request(self, obs: Mapping[str, Any], time_ns: int) -> list[ModelMessage] | PolicyEnd:
        """The messages of the next model call, or ``PolicyEnd.GIVE_UP`` once the call budget is spent.

        ``obs`` starts a new decision; a decision in progress keeps the observation it started with.
        """
        if self.budget_spent:
            self._set_metadata(policy_keys.STOP_REASON, 'call_budget')
            self._record({'event': 'budget_exhausted', 'calls': self._calls})
            self._record({
                'event': 'accepted',
                'call': self._calls,
                'time_ns': time_ns,
                policy_keys.STOP_REASON: 'call_budget',
            })
            return PolicyEnd.GIVE_UP
        if self._obs is None:
            self._observe(obs, time_ns)
        assert self._obs is not None
        if self._pictures:
            self._frames[-1].append(len(self._messages))
            self._messages.append(self._obs.frames(self._pictures, self._policy.image_size))
            self._pictures = []
        self._calls += 1
        self._record({
            'event': 'request',
            'call': self._calls,
            'cameras': sorted(self._revealed),
            policy_keys.OBS_TIME_NS: self._obs.time_ns,
        })
        # A provider caches a prompt only up to a cache point or to the end of a request. The stored history
        # changes only where a frame message ages out, so the newest stubbed one marks the stable prefix.
        # Only a user prompt can hold a cache point: OpenAI and Bedrock map a prompt that holds only a cache point
        # to an empty message or an error. So when a request ends in tool results, the newest prompt holds it.
        # Cache points go on the outgoing copy only: stored ones would pass the provider's limit.
        messages = list(self._messages)
        if self._anchor is not None:
            messages[self._anchor] = self._with_cache_point(messages[self._anchor])
        end = next(i for i in reversed(range(len(messages))) if self._ends_with_prompt(messages[i]))
        messages[end] = self._with_cache_point(messages[end])
        return messages

    def accept(
        self, call: ToolCallPart, data: MovePath | Finish, obs: Mapping[str, Any], time_ns: int
    ) -> list[list[dict]] | PolicyEnd:
        """The commands of each segment and gripper hold of a path from the pose in ``obs``, or the ``PolicyEnd`` of
        a finish."""
        self._obs = None
        if isinstance(data, Finish):
            self._messages.append(
                ModelRequest([ToolReturnPart(call.tool_name, 'No further actions.', tool_call_id=call.tool_call_id)])
            )
            self._record({
                'event': 'accepted',
                'call': self._calls,
                'time_ns': time_ns,
                policy_keys.STOP_REASON: call.tool_name,
            })
            self._set_metadata(policy_keys.STOP_REASON, call.tool_name)
            self._set_metadata(policy_keys.HINDSIGHT, data.hindsight)
            return PolicyEnd.COMPLETE if call.tool_name == Tool.DONE else PolicyEnd.GIVE_UP
        start = _Observation.measured_pose(obs)
        grip = float(obs[keys.GRIP]) if self._target is None else self._target.gripper
        waypoints, parts = self._policy.motion.trajectory(start, grip, data.waypoints)
        self._target = waypoints[-1]
        result = {
            'waypoints': [waypoint.model_dump() for waypoint in waypoints],
            'clamped': waypoints != data.waypoints,
            'duration_s': (sum(map(len, parts)) + 1) / self._policy.motion.fps,
            'feedback': 'Path scheduled. Check the next measured observation for actual arrival.',
        }
        self._messages.append(ModelRequest([ToolReturnPart(call.tool_name, result, tool_call_id=call.tool_call_id)]))
        self._record({'event': 'accepted', 'call': self._calls, 'time_ns': time_ns, **result})
        return parts


class LLMPolicy(Policy):
    """An API-backed, single-arm Cartesian policy with independent episode conversations."""

    def __init__(
        self,
        endpoint: Endpoint,
        motion: Motion,
        *,
        images: Images = Images.ALWAYS,
        camera_keys: tuple[str, ...] = (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE),
        image_size: int = 640,
        image_horizon: int = 3,
        max_calls: int = 100,
        max_invalid: int = 3,
    ):
        if min(image_size, image_horizon, max_calls, max_invalid) < 1:
            raise ValueError('Image size, image horizon, call budget and invalid-reply limit must be positive')
        if not camera_keys or len(set(camera_keys)) != len(camera_keys):
            raise ValueError('camera_keys must name at least one camera, without duplicates')
        self.endpoint, self.motion, self.images = endpoint, motion, images
        self.camera_keys, self.image_size, self.image_horizon = camera_keys, image_size, image_horizon
        self.max_calls, self.max_invalid = max_calls, max_invalid
        self.tools = [
            ToolDefinition(
                name=tool.value, description=schema.__doc__ or '', parameters_json_schema=schema.model_json_schema()
            )
            for tool, schema in _SCHEMAS.items()
            if tool is not Tool.REVEAL_FRAMES or images is Images.ON_DEMAND
        ]

    def meta(self) -> dict[str, Any]:
        # The model reads the measured pose and moves the hand in the frame the rig reports it in.
        frames = ee_frames_entry(dict.fromkeys((keys.EE_POSE, keys.TARGET_EE_POSE), EEFrame(site=GRIPPER_SITE)))
        return frames | flatten_dict({
            policy_keys.TYPE: policy_keys.LLM_TYPE,
            policy_keys.MODEL: self.endpoint.model_id,
            'settings': dict(self.endpoint.settings),
            'cache_breakpoints': self.endpoint.cache_breakpoints,
            'motion': asdict(self.motion),
            'max_calls': self.max_calls,
            'max_invalid': self.max_invalid,
            'timeout': self.endpoint.timeout,
            'images': self.images.value,
            'camera_keys': list(self.camera_keys),
            'image_size': self.image_size,
            'image_horizon': self.image_horizon,
        })

    def run(self, runtime: Runtime) -> PolicyRun:
        with self.endpoint.open() as client:
            if client.api_key_fingerprint is not None:
                runtime.set_metadata(policy_keys.API_KEY_FINGERPRINT, client.api_key_fingerprint)
            conversation = Conversation(self, runtime.set_metadata)
            obs = yield
            while True:
                messages = conversation.request(obs, runtime.time_ns)
                if isinstance(messages, PolicyEnd):
                    return messages
                answer = runtime.submit(client.request, messages, self.tools)
                obs = yield Step({}, runtime.time_ns)
                while not answer.done():
                    obs = yield Step({}, runtime.time_ns)
                decision = conversation.respond(answer.result())
                if decision is None:
                    continue
                parts = conversation.accept(*decision, obs, runtime.time_ns)
                if isinstance(parts, PolicyEnd):
                    return parts
                chunk = [command for part in parts for command in part]
                # A late step combines overdue commands, but never past the end of a part: each waypoint and
                # each gripper hold reaches the robot.
                part_ends = set(itertools.accumulate(map(len, parts[:-1])))
                started_ns, index = runtime.time_ns, 0
                end_ns = started_ns + round((len(chunk) + 1) / self.motion.fps * 1e9)
                while index < len(chunk) or runtime.time_ns < end_ns:
                    due_ns = started_ns + round((index + 1) / self.motion.fps * 1e9)
                    commands = {}
                    while index < len(chunk) and runtime.time_ns >= due_ns:
                        commands.update(chunk[index])
                        index += 1
                        if index in part_ends:
                            # A late part end delays the rest of the path, so each later part keeps its duration.
                            started_ns += runtime.time_ns - due_ns
                            end_ns += runtime.time_ns - due_ns
                            due_ns = started_ns + round((index + 1) / self.motion.fps * 1e9)
                            break
                        due_ns = started_ns + round((index + 1) / self.motion.fps * 1e9)
                    obs = yield Step(commands, due_ns)


@cfn.config(
    timeout=120.0,
    settings={},
    motion=cfn.Config(Motion),
    images=Images.ALWAYS.value,
    camera_keys=(keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE),
    image_size=640,
    image_horizon=3,
    max_calls=100,
    max_invalid=3,
    cache_breakpoints=False,
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
    cache_breakpoints: bool,
) -> LLMPolicy:
    """The LLM policy over a direct model API."""
    return LLMPolicy(
        Endpoint(model, timeout=timeout, settings=settings, cache_breakpoints=cache_breakpoints),
        motion,
        images=Images(images),
        camera_keys=tuple(camera_keys),
        image_size=image_size,
        image_horizon=image_horizon,
        max_calls=max_calls,
        max_invalid=max_invalid,
    )
