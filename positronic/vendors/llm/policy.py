"""A stop–look–move policy whose model calls run on the rig's worker runtime."""

import io
import json
import threading
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
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from positronic import geom, keys, telemetry, telemetry_keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import Answer, Policy, Runtime, Session
from positronic.policy.layers import ChunkedSchedule, StopOnFault

from .client import API, Endpoint
from .motion import Motion, MoveTo
from .recording import Transcript


class Images(StrEnum):
    ALWAYS = 'always'
    ON_DEMAND = 'on_demand'


class Tool(StrEnum):
    MOVE_TO = 'move_to'
    TAKE_PIC = 'take_pic'
    DONE = 'done'
    GIVE_UP = 'give_up'


class Finish(BaseModel):
    """End this attempt, giving a reason and hindsight for the recorded transcript."""

    model_config = ConfigDict(extra='forbid', strict=True)
    reason: Annotated[str, Field(min_length=1)]
    hindsight: Annotated[str, Field(min_length=1)]


class TakePic(BaseModel):
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
    images: dict[str, BinaryContent]

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
    def read(cls, obs: Mapping[str, Any], camera_keys: tuple[str, ...], image_size: int) -> '_Observation':
        pose = cls.measured_pose(obs)
        grip = float(obs[keys.GRIP])
        if not np.isfinite(grip) or not 0 <= grip <= 1:
            raise ValueError('Measured gripper position must be in [0, 1]')
        images = {}
        for name in camera_keys:
            frame = np.asarray(obs[name])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f'{name} must be an RGB uint8 image')
            image = Image.fromarray(frame)
            image.thumbnail((image_size, image_size))
            data = io.BytesIO()
            image.save(data, format='PNG')
            images[name] = BinaryContent(data.getvalue(), media_type='image/png')
        return cls(pose, grip, str(obs[keys.TASK]), int(obs[keys.OBS_TIME_NS]), images)

    def state(self, target: MoveTo | None) -> dict[str, Any]:
        state = {
            'task': self.task,
            'obs_time_ns': self.time_ns,
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

    def frames(self, cameras: list[str]) -> ModelRequest:
        content: list[str | BinaryContent] = []
        for camera in cameras:
            content.extend([f'Camera {camera}, observation {self.time_ns} ns:', self.images[camera]])
        return ModelRequest([UserPromptPart(content)], metadata={keys.OBS_TIME_NS: self.time_ns})


@dataclass
class _Decision:
    messages: list[ModelMessage]
    target: MoveTo | None = None
    stop_reason: str | None = None
    hindsight: str | None = None


_SCHEMAS: dict[Tool, type[BaseModel]] = {
    Tool.MOVE_TO: MoveTo,
    Tool.TAKE_PIC: TakePic,
    Tool.DONE: Finish,
    Tool.GIVE_UP: Finish,
}


class _Conversation:
    """Worker-owned call budget. Only accepted decisions commit their messages to the session."""

    def __init__(self, policy: 'LLMPolicy', transcript: Transcript):
        self.policy = policy
        self.transcript = transcript
        self.calls = 0
        self.tools = [
            ToolDefinition(
                name=tool.value, description=schema.__doc__ or '', parameters_json_schema=schema.model_json_schema()
            )
            for tool, schema in _SCHEMAS.items()
            if tool is not Tool.TAKE_PIC or policy.images is Images.ON_DEMAND
        ]

    @staticmethod
    def _has_images(message: ModelRequest) -> bool:
        return any(
            isinstance(part, UserPromptPart)
            and not isinstance(part.content, str)
            and any(isinstance(item, BinaryContent) for item in part.content)
            for part in message.parts
        )

    def _outgoing(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        observations = list(
            dict.fromkeys(
                message.metadata[keys.OBS_TIME_NS]
                for message in messages
                if isinstance(message, ModelRequest) and message.metadata is not None and self._has_images(message)
            )
        )
        retained = set(observations[-self.policy.image_horizon :])
        outgoing = []
        for message in messages:
            if (
                not isinstance(message, ModelRequest)
                or message.metadata is None
                or message.metadata[keys.OBS_TIME_NS] in retained
            ):
                outgoing.append(message)
                continue
            parts = [
                replace(
                    part,
                    content=[
                        '[older camera frame omitted]' if isinstance(item, BinaryContent) else item
                        for item in part.content
                    ],
                )
                if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
                else part
                for part in message.parts
            ]
            outgoing.append(replace(message, parts=parts))
        return outgoing

    def _initial(self) -> ModelRequest:
        prompt = (
            'Control one robot arm through exactly one tool call per response. '
            'Use absolute hand targets in the coordinate frame of the measured pose. '
            'Positions are metres; roll/pitch/yaw are radians with R=Rz(yaw)Ry(pitch)Rx(roll). '
            'The hand frame is the same frame as the measured hand pose. Gripper 0 is open and 1 is closed. '
            'Moves play before the next observation; actual arrival must be checked from the measured state. '
            'Camera frames remain fixed during a decision. '
            'take_pic only reveals a frame; it does not move a camera. '
            'A note should briefly describe what you see and why you chose the motion. '
            'Use done when you believe the task is complete, or give_up when you cannot continue. '
            'Both require hindsight for inspection; no advice is carried into another episode. '
            f'Motion limits: {json.dumps(asdict(self.policy.motion))}. '
            f'API call budget, including corrections and pictures: {self.policy.max_calls}.'
        )
        self.transcript.write(
            'instructions', prompt=prompt, tools={tool.name: tool.parameters_json_schema for tool in self.tools}
        )
        return ModelRequest([SystemPromptPart(prompt)])

    @staticmethod
    def _reject(messages: list[ModelMessage], calls: list[ToolCallPart], error: str) -> None:
        parts = [ToolReturnPart(call.tool_name, f'Rejected: {error}', tool_call_id=call.tool_call_id) for call in calls]
        messages.append(ModelRequest(parts if parts else [UserPromptPart(f'Rejected: {error}. Make one tool call.')]))

    def _tool(
        self, call: ToolCallPart, messages: list[ModelMessage], obs: _Observation, revealed: set[str]
    ) -> _Decision | None:
        tool = Tool(call.tool_name)
        data = _SCHEMAS[tool].model_validate_json(call.args_as_json_str())
        result: Any
        match data:
            case MoveTo():
                duration = self.policy.motion.duration(obs.pose, data.pose)
                result = (
                    f'Target accepted for a move lasting at least {duration:.3f} seconds. Check the next observation.'
                )
                messages.append(ModelRequest([ToolReturnPart(tool.value, result, tool_call_id=call.tool_call_id)]))
                return _Decision(messages, target=data)
            case Finish():
                messages.append(
                    ModelRequest([ToolReturnPart(tool.value, 'Attempt ended.', tool_call_id=call.tool_call_id)])
                )
                return _Decision(messages, stop_reason=tool.value, hindsight=data.hindsight)
            case TakePic():
                if self.policy.images is not Images.ON_DEMAND:
                    raise ValueError('take_pic is available only with images=on_demand')
                cameras = data.cameras if data.cameras else list(obs.images)
                if len(set(cameras)) != len(cameras) or set(cameras) - obs.images.keys() or set(cameras) & revealed:
                    raise ValueError('Choose available cameras not already revealed in this observation')
                revealed.update(cameras)
                messages.append(
                    ModelRequest([ToolReturnPart(tool.value, 'Frames attached.', tool_call_id=call.tool_call_id)])
                )
                messages.append(obs.frames(cameras))
                return None
        raise ValueError(f'Unsupported tool: {tool}')

    def _turn(
        self, response: ModelResponse, messages: list[ModelMessage], obs: _Observation, revealed: set[str]
    ) -> _Decision | None:
        calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
        try:
            if len(calls) != 1 or response.finish_reason in ('length', 'content_filter', 'error'):
                raise ValueError('Expected exactly one complete tool call')
            return self._tool(calls[0], messages, obs, revealed)
        except ValueError as exc:
            self._reject(messages, calls, str(exc))
            self.transcript.write('rejected', call=self.calls, reason=str(exc))
            raise

    def _request(self, messages: list[ModelMessage], obs_time_ns: int, cameras: set[str]) -> ModelResponse:
        self.calls += 1
        self.transcript.write('request', call=self.calls, obs_time_ns=obs_time_ns, cameras=sorted(cameras))
        response = self.policy.endpoint.request(self._outgoing(messages), self.tools)
        self.transcript.write(
            'response',
            call=self.calls,
            model=response.model_name,
            finish_reason=response.finish_reason,
            usage=asdict(response.usage),
            tools=[
                {'name': part.tool_name, 'arguments': part.args, 'id': part.tool_call_id}
                for part in response.parts
                if isinstance(part, ToolCallPart)
            ],
            text=[part.content for part in response.parts if isinstance(part, TextPart)],
        )
        return response

    @telemetry.traced(telemetry_keys.SPAN_POLICY_INFER)
    def advance(
        self, history: list[ModelMessage], target: MoveTo | None, raw: Mapping[str, Any], cancelled: threading.Event
    ) -> _Decision | None:
        obs = _Observation.read(raw, self.policy.camera_keys, self.policy.image_size)
        messages: list[ModelMessage] = list(history) if history else [self._initial()]
        state = obs.state(target)
        self.transcript.write('observation', **state)
        messages.append(ModelRequest.user_text_prompt(json.dumps(state, allow_nan=False)))
        revealed: set[str] = set()
        if self.policy.images is Images.ALWAYS:
            messages.append(obs.frames(list(obs.images)))
            revealed.update(obs.images)
        failures = 0
        while not cancelled.is_set():
            if self.calls >= self.policy.max_calls:
                self.transcript.write('budget_exhausted', calls=self.calls)
                return _Decision(messages, stop_reason='call_budget')
            response = self._request(messages, obs.time_ns, revealed)
            if cancelled.is_set():
                self.transcript.write('discarded', call=self.calls)
                return None
            messages.append(response)
            try:
                decision = self._turn(response, messages, obs, revealed)
            except ValueError as exc:
                failures += 1
                if failures >= self.policy.max_invalid:
                    raise RuntimeError(f'Model produced {failures} consecutive invalid replies') from exc
                continue
            failures = 0
            if decision is not None:
                return decision
        return None


class LLMPolicy(Policy):
    """An API-backed, single-arm Cartesian policy with independent episode conversations."""

    _ADVANCE = 'llm.advance'

    class _Session(Session):
        def __init__(self, policy: 'LLMPolicy', rt: Runtime):
            self._rt = rt
            self._transcript = Transcript()
            self._conversation = _Conversation(policy, self._transcript)
            self._history: list[ModelMessage] = []
            self._target: MoveTo | None = None
            self._answer: Answer | None = None
            self._cancelled = threading.Event()
            self._stop_reason: str | None = None
            self._hindsight: str | None = None

        @property
        def stop_requested(self) -> bool:
            return self._stop_reason is not None

        @property
        def meta(self) -> dict[str, Any]:
            policy = self._conversation.policy
            meta: dict[str, Any] = {
                policy_keys.TYPE: 'llm',
                'model': policy.endpoint.model,
                'api': policy.endpoint.api.value,
                'settings': dict(policy.endpoint.settings),
                'motion': asdict(policy.motion),
                'max_calls': policy.max_calls,
                'max_invalid': policy.max_invalid,
                'timeout': policy.endpoint.timeout,
                'images': policy.images.value,
                'camera_keys': list(policy.camera_keys),
                'image_size': policy.image_size,
                'image_horizon': policy.image_horizon,
                'transcript': self._transcript.snapshot(),
            }
            if policy.endpoint.base_url is not None:
                meta['base_url'] = policy.endpoint.base_url
            if self._stop_reason is not None:
                meta['stop_reason'] = self._stop_reason
            if self._hindsight is not None:
                meta['hindsight'] = self._hindsight
            return meta

        def __call__(self, obs: Mapping[str, Any], time_ns: int) -> list[dict] | None:
            if self.stop_requested:
                return []
            if self._answer is None:
                self._cancelled = threading.Event()
                self._answer = self._rt.fns[LLMPolicy._ADVANCE](
                    self._conversation, list(self._history), self._target, obs, self._cancelled
                )
                return None
            if not self._answer.done():
                return None
            answer, self._answer = self._answer, None
            decision = answer.result()
            if decision is None:
                return None
            if self._cancelled.is_set():
                self._transcript.write('discarded', call=self._conversation.calls)
                return None
            self._history, self._target = decision.messages, decision.target
            self._stop_reason, self._hindsight = decision.stop_reason, decision.hindsight
            trajectory = []
            if self._target is not None:
                start = _Observation.measured_pose(obs)
                try:
                    trajectory = self._conversation.policy.motion.trajectory(start, self._target)
                except ValueError as exc:
                    self._target = None
                    self._history.append(
                        ModelRequest.user_text_prompt(
                            f'Move was not executed: the measured pose changed while waiting. {exc} '
                            'Reassess the next observation.'
                        )
                    )
                    self._transcript.write('discarded', call=self._conversation.calls, reason=str(exc))
                    return []
            self._transcript.write(
                'accepted', call=self._conversation.calls, time_ns=time_ns, stop_reason=self._stop_reason
            )
            return trajectory

        def cancel(self):
            if (self._answer is not None or self._target is not None) and not self._cancelled.is_set():
                self._cancelled.set()
                self._target = None
                self._history.append(
                    ModelRequest.user_text_prompt(
                        'Motion or inference was cancelled. Reassess from the next measured observation.'
                    )
                )
                self._transcript.write('cancelled')

        def close(self):
            if self._answer is not None and not self._answer.done():
                raise RuntimeError('Close the runtime before closing the LLM session')
            self.cancel()
            if self._answer is not None:
                self._transcript.write('discarded', call=self._conversation.calls)

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

    @property
    def functions(self):
        return {self._ADVANCE: _Conversation.advance}

    def new_session(self, context=None, rt=None):
        if rt is None:
            raise ValueError('An LLM session needs a runtime; pass rt to new_session')
        return self._Session(self, rt)


@cfn.config(
    api=API.OPENAI_RESPONSES.value,
    base_url=None,
    api_key_env=None,
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
    model: str,
    api: str,
    base_url: str | None,
    api_key_env: str | None,
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
        Endpoint(model, API(api), base_url, api_key_env, timeout, settings),
        motion,
        images=Images(images),
        camera_keys=tuple(camera_keys),
        image_size=image_size,
        image_horizon=image_horizon,
        max_calls=max_calls,
        max_invalid=max_invalid,
    )
    return (StopOnFault() | ChunkedSchedule()).wrap(policy)
