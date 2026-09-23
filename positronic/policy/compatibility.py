"""Version 1 stack semantics executed by the processor runtime.

V1 layers exchange whole timestamped trajectories. Cancellation clears scheduling and history,
and discards an outstanding inference result after reading it. Only this adapter interprets those
trajectories; the harness receives ordinary Steps.
"""

import time
from abc import abstractmethod
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from positronic.policy import keys as policy_keys
from positronic.policy.base import ARGS, NAME, VERSION, Answer, Obs, Policy, PolicyRun, Runtime, Step
from positronic.policy.codec import Codec
from positronic.policy.layers import _arms_available, _StackBuffer
from positronic.policy.sequential import Sequential

TIMESTAMP = 'timestamp'
OBS_TIME_NS = 'obs_time_ns'
WALL_TIME_NS = 'wall_time_ns'

Trajectory = list[dict[str, Any]] | None


@dataclass
class _Call:
    send: Callable[[Obs], Trajectory]
    cancel: Callable[[], None]


class _LayerV1(Policy):
    @abstractmethod
    def bind(self, runtime: Runtime, inner: _Call) -> _Call:
        raise NotImplementedError

    def run(self, runtime: Runtime, infer: Callable[[Obs], Any]) -> PolicyRun:
        yield from StackV1(self).run(runtime, infer)

    def to_spec(self) -> dict[str, Any]:
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION}


class StopOnFaultV1(_LayerV1):
    WIRE_NAME = 'stop_on_fault'

    def bind(self, runtime: Runtime, inner: _Call) -> _Call:
        def send(obs: Obs) -> Trajectory:
            if _arms_available(obs):
                return inner.send(obs)
            inner.cancel()
            return []

        return _Call(send, inner.cancel)


class ChunkedScheduleV1(_LayerV1):
    WIRE_NAME = 'chunked_schedule'

    def bind(self, runtime: Runtime, inner: _Call) -> _Call:
        end_ns: int | None = None

        def send(obs: Obs) -> Trajectory:
            nonlocal end_ns
            if end_ns is not None and obs[OBS_TIME_NS] < end_ns:
                return None
            result = inner.send(obs)
            if result is not None:
                anchor = runtime.time_ns / 1e9
                result = [{**action, TIMESTAMP: anchor + action.get(TIMESTAMP, 0.0)} for action in result]
                end_ns = round(result[-1][TIMESTAMP] * 1e9) if result else None
            return result

        def cancel() -> None:
            nonlocal end_ns
            end_ns = None
            inner.cancel()

        return _Call(send, cancel)


class TemporalStackV1(_LayerV1):
    WIRE_NAME = 'temporal_stack'

    def __init__(self, keys: tuple[str, ...], offsets_sec: tuple[float, ...], pad_start: bool = True):
        self._keys, self._offsets_sec, self._pad_start = tuple(keys), tuple(offsets_sec), pad_start
        if not pad_start and 0.0 not in offsets_sec:
            raise ValueError('pad_start=False requires a current-frame offset of 0.0')

    def bind(self, runtime: Runtime, inner: _Call) -> _Call:
        buffer = _StackBuffer(self._offsets_sec, self._pad_start)

        def send(obs: Obs) -> Trajectory:
            now = obs[OBS_TIME_NS] / 1e9
            buffer.append(now, {key: obs[key] for key in self._keys})
            return inner.send(buffer.sample(now, obs))

        def cancel() -> None:
            buffer.reset()
            inner.cancel()

        return _Call(send, cancel)

    def to_spec(self) -> dict[str, Any]:
        return {
            **super().to_spec(),
            ARGS: {'keys': list(self._keys), 'offsets_sec': list(self._offsets_sec), 'pad_start': self._pad_start},
        }


def _returning_a_trajectory(infer: Callable[[Obs], Any]) -> Callable[[Obs], Any]:
    def infer_trajectory(obs: Obs) -> Any:
        result = infer(obs)
        return [dict(result)] if isinstance(result, Mapping) else result

    return infer_trajectory


class StackV1(Sequential):
    def __init__(self, first, *rest):
        components = tuple(
            child for part in (first, *rest) for child in (part._components if isinstance(part, StackV1) else (part,))
        )
        if not all(isinstance(part, (_LayerV1, Codec)) for part in components):
            raise ValueError('V1 trajectory layers cannot be mixed with Step processors; upgrade the server stack')
        super().__init__(components[0], *components[1:])

    @staticmethod
    def _inference(runtime: Runtime, infer: Callable[[Obs], Any]) -> _Call:
        answer: Answer[Any] | None = None
        cancelled = False

        def send(obs: Obs) -> Trajectory:
            nonlocal answer, cancelled
            if answer is None:
                answer = runtime.submit(infer, obs)
                return None
            if not answer.done():
                return None
            completed, discard = answer, cancelled
            answer, cancelled = None, False
            result = completed.result()
            if discard:
                return None
            return result

        def cancel() -> None:
            nonlocal cancelled
            cancelled = answer is not None

        return _Call(send, cancel)

    def run(self, runtime: Runtime, *dependencies: Any) -> PolicyRun:
        (infer,) = dependencies
        infer = _returning_a_trajectory(infer)
        components = list(self._components)
        # Codecs under the innermost layer run in the submitted work, so a tick that sends nothing encodes nothing.
        while components and isinstance(tail := components[-1], Codec):
            components.pop()
            infer = tail.wrap(infer)
        call = self._inference(runtime, infer)
        for component in reversed(components):
            if isinstance(component, _LayerV1):
                call = component.bind(runtime, call)
            else:
                call = _Call(cast(Codec, component).wrap(call.send), call.cancel)
        trajectory: deque[dict[str, Any]] = deque()
        obs = yield
        while True:
            now_ns = runtime.time_ns
            result = call.send({**obs, OBS_TIME_NS: now_ns, WALL_TIME_NS: time.time_ns()})
            if result is not None:
                trajectory = deque(result)
            commands = {}
            while trajectory and round(trajectory[0].get(TIMESTAMP, 0.0) * 1e9) <= now_ns:
                commands.update({key: value for key, value in trajectory.popleft().items() if key != TIMESTAMP})
            resume_at_ns = round(trajectory[0][TIMESTAMP] * 1e9) if trajectory else now_ns
            obs = yield Step(commands, resume_at_ns)


class ActionTimestampV1(Codec):
    WIRE_NAME = 'action_timestamp'

    def __init__(self, *, fps: float):
        self._fps = fps
        self._dt = 1.0 / fps

    def encode(self, data):
        return data

    def decode(self, data):
        if isinstance(data, list):
            stamped = [{**action, TIMESTAMP: i * self._dt} for i, action in enumerate(data)]
            if stamped:
                stamped.append({TIMESTAMP: len(stamped) * self._dt})
            return stamped
        return {**data, TIMESTAMP: 0}

    @property
    def meta(self):
        return {policy_keys.ACTION_FPS: self._fps}

    def to_spec(self):
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: {'fps': self._fps}}


class ActionHorizonV1(Codec):
    WIRE_NAME = 'action_horizon'

    def __init__(self, horizon_sec: float):
        self._horizon_sec = horizon_sec

    def encode(self, data):
        return data

    def decode(self, data):
        if not isinstance(data, list):
            return data
        kept = [action for action in data if action.get(TIMESTAMP, 0.0) < self._horizon_sec]
        if len(kept) < len(data):
            kept.append({TIMESTAMP: self._horizon_sec})
        return kept

    @property
    def meta(self):
        return {policy_keys.ACTION_HORIZON_SEC: self._horizon_sec}

    def to_spec(self):
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: {'horizon_sec': self._horizon_sec}}
