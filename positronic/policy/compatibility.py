"""Build the stack that a protocol v1 server declares from v2 processors.

The offboard README states the translation and what the client refuses.
"""

import time
from collections.abc import Mapping
from typing import Any

from positronic.policy import keys as policy_keys
from positronic.policy.base import ARGS, NAME, SEQ, VERSION, Obs, Policy, PolicyRun, Processor, Runtime
from positronic.policy.codec import Codec
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable, TemporalStack
from positronic.policy.sequential import Sequential
from positronic.policy.spec import from_spec

TIMESTAMP = 'timestamp'
WALL_TIME_NS = 'wall_time_ns'
ACTION_TIMESTAMP = 'action_timestamp'
ACTION_HORIZON = 'action_horizon'
V1_LAYER_NAMES = (PauseOnUnavailable.WIRE_NAME, TemporalStack.WIRE_NAME)


class StampObservationTimes(Policy):
    """Add the control clock's ``obs_time_ns`` and the wall clock's ``wall_time_ns``, which a v1 server reads."""

    def run(self, runtime: Runtime, inner: PolicyRun) -> PolicyRun:
        obs = yield
        while True:
            stamped = {**obs, policy_keys.OBS_TIME_NS: runtime.time_ns, WALL_TIME_NS: time.time_ns()}
            obs = yield inner.send(stamped)


class ChunkFromV1Answer(Codec):
    """Turn a v1 answer into a chunk: a single row becomes one row, and timestamps and the end row go."""

    def encode(self, data: dict) -> dict:
        return data

    def decode(self, data: Any) -> list[dict[str, Any]]:
        rows = [data] if isinstance(data, Mapping) else data
        return [{key: row[key] for key in row if key != TIMESTAMP} for row in rows if set(row) != {TIMESTAMP}]


def _flat_parts(node: dict[str, Any]) -> list[dict[str, Any]]:
    if SEQ in node:
        return [part for child in node[SEQ] for part in _flat_parts(child)]
    return [node]


def _is_v1(part: dict[str, Any], name: str) -> bool:
    return part.get(NAME) == name and part.get(VERSION, 1) == 1


def from_v1_spec(node: dict[str, Any], server_meta: Mapping[str, Any]) -> Processor[Obs, Any]:
    """Build the stack a protocol v1 server declares, from its spec and its handshake metadata."""
    timing: dict[str, float] = {}
    parts: list[dict[str, Any]] = []
    for part in _flat_parts(node):
        if _is_v1(part, ACTION_TIMESTAMP):
            timing['fps'] = part[ARGS]['fps']
        elif _is_v1(part, ACTION_HORIZON):
            timing['horizon_sec'] = part[ARGS]['horizon_sec']
        elif part.get(VERSION, 1) == 1 and part.get(NAME) in V1_LAYER_NAMES:
            parts.append({**part, VERSION: 2})
        else:
            parts.append(part)
    if 'fps' not in timing:
        if policy_keys.ACTION_FPS not in server_meta:
            raise ValueError(
                'The v1 server declares no action_timestamp and sends no action_fps, so the client cannot '
                'schedule its answers; upgrade the server'
            )
        timing['fps'] = server_meta[policy_keys.ACTION_FPS]
    if 'horizon_sec' not in timing and server_meta.get(policy_keys.ACTION_HORIZON_SEC) is not None:
        timing['horizon_sec'] = server_meta[policy_keys.ACTION_HORIZON_SEC]

    schedule = {NAME: ChunkedSchedule.WIRE_NAME, VERSION: 2, ARGS: timing}
    positions = [i for i, part in enumerate(parts) if _is_v1(part, ChunkedSchedule.WIRE_NAME)]
    if positions:
        parts[positions[0]] = schedule
    else:
        layers = [i for i, part in enumerate(parts) if part.get(NAME) in V1_LAYER_NAMES]
        parts.insert(layers[-1] + 1 if layers else 0, schedule)
    return Sequential(StampObservationTimes(), from_spec({SEQ: parts}), ChunkFromV1Answer())
