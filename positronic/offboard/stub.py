"""A server with no model: every inference answers the same chunk, and a session measures the wire alone.

``delay_sec`` is a session param: ``?delay_sec=120`` holds one inference open for two minutes, with
nothing on the wire meanwhile.
"""

import time
from collections.abc import Mapping
from typing import Any

import configuronic as cfn

from pimm.logging import init_logging
from positronic import keys
from positronic.offboard.server import serve
from positronic.policy import Policy, Session
from positronic.policy.base import DelegatingSession, Layer, Runtime
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.spec import Pipeline, PolicySource, remote

# The smallest trajectory a served session can answer: one action, at the start of the chunk.
CHUNK = [{keys.ACTION_TIMESTAMP: 0.0}]


class StubSession(Session):
    def __call__(self, obs: Mapping[str, Any], time_ns: int) -> list[dict[str, Any]]:
        return CHUNK

    @property
    def meta(self) -> dict[str, Any]:
        return {'model_name': 'stub'}


class StubPolicy(Policy):
    """Answers ``CHUNK``, whatever it is asked."""

    def new_session(self, context: dict[str, Any] | None = None, rt: Runtime | None = None) -> Session:
        return StubSession()


class DelayedSession(DelegatingSession):
    def __init__(self, inner: Session, delay_sec: float):
        super().__init__(inner)
        self._delay_sec = delay_sec

    def __call__(self, obs: Mapping[str, Any], time_ns: int) -> list[dict[str, Any]] | None:
        time.sleep(self._delay_sec)
        return super().__call__(obs, time_ns)


class Delay(Layer):
    """Holds every answer for ``delay_sec``, in place of a model slow enough to outlast a front's idle close.

    A session param can tune a layer and cannot change the model source.
    """

    def __init__(self, delay_sec: float = 0.0):
        self._delay_sec = delay_sec

    def make_session(self, inner: Session) -> Session:
        return DelayedSession(inner, self._delay_sec)


# One instance for the process. A server refuses a session param that rebuilds a different source, and
# two ``PolicySource``s are equal only over one policy object.
POLICY = StubPolicy()


@cfn.config(delay_sec=0.0)
def pipeline(delay_sec: float) -> Pipeline:
    return ChunkedSchedule() | remote | Delay(delay_sec) | PolicySource(POLICY)


if __name__ == '__main__':
    init_logging()
    cfn.cli(serve.override(pipeline=pipeline))
