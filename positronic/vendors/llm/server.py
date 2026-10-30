"""Serve an API-backed LLM policy over the offboard protocol, one model per process.

    uv run --locked --extra llm-anthropic python -m positronic.vendors.llm.server \\
      --policy.model=anthropic:claude-opus-5-5 --policy.motion.max_translation=0.02
"""

import json
import logging
import math
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any

import configuronic as cfn
from positronic_model_server import server_wire

from pimm.logging import init_logging
from positronic.offboard import server as offboard_server
from positronic.offboard.server import serve_model
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs, PolicyEnd
from positronic.policy.processors import ChunkedSchedule, EndOnField
from positronic.policy.sequential import Sequential

from .policy import Conversation, LLMPolicy, llm

logger = logging.getLogger(__name__)


class _Session:
    """One session's model client, conversation and episode values, and how it ended: a decided end or a failure."""

    def __init__(self, policy: LLMPolicy):
        self._resources = ExitStack()
        self.client = self._resources.enter_context(policy.endpoint.open())
        self.metadata: dict[str, Any] = {}
        if self.client.api_key_fingerprint is not None:
            self.metadata[policy_keys.API_KEY_FINGERPRINT] = self.client.api_key_fingerprint
        self.conversation = Conversation(policy, self.metadata.__setitem__)
        self.end: PolicyEnd | None = None
        self.failure: Exception | None = None

    def close(self) -> None:
        self._resources.close()


class LLMModel(Model):
    """The conversations of one LLM policy, one per session, each over its own model client.

    Each call is one decision about its observation. A move answers with its trajectory. An end answers with an
    element that carries ``END``, and so does every later call of that session. A decision that raises, or that
    passes ``decision_timeout`` seconds before its next model call, fails the session, and every later call of the
    session raises. A session writes ``<sessions_dir>/<session_id>.json`` when it ends or fails and when it closes.
    The file holds the policy meta and the values the conversation writes, as the episode of a local run records them.
    A session opens its client, and draws its API key, at its first call. It closes the client when it ends, fails or
    closes.
    """

    def __init__(self, policy: LLMPolicy, sessions_dir: Path, decision_timeout: float):
        if not math.isfinite(decision_timeout) or decision_timeout <= 0:
            raise ValueError('decision_timeout must be finite and positive')
        self._policy = policy
        self._sessions_dir = sessions_dir
        self._decision_timeout = decision_timeout
        self._sessions: dict[str, _Session] = {}

    def _decide(self, session: _Session, obs: Obs) -> list[dict[str, Any]] | PolicyEnd:
        conversation = session.conversation
        deadline = time.monotonic() + self._decision_timeout
        follow_up = False
        while True:
            # A spent budget gives up as it does in a local run, so the deadline does not fail that session.
            if follow_up and not conversation.budget_spent and time.monotonic() >= deadline:
                raise TimeoutError(f'The decision took longer than its {self._decision_timeout} s deadline')
            # TODO: Stamp the observation on the client's runtime clock, as the local policy does. A client stack
            # that stamps it needs a new registered component, so the server's clock stands in for it.
            messages = conversation.request(obs, time.monotonic_ns())
            if isinstance(messages, PolicyEnd):
                return messages
            decision = conversation.respond(session.client.request(messages, self._policy.tools))
            if decision is not None:
                return conversation.accept(*decision, obs, time.monotonic_ns())
            follow_up = True

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        session = self._sessions.get(session_id)
        if session is None:
            session = self._sessions[session_id] = _Session(self._policy)
        if session.failure is not None:
            raise RuntimeError(f'This session failed in an earlier decision: {session.failure!r}')
        if session.end is None:
            try:
                decided = self._decide(session, obs)
            except Exception as failure:
                # The outcome latches before the record, so a failed write cannot let a later call ask the API again.
                session.failure = failure
                self._settle(session_id, session)
                raise
            if not isinstance(decided, PolicyEnd):
                # `ChunkedSchedule` plays the first element when it reads the answer. The move's `duration_s`
                # counts one period before the first waypoint, so an empty element holds that period.
                return [{}, *decided]
            session.end = decided
            self._settle(session_id, session)
        return [{policy_keys.END: session.end.value}]

    def _settle(self, session_id: str, session: _Session) -> None:
        try:
            self._write(session_id, session)
        finally:
            session.close()

    def _write(self, session_id: str, session: _Session) -> None:
        record = {**self._policy.meta(), **session.metadata, policy_keys.SESSION_ID: session_id}
        path = self._sessions_dir / f'{session_id}.json'
        pending = path.with_suffix('.json.tmp')
        pending.write_text(json.dumps(record))
        pending.replace(path)

    def end_session(self, session_id: str) -> None:
        if (session := self._sessions.pop(session_id, None)) is not None:
            self._settle(session_id, session)

    def meta(self) -> dict[str, Any]:
        return self._policy.meta()

    def close(self) -> None:
        # A session that fails to close does not stop the others from closing.
        with ExitStack() as sessions:
            for session_id in list(self._sessions):
                sessions.callback(self.end_session, session_id)


def deployment(policy: LLMPolicy) -> PolicyDeployment:
    """The client stack that plays the trajectories of ``policy`` and ends the run on its ``END``."""
    # The local policy plays its trajectory through an arm error, and so does this stack.
    return PolicyDeployment(Sequential(EndOnField(), ChunkedSchedule(policy.motion.fps, replan_on_error=False)))


@cfn.config(
    policy=llm,
    sessions_dir='llm_sessions',
    decision_timeout=360.0,
    websocket=offboard_server.websocket,
    grpc=None,
    idle_timeout_min=None,
)
def serve(
    policy: LLMPolicy,
    sessions_dir: str,
    decision_timeout: float,
    websocket: server_wire.Wire | None,
    grpc: server_wire.Wire | None,
    idle_timeout_min: float | None,
):
    """Serve ``policy`` with one conversation per session. ``--policy.*`` takes the flags of the local ``llm`` config.

    Each session's record goes to ``sessions_dir``, relative to the working directory unless absolute. The
    deployment comes from ``policy``, so it refuses session parameters: ``motion.fps`` has one owner. Each session
    takes its API key from this process's environment, as a local episode does. The wire flags are those of the
    offboard ``serve``, and the bearer token is that of ``serve_model``.
    """
    directory = Path(sessions_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    logger.info('Session records go to %s', directory)
    # No warmup inference: the model runs at its provider, and a warm call is a paid request. Opening one client
    # makes no request, and it refuses a bad endpoint config or key pool before the first session does.
    with policy.endpoint.open():
        pass
    wires = [w for w in (websocket, grpc) if w is not None]
    serve_model(partial(LLMModel, policy, directory, decision_timeout), deployment(policy), wires, idle_timeout_min)


if __name__ == '__main__':
    init_logging()
    cfn.cli(serve)
