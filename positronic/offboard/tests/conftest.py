import asyncio
import socket
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import ExitStack, contextmanager
from typing import Any

import pytest
import uvicorn

from positronic.offboard.server import PolicyServer
from positronic.policy import INFER, Policy, Session
from positronic.policy.executor import Executor
from positronic.policy.layers import ChunkPlayer
from positronic.policy.spec import ModelSource, PolicySource, remote


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


StartServer = Callable[..., tuple[str, int, PolicyServer]]


@pytest.fixture
def start_server() -> Generator[StartServer, None, None]:
    """Factory serving pipelines on daemon threads; every started server is stopped and joined at teardown."""
    running: list[tuple[uvicorn.Server, threading.Thread]] = []

    def start(pipeline, **server_kwargs) -> tuple[str, int, PolicyServer]:
        server = PolicyServer(pipeline, host='localhost', port=_find_free_port(), **server_kwargs)
        uv_server = uvicorn.Server(uvicorn.Config(server.app, host=server.host, port=server.port, log_level='warning'))

        async def _run():
            await server._startup()
            await uv_server.serve()

        thread = threading.Thread(target=asyncio.run, args=(_run(),), daemon=True)
        thread.start()
        running.append((uv_server, thread))

        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                with socket.create_connection((server.host, server.port), timeout=0.1):
                    return server.host, server.port, server
            except (ConnectionRefusedError, OSError):
                time.sleep(0.05)
        raise RuntimeError('Server failed to start')

    yield start
    for uv_server, thread in running:
        uv_server.should_exit = True
        thread.join(timeout=5.0)


@pytest.fixture
def open_episode() -> Generator[Callable[[Policy], Mapping[str, Callable[..., Any]]], None, None]:
    """Opens a policy's episode; every one it opened is closed at teardown."""
    closing = ExitStack()

    def make(policy: Policy) -> Mapping[str, Callable[..., Any]]:
        return closing.enter_context(policy.episode())

    yield make
    closing.close()


@pytest.fixture
def open_session() -> Generator[Callable[[Policy], tuple[Session, Executor]], None, None]:
    """Opens a policy's session against a runtime that serves its episode, as the harness does."""
    closing = ExitStack()

    def make(policy: Policy) -> tuple[Session, Executor]:
        rt = Executor(closing.enter_context(policy.episode()))
        closing.callback(rt.close)
        session = policy.new_session(rt)
        closing.callback(session.close)
        return session, rt

    yield make
    closing.close()


# How long a round trip against a local server may take before a test calls it lost.
ANSWER_SEC = 5.0


def played_round_trip(session: Session, rt: Executor, obs, time_ns: int = 0) -> Mapping[str, Any]:
    """What ``session`` commands for ``obs`` once the call under its ``ChunkPlayer`` has come back.

    Every call gets the same ``time_ns``, so the chunk comes back anchored at the value the test passed and
    the answer is the waypoints due at it.
    """
    commands, _ = session(obs, time_ns)
    if commands:
        return commands
    rt.wait(ANSWER_SEC)
    assert not rt.in_flight, 'the round-trip never came back'
    return session(obs, time_ns)[0]


class MockPolicy(Policy):
    """Answers one fixed chunk, or raises ``failure``. Records its episodes and the observations it took."""

    def __init__(self, action, meta: dict[str, Any], failure: Exception | None = None):
        self._action = action
        self._meta = meta
        self._failure = failure
        self.episodes = 0
        self.closed = 0
        self.observations: list[Any] = []

    @contextmanager
    def episode(self, context=None):
        self.episodes += 1
        try:
            yield {INFER: self._infer}
        finally:
            self.closed += 1

    def _infer(self, obs):
        self.observations.append(obs)
        if self._failure is not None:
            raise self._failure
        return self._action

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)


@pytest.fixture
def make_mock_policy() -> Callable[..., MockPolicy]:
    return MockPolicy


class _DictSource(ModelSource):
    """Multi-model source over ready policies; the dict's first key is the default."""

    def __init__(self, policies: Mapping[str, Policy]):
        self._policies = policies

    def get_models(self) -> list[str]:
        return list(self._policies)

    def resolve(self, model_id: str | None) -> str:
        if model_id is None:
            return next(iter(self._policies))
        if model_id not in self._policies:
            raise ValueError(f'Unknown model {model_id!r}. Available: {list(self._policies)}')
        return model_id

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        return self._policies[model_id]


@pytest.fixture
def mock_policy() -> MockPolicy:
    return MockPolicy({'action_data': [1, 2, 3]}, {'model_name': 'test_model'})


@pytest.fixture
def mock_policy_registry() -> dict[str, MockPolicy]:
    return {
        'alpha': MockPolicy({'action_data': ['alpha']}, {'model_name': 'alpha'}),
        'beta': MockPolicy({'action_data': ['beta']}, {'model_name': 'beta'}),
    }


@pytest.fixture
def inference_server(start_server: StartServer, mock_policy: MockPolicy) -> tuple[str, int]:
    """A served single-policy pipeline.

    Returns:
        tuple[str, int]: (host, port)
    """
    host, port, _server = start_server(ChunkPlayer() | remote | PolicySource(mock_policy))
    return host, port


@pytest.fixture
def multi_policy_server(
    start_server: StartServer, mock_policy_registry: dict[str, MockPolicy]
) -> tuple[str, int, dict[str, MockPolicy]]:
    host, port, _server = start_server(ChunkPlayer() | remote | _DictSource(mock_policy_registry))
    return host, port, mock_policy_registry
