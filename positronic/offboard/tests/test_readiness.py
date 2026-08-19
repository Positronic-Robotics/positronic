"""The readiness gate: a policy that cannot serve must be found before an operator is offered Start.

The endpoint under test connects and answers its HTTP probe but never becomes able to serve; nothing
below the status protocol distinguishes it from a healthy one.

The server end is a bare websocket, not ``PolicyServer``: a correct implementation cannot produce the
misbehaviour the client is being tested against.
"""

import asyncio
import itertools
import socket
import threading
import time
from collections.abc import Callable, Generator
from types import SimpleNamespace
from typing import cast

import pytest
import websockets

from positronic import keys
from positronic.cli.eval.run import main
from positronic.eval import Eval
from positronic.offboard.client import InferenceClient, ServerNotReady
from positronic.offboard.protocol import ERROR, MESSAGE, META, STATUS, ServerStatus, serialise
from positronic.policy.base import Policy
from positronic.policy.remote import RemotePolicy
from positronic.policy.wrappers import ChunkedSchedule


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_server() -> Generator[Callable[..., str], None, None]:
    """Serve a scripted status stream on a free port; returns the ws:// url. Stopped at teardown.

    Teardown closes the server from inside its own loop: killing the loop under a streaming handler
    leaves websockets closing against a closed loop and warning about it.
    """
    started: list[tuple[threading.Event, threading.Thread]] = []

    def start(handler) -> str:
        port = _free_port()
        bound, stop = threading.Event(), threading.Event()

        async def _serve():
            async with websockets.serve(handler, 'localhost', port):
                bound.set()
                await asyncio.get_running_loop().run_in_executor(None, stop.wait)

        thread = threading.Thread(target=lambda: asyncio.run(_serve()), daemon=True)
        thread.start()
        assert bound.wait(5.0), 'fake server did not bind'
        started.append((stop, thread))
        return f'ws://localhost:{port}'

    yield start
    for stop, thread in started:
        stop.set()
        thread.join(timeout=5.0)


async def _stream_loading(websocket):
    """Stream ``loading`` indefinitely.

    Frames go out inside the per-message allowance, so the timeout that bounds SILENCE never fires.
    """
    while True:
        await websocket.send(serialise({STATUS: ServerStatus.LOADING, MESSAGE: 'loading checkpoint 50k'}))
        await asyncio.sleep(0.05)


async def _forever_loading(websocket):
    """Accept, then never become ready."""
    await _stream_loading(websocket)


# Minimal buildable declaration: a ready server must still name a stack for the endpoint to be usable.
# Built by the wrapper that owns the wire name, so the fixture cannot drift from what `from_spec` accepts.
READY_META = {keys.LOCAL_STACK: ChunkedSchedule().to_spec()}


async def _accept_then_drop(websocket):
    """Accept and close without a status frame — a cold backend's shape, which the connect loop retries."""
    await websocket.close()


async def _ready_at_once(websocket):
    await websocket.send(serialise({STATUS: ServerStatus.READY, META: READY_META}))
    await asyncio.Future()


# --- the protocol layer ------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_a_server_that_never_becomes_ready_is_given_up_on_and_names_its_last_state(fake_server):
    """Connected, talking, never servable — and bounded."""
    url = fake_server(_forever_loading)
    client = InferenceClient(url)

    started = time.monotonic()
    with pytest.raises(ServerNotReady) as excinfo:
        client.new_session(ready_deadline=time.monotonic() + 1.0)

    assert time.monotonic() - started < 10, 'the wait was not bounded by the deadline'
    # The refusal names the endpoint and its last status frame.
    assert url in str(excinfo.value)
    assert excinfo.value.status == ServerStatus.LOADING
    assert 'loading checkpoint 50k' in str(excinfo.value)


@pytest.mark.timeout(20)
def test_a_server_that_reports_ready_passes_the_gate(fake_server):
    """A healthy endpoint pays nothing: ready is read on the first frame."""
    session = InferenceClient(fake_server(_ready_at_once)).new_session(ready_deadline=time.monotonic() + 5.0)
    assert session.metadata == READY_META
    session.close()


@pytest.mark.timeout(20)
def test_a_server_reporting_an_error_status_surfaces_it_rather_than_waiting(fake_server):
    """``ERROR`` is one word in two positions — the STATUS value and the field holding the reason.

    Each is asserted alone, since a frame carrying both raises on either one and would pass while
    half the contract was misspelled. A client that spells one differently from the server reads a
    failed session as an unrecognised frame, and waits out the deadline instead of reporting it.
    """

    async def _status_only(websocket):
        await websocket.send(serialise({STATUS: ServerStatus.ERROR}))
        await asyncio.Future()

    async def _reason_only(websocket):
        await websocket.send(serialise({ERROR: 'checkpoint 50k is not on this node'}))
        await asyncio.Future()

    with pytest.raises(RuntimeError, match='Server error'):
        InferenceClient(fake_server(_status_only)).new_session(ready_deadline=time.monotonic() + 5.0)

    with pytest.raises(RuntimeError, match='checkpoint 50k is not on this node'):
        InferenceClient(fake_server(_reason_only)).new_session(ready_deadline=time.monotonic() + 5.0)


@pytest.mark.timeout(30)
def test_a_retry_backoff_does_not_sleep_past_the_deadline(fake_server):
    """The backoff doubles towards 30s, so a retryable failure landing near the deadline would
    otherwise hold the caller well past the bound it asked for. Every attempt here fails at once,
    so the elapsed time is the sleeping."""
    client = InferenceClient(fake_server(_accept_then_drop))

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        client.new_session(ready_deadline=time.monotonic() + 1.5)
    elapsed = time.monotonic() - started

    # Uncapped, the second backoff alone is 2s and lands at ~3s; capped, the waits sum to the deadline.
    assert elapsed < 2.5, f'the retry slept {elapsed:.1f}s past a 1.5s deadline'


@pytest.mark.timeout(30)
def test_the_reported_wait_covers_the_connect_retries_too(fake_server):
    """The deadline bounds the connects as well as the handshake, so the wait the refusal reports has
    to measure the same span. Clocked from the handshake instead, an endpoint that spends most of its
    budget refusing connections and only then streams ``loading`` reports a fraction of what it took."""
    attempts = itertools.count()

    async def _drop_once_then_load(websocket):
        # One refused connect costs the first backoff, a second of the two the deadline allows.
        if next(attempts) == 0:
            await websocket.close()
            return
        await _stream_loading(websocket)

    with pytest.raises(ServerNotReady) as excinfo:
        InferenceClient(fake_server(_drop_once_then_load)).new_session(ready_deadline=time.monotonic() + 2.0)

    assert excinfo.value.waited_s >= 1.5, (
        f'a 2.0s wait was reported as {excinfo.value.waited_s:.2f}s — the second spent retrying went unmeasured'
    )


@pytest.mark.timeout(30)
def test_the_handshake_is_bounded_by_the_shorter_of_the_two_deadlines(fake_server):
    """``connect_deadline`` bounds the whole call, so a server that connects and then streams
    ``loading`` cannot outlive it by handshaking under the caller's later deadline."""
    client = InferenceClient(fake_server(_forever_loading), connect_deadline=1.0)

    started = time.monotonic()
    with pytest.raises(ServerNotReady):
        client.new_session(ready_deadline=time.monotonic() + 300.0)
    elapsed = time.monotonic() - started

    assert elapsed < 10.0, f'the handshake ran {elapsed:.1f}s under a 1.0s connect deadline'


@pytest.mark.timeout(20)
def test_giving_up_is_not_retried_as_a_cold_start(fake_server):
    """A reconnect does not fix it, so the cold-backend retry loop must not swallow it."""
    url = fake_server(_forever_loading)
    client = InferenceClient(url, connect_deadline=600.0)

    started = time.monotonic()
    with pytest.raises(ServerNotReady):
        client.new_session(ready_deadline=time.monotonic() + 1.0)

    assert time.monotonic() - started < 10, 'the give-up was retried instead of surfacing'


# --- the policy layer --------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_a_remote_policy_waits_on_its_own_endpoint_and_names_it(fake_server):
    url = fake_server(_forever_loading)
    with pytest.raises(ServerNotReady) as excinfo:
        RemotePolicy(url).wait_ready(1.0)
    assert url in str(excinfo.value)


@pytest.mark.timeout(30)
def test_a_ready_server_declaring_a_stack_this_rig_cannot_build_is_caught_at_the_gate(fake_server):
    """A ready server can declare a wrapper this rig does not have; building it is what raises."""

    async def _ready_with_unbuildable_stack(websocket):
        # Spelled out rather than built: no wrapper owns this name, which is the point of the case.
        declared = {keys.LOCAL_STACK: {'name': 'no_such_wrapper_anywhere'}}
        await websocket.send(serialise({STATUS: ServerStatus.READY, META: declared}))
        await asyncio.Future()

    with pytest.raises(ValueError, match='local stack'):
        RemotePolicy(fake_server(_ready_with_unbuildable_stack)).wait_ready(5.0)


@pytest.mark.timeout(20)
def test_a_local_policy_is_ready_when_it_is_constructed():
    """No network, nothing to wait for: an in-process policy needs no override to pass the gate."""

    class Local(Policy):
        def new_session(self, context=None, now=None):
            raise AssertionError('not reached')

    Local().wait_ready(0.0)


# --- the gate, where the run is refused --------------------------------------------------------


class _Untouchable:
    """An embodiment nothing may read: the first attribute read is a world coming up."""

    def __getattr__(self, name):
        raise AssertionError(f'a world was built for a policy that cannot serve: read {name!r}')


@pytest.mark.timeout(30)
def test_a_run_is_refused_before_anything_downstream_is_built(fake_server, tmp_path):
    """Everything the sweep sets up costs something a refused run should not pay: the output directory
    is synced and snapshotted into, and each eval brings a world up around the hardware."""
    with pytest.raises(ServerNotReady):
        main(
            policy=RemotePolicy(fake_server(_forever_loading)),
            evals=[cast(Eval, SimpleNamespace(embodiment=_Untouchable(), task=None, trials=[]))],
            output_dir=tmp_path,
            ready_timeout=1.0,
        )

    assert list(tmp_path.iterdir()) == [], 'the output directory was prepared for a policy that cannot serve'
