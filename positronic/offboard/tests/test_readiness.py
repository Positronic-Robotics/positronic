"""The readiness gate: a policy that cannot serve must be found before an operator is offered Start.

The endpoint under test connects and answers its HTTP probe but never becomes able to serve; nothing
below the status protocol distinguishes it from a healthy one.

The server end is a bare websocket, not ``PolicyServer``: a correct implementation cannot produce the
misbehaviour the client is being tested against.
"""

import asyncio
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
from positronic.eval import Embodiment
from positronic.offboard.client import InferenceClient, ServerNotReady
from positronic.offboard.protocol import ERROR, LOADING, MESSAGE, META, READY, STATUS
from positronic.policy.base import Policy, SampledPolicy
from positronic.policy.remote import RemotePolicy
from positronic.utils.serialization import serialise


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


async def _forever_loading(websocket):
    """Accept, then stream ``loading`` indefinitely.

    Frames go out inside the per-message allowance, so the timeout that bounds SILENCE never fires.
    """
    while True:
        await websocket.send(serialise({STATUS: LOADING, MESSAGE: 'loading checkpoint 50k'}))
        await asyncio.sleep(0.05)


# Minimal buildable declaration: a ready server must still name a stack for the endpoint to be usable.
READY_META = {keys.LOCAL_STACK: {'name': 'chunked_schedule'}}


async def _ready_at_once(websocket):
    await websocket.send(serialise({STATUS: READY, META: READY_META}))
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
    assert excinfo.value.status == LOADING
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
        await websocket.send(serialise({STATUS: ERROR}))
        await asyncio.Future()

    async def _reason_only(websocket):
        await websocket.send(serialise({ERROR: 'checkpoint 50k is not on this node'}))
        await asyncio.Future()

    with pytest.raises(RuntimeError, match='Server error'):
        InferenceClient(fake_server(_status_only)).new_session(ready_deadline=time.monotonic() + 5.0)

    with pytest.raises(RuntimeError, match='checkpoint 50k is not on this node'):
        InferenceClient(fake_server(_reason_only)).new_session(ready_deadline=time.monotonic() + 5.0)


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
        await websocket.send(serialise({STATUS: READY, META: {keys.LOCAL_STACK: {'name': 'no_such_wrapper_anywhere'}}}))
        await asyncio.Future()

    with pytest.raises(ValueError, match='local stack'):
        RemotePolicy(fake_server(_ready_with_unbuildable_stack)).wait_ready(5.0)


@pytest.mark.timeout(30)
def test_a_sampled_batch_refuses_when_any_endpoint_cannot_serve(fake_server):
    """Carrying on without one member hands its episodes to the others, changing everyone's numbers."""
    good, bad = fake_server(_ready_at_once), fake_server(_forever_loading)
    batch = SampledPolicy(RemotePolicy(good), RemotePolicy(bad))

    with pytest.raises(RuntimeError, match='cannot serve') as excinfo:
        batch.wait_ready(1.0)

    assert bad in str(excinfo.value)
    assert '1 of 2' in str(excinfo.value)


@pytest.mark.timeout(30)
def test_every_endpoint_is_reported_not_just_the_first(fake_server):
    """Reporting one of two bad members sends the operator round the loop twice."""
    first, second = fake_server(_forever_loading), fake_server(_forever_loading)
    batch = SampledPolicy(RemotePolicy(first), RemotePolicy(second))

    with pytest.raises(RuntimeError) as excinfo:
        batch.wait_ready(1.0)

    assert first in str(excinfo.value) and second in str(excinfo.value)
    assert '2 of 2' in str(excinfo.value)


@pytest.mark.timeout(30)
def test_the_endpoints_are_waited_on_concurrently(fake_server):
    """Sequentially the gate's cost would grow with the sample it protects: bound times set size."""
    urls = [fake_server(_forever_loading) for _ in range(3)]
    batch = SampledPolicy(*(RemotePolicy(u) for u in urls))

    started = time.monotonic()
    with pytest.raises(RuntimeError):
        batch.wait_ready(2.0)
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f'waited {elapsed:.1f}s for 3 endpoints bounded at 2.0s each'


@pytest.mark.timeout(20)
def test_a_local_policy_is_ready_when_it_is_constructed():
    """No network, nothing to wait for: an in-process policy needs no override to pass the gate."""

    class Local(Policy):
        def new_session(self, context=None, now=None):
            raise AssertionError('not reached')

    Local().wait_ready(0.0)


# --- the gate, where the run is refused --------------------------------------------------------


@pytest.mark.timeout(30)
def test_a_run_is_refused_before_the_operator_surface_is_built(fake_server):
    """The driver factory builds the operator's UI, and must never be called for a policy that
    cannot answer."""
    built = []

    with pytest.raises(ServerNotReady):
        main(
            policy=RemotePolicy(fake_server(_forever_loading)),
            embodiment=cast(Embodiment, SimpleNamespace(simulated=False)),
            driver=lambda _out: built.append('driver') or (_ for _ in ()).throw(AssertionError('not reached')),
            ready_timeout=1.0,
        )

    assert built == [], 'the operator surface was built for a policy that cannot serve'
