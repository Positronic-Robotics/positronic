"""The readiness gate: a policy that cannot serve must be found before an operator is offered Start.

The failure these pin is one an operator met at the rig: an endpoint that is deployed, answers its
HTTP probe, and completes a websocket connection — and then never becomes able to answer. Nothing
below the status protocol distinguishes it from a healthy one, so a run started on it comes up, the
arm energizes, and the fault surfaces minutes later in the middle of an episode.

The server end here is a bare websocket, not ``PolicyServer``: what is under test is how the CLIENT
reads a server that misbehaves, and our own server does not misbehave. A server streaming ``loading``
for ever is exactly what was observed, and no fixture over a correct implementation can produce it.
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

from positronic.cli.eval.run import main
from positronic.eval import Embodiment
from positronic.offboard.client import InferenceClient, ServerNotReady
from positronic.offboard.protocol import LOADING, MESSAGE, META, READY, STATUS
from positronic.policy.base import SEQ, Policy, SampledPolicy
from positronic.policy.remote import RemotePolicy
from positronic.utils.serialization import serialise


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_server() -> Generator[Callable[..., str], None, None]:
    """Serve a scripted status stream on a free port; returns the ws:// url. Stopped at teardown.

    Teardown closes the server from inside its own loop and lets that loop finish, rather than
    stopping the loop under it: a handler still streaming when the loop dies leaves the websockets
    server closing against a closed loop, and the resulting warnings are the kind that teach a
    reader to ignore warnings.
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
    """Accept, then stream ``loading`` indefinitely — the observed failure, in its own words.

    Frames go out well inside the handshake's per-message allowance, so the timeout that bounds
    SILENCE never fires: this server is talking, and would talk until somebody gave up by hand.
    """
    while True:
        await websocket.send(serialise({STATUS: LOADING, MESSAGE: 'loading checkpoint 50k'}))
        await asyncio.sleep(0.05)


async def _ready_at_once(websocket):
    await websocket.send(serialise({STATUS: READY, META: {'local_stack': {SEQ: []}}}))
    await asyncio.Future()


# --- the protocol layer ------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_a_server_that_never_becomes_ready_is_given_up_on_and_names_its_last_state(fake_server):
    """The whole failure in one place: connected, talking, never servable — and bounded."""
    url = fake_server(_forever_loading)
    client = InferenceClient(url)

    started = time.monotonic()
    with pytest.raises(ServerNotReady) as excinfo:
        client.new_session(ready_deadline=time.monotonic() + 1.0)

    assert time.monotonic() - started < 10, 'the wait was not bounded by the deadline'
    # Named: which endpoint, and what it was doing when the wait ended. Both are what an operator
    # reads off the refusal, and the status frame is the only evidence of the second.
    assert url in str(excinfo.value)
    assert excinfo.value.status == LOADING
    assert 'loading checkpoint 50k' in str(excinfo.value)


@pytest.mark.timeout(20)
def test_a_server_that_reports_ready_passes_the_gate(fake_server):
    """The gate must not cost a healthy endpoint anything: ready is read on the first frame."""
    session = InferenceClient(fake_server(_ready_at_once)).new_session(ready_deadline=time.monotonic() + 5.0)
    assert session.metadata == {'local_stack': {SEQ: []}}
    session.close()


@pytest.mark.timeout(20)
def test_giving_up_is_not_retried_as_a_cold_start(fake_server):
    """A server that keeps talking and never becomes ready is not a server a reconnect fixes, so it
    must not be swallowed by the retry loop that exists for cold backends — which would spend the
    connect deadline discovering the same thing several times over."""
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
    """Being ready is not the only way an endpoint fails a run. A server can report ready and declare
    a local stack naming a wrapper this rig does not have, and building it is what raises — at the
    opening of the first episode, unless the gate does it here."""

    async def _ready_with_unbuildable_stack(websocket):
        await websocket.send(serialise({STATUS: READY, META: {'local_stack': {'name': 'no_such_wrapper_anywhere'}}}))
        await asyncio.Future()

    with pytest.raises(ValueError, match='local stack'):
        RemotePolicy(fake_server(_ready_with_unbuildable_stack)).wait_ready(5.0)


@pytest.mark.timeout(30)
def test_a_sampled_batch_refuses_when_any_endpoint_cannot_serve(fake_server):
    """A sampled set is a comparison. Carrying on without one member silently hands its share of the
    episodes to the others, so every member's numbers change and nothing records which set ran."""
    good, bad = fake_server(_ready_at_once), fake_server(_forever_loading)
    batch = SampledPolicy(RemotePolicy(good), RemotePolicy(bad))

    with pytest.raises(RuntimeError, match='cannot serve') as excinfo:
        batch.wait_ready(1.0)

    assert bad in str(excinfo.value)
    assert '1 of 2' in str(excinfo.value)


@pytest.mark.timeout(30)
def test_every_endpoint_is_reported_not_just_the_first(fake_server):
    """Two bad members are two facts. Reporting one sends the operator round the loop twice."""
    first, second = fake_server(_forever_loading), fake_server(_forever_loading)
    batch = SampledPolicy(RemotePolicy(first), RemotePolicy(second))

    with pytest.raises(RuntimeError) as excinfo:
        batch.wait_ready(1.0)

    assert first in str(excinfo.value) and second in str(excinfo.value)
    assert '2 of 2' in str(excinfo.value)


@pytest.mark.timeout(30)
def test_the_endpoints_are_waited_on_concurrently(fake_server):
    """Sequentially, a per-endpoint bound is a bound times the size of the set — so the gate's own
    cost would grow with the sample it exists to protect. Asserted as wall clock against the sum,
    which is the property that matters and the one a sequential implementation cannot have."""
    urls = [fake_server(_forever_loading) for _ in range(3)]
    batch = SampledPolicy(*(RemotePolicy(u) for u in urls))

    started = time.monotonic()
    with pytest.raises(RuntimeError):
        batch.wait_ready(2.0)
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f'waited {elapsed:.1f}s for 3 endpoints bounded at 2.0s each'


@pytest.mark.timeout(20)
def test_a_local_policy_is_ready_when_it_is_constructed():
    """Nothing holds a model over a network here, so there is nothing to wait for — and a gate that
    demanded an override of every in-process policy would be a gate nobody could adopt."""

    class Local(Policy):
        def new_session(self, context=None, now=None):
            raise AssertionError('not reached')

    Local().wait_ready(0.0)


# --- the gate, where the run is refused --------------------------------------------------------


@pytest.mark.timeout(30)
def test_a_run_is_refused_before_the_operator_surface_is_built(fake_server):
    """The point of the whole change: the driver factory builds the operator's UI, and it must never
    be called for a policy that cannot answer. A run refused here costs the minutes it takes to bring
    the endpoint up; one refused after it costs an operator standing at a live arm."""
    built = []

    with pytest.raises(ServerNotReady):
        main(
            policy=RemotePolicy(fake_server(_forever_loading)),
            embodiment=cast(Embodiment, SimpleNamespace(simulated=False)),
            driver=lambda _out: built.append('driver') or (_ for _ in ()).throw(AssertionError('not reached')),
            ready_timeout=1.0,
        )

    assert built == [], 'the operator surface was built for a policy that cannot serve'
