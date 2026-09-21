"""Where one inference's time goes, divided by the phases the server reports.

Replays a recorded episode against a server and reports what each round trip cost beside what the
server says it spent. Each run prints the stack it sent through: the one the ``--server_host``
server declares in its handshake, or the one the flags below build for the loopback server.

Usage
    uv run --locked python -m positronic.offboard.serving_cost \\
        --dataset.path=<episode root> --requests=20
    ... --server_host=<endpoint>          # AUTH_TOKEN gates a served endpoint, reached over TLS
    ... --server_wire=websocket --server_port=8000   # a plain server started by hand
    ... --compress_images=False           # loopback only: send raw stacks instead of per-frame JPEG
    ... --frames=25 --rate_hz=15 --width=1024 --height=288 --chunk_rows=24 --out=rows.json
"""

import contextlib
import json
import os
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import configuronic as cfn
import numpy as np
import pos3
from positronic_wire import registry, wire
from positronic_wire.websocket import WebsocketClientWire

import positronic.cfg.ds
from pimm.logging import init_logging
from positronic import keys
from positronic.cfg.policy import bearer_headers
from positronic.dataset.dataset import Dataset
from positronic.dataset.episode import Episode
from positronic.offboard import keys as offboard_keys
from positronic.offboard import protocol, server_wire, websocket_wire
from positronic.offboard.client import InferenceClient, InferenceSession
from positronic.offboard.server import AUTH_TOKEN_ENV, PolicyServer
from positronic.policy.base import DelegatingPolicy, DelegatingSession, Layer, Policy, Session
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, StopOnFault, TemporalStack
from positronic.policy.remote import declared_stack, prepare_obs
from positronic.policy.spec import PolicySource, remote


class InstantChunk(Policy):
    """The model the probe serves: it answers a fixed chunk of ``rows`` and does no work at all."""

    def __init__(self, rows: int, period_s: float):
        self.chunk = [
            {
                keys.ACTION_TIMESTAMP: row * period_s,
                keys.TARGET_JOINTS: np.zeros(7, dtype=np.float32),
                keys.TARGET_GRIP: np.float32(0.0),
            }
            for row in range(rows)
        ]

    def new_session(self, context: dict[str, Any] | None = None, rt=None) -> Session:
        return InstantChunk._Session(self.chunk)

    class _Session(Session):
        def __init__(self, chunk: list[dict[str, Any]]):
            self._chunk = chunk

        def __call__(self, obs, time_ns):
            return self._chunk


class CapturingWire(DelegatingPolicy):
    """Stands where the wire stands during capture: answers like the server and keeps what it was sent.

    It collects the messages the rig would have put on the wire, and the replay sends those.
    """

    def __init__(self, inner: Policy):
        super().__init__(inner)
        self.sent: list[dict[str, Any]] = []

    def new_session(self, context: dict[str, Any] | None = None, rt=None) -> Session:
        return CapturingWire._Session(self._inner.new_session(context, rt), self.sent)

    class _Session(DelegatingSession):
        def __init__(self, inner: Session, sent: list[dict[str, Any]]):
            super().__init__(inner)
            self._sent = sent

        def __call__(self, obs, time_ns):
            self._sent.append(dict(obs))
            return super().__call__(obs, time_ns)


def rig_stack(cameras: Sequence[str], frames: int, rate_hz: float, width: int, height: int) -> Layer:
    """The rig-side stack the client builds, with the depth and the image bound the caller names."""
    offsets = tuple(-(frames - 1 - step) / rate_hz for step in range(frames))
    stacked = (*cameras, keys.EE_POSE, keys.GRIP)
    return (
        StopOnFault()
        | TemporalStack(stacked, offsets)
        | ChunkedSchedule()
        | RestrictImageSize(width=width, height=height)
    )


def observations(episode: Episode, rate_hz: float, cameras: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
    """The episode as the harness hands it to the stack: one observation per control tick.

    Every signal the episode recorded goes in, so a declared stack finds whatever it asks for and the
    harness names none either. ``cameras`` keeps only those, for a flag-built stack: it stacks the
    cameras it was told about and forwards the rest at full size, which the wire then carries.
    """
    period_ns = int(1e9 / rate_hz)
    for ts in range(episode.start_ts, episode.last_ts + 1, period_ns):
        sample = dict(episode.time[ts])
        if cameras is not None:
            unasked = [key for key in sample if key.startswith(keys.IMAGE_PREFIX) and key not in cameras]
            for key in unasked:
                del sample[key]
        yield {**sample, keys.OBS_TIME_NS: ts, keys.WALL_TIME_NS: ts}


def capture(ticks: Iterable[dict[str, Any]], stack: Layer, model: Policy, requests: int) -> list[dict[str, Any]]:
    """Run the rig-side stack over ``ticks`` and collect the first ``requests`` payloads it sends."""
    wire = CapturingWire(model)
    session = stack.wrap(wire).new_session()
    try:
        for obs in ticks:
            session(obs, obs[keys.OBS_TIME_NS])
            if len(wire.sent) >= requests:
                break
    except KeyError as missing:
        raise ValueError(
            f'the stack asks for {missing.args[0]!r} and the episode does not record it, so this episode '
            f'cannot stand in for what the rig sends; replay one recorded on the rig the server serves'
        ) from missing
    finally:
        session.close()
    return wire.sent


def serve(pipeline) -> tuple[PolicyServer, threading.Thread, int]:
    """Serve ``pipeline`` on a free loopback port, and hand back what stops it."""
    server = PolicyServer(pipeline)
    ws = websocket_wire.WebsocketWire(server_wire.ServedHostPort('127.0.0.1', 0))
    ready = threading.Event()
    thread = threading.Thread(target=server.serve, args=([ws], ready.set), daemon=True)
    thread.start()
    if not ready.wait(timeout=30.0):
        raise RuntimeError('the probe server never came up')
    served = ws.served_address
    assert isinstance(served, server_wire.ServedHostPort), 'the probe server binds a port, not a socket'
    return server, thread, served.port


class Measured(NamedTuple):
    """What one run measures: an open session, the stack its requests cross, and the wire's own setting."""

    session: InferenceSession
    stack: Layer
    compress_images: bool
    target: str


@contextlib.contextmanager
def against_server(wire_name: str, host: str, port: int, model: str, query: str) -> Iterator[Measured]:
    """A session on the named server, running the stack and wire settings that server declares.

    ``wire_name`` selects the transport (``positronic_wire.registry.CLIENT_WIRES``), and ``model`` names
    the checkpoint — empty for the one the server pinned. ``AUTH_TOKEN_ENV`` gates a served endpoint; a
    server started by hand needs none.
    """
    headers = bearer_headers.instantiate() if os.environ.get(AUTH_TOKEN_ENV) else None
    client_wire = registry.client_wire(wire_name)
    address = wire.HostPortAddress(host, port, wire.session_path(model), query)
    session = InferenceClient(client_wire, address, headers=headers).new_session()
    try:
        meta = session.metadata
        target = client_wire.session_url(address)
        yield Measured(session, declared_stack(meta), bool(meta.get(offboard_keys.COMPRESS_IMAGES)), target)
    finally:
        session.close()


@contextlib.contextmanager
def against_loopback(stack: Layer, compress_images: bool, model: Policy) -> Iterator[Measured]:
    """A session on a server this process starts, serving ``model`` behind ``stack``."""
    server, thread, port = serve(stack | remote(compress_images=compress_images) | PolicySource(model))
    try:
        client_wire = WebsocketClientWire()
        address = wire.HostPortAddress('127.0.0.1', port, wire.SESSION_PATH, '')
        session = InferenceClient(client_wire, address).new_session()
        try:
            yield Measured(session, stack, compress_images, f'{client_wire.session_url(address)} (no model behind it)')
        finally:
            session.close()
    finally:
        server.shutdown()
        thread.join(timeout=10.0)


def replay(session: InferenceSession, payloads: list[dict[str, Any]], compress_images: bool) -> list[dict[str, float]]:
    """Send each payload, and report what its round trip cost beside what the server reports spending.

    One run tells an uplink the receiver would not drain from a wait the server spent inside its own
    span: the session's own ``send_ms``/``recv_ms`` ride along.
    """
    rows = []
    for obs in payloads:
        started = time.perf_counter()
        prepared = prepare_obs(obs, compress_images)
        encoded = time.perf_counter()
        # The same pack ``infer`` does next, timed on its own so the round trip below divides.
        message = protocol.serialise(prepared)
        if len(message) > wire.MAX_MESSAGE_BYTES:
            raise ValueError(
                f"a {len(message) / 2**20:.1f} MiB payload exceeds the server's "
                f'{wire.MAX_MESSAGE_BYTES // 2**20} MiB message '
                f'limit; lower --frames, --width or --height, or keep --compress_images'
            )
        packed = time.perf_counter()
        session.infer(prepared)
        answered = time.perf_counter()
        served = dict(session.served_timing)
        round_trip_ms = (answered - packed) * 1000.0
        pack_ms = (packed - encoded) * 1000.0
        rows.append({
            'wire_kib': len(message) / 1024.0,
            'prepare_ms': (encoded - started) * 1000.0,
            'pack_ms': pack_ms,
            'round_trip_ms': round_trip_ms,
            **dict(session.wire_timing),
            **served,
            # What the round trip spends outside the server's own span: the socket both ways, the server's
            # receive and encode around it, and this client's decode.
            'outside_served_ms': round_trip_ms - pack_ms - served.get(protocol.TIMING_SERVED, 0.0),
        })
    return rows


def report(rows: list[dict[str, float]]) -> str:
    columns = list(rows[0])
    label_width = max(len(column) for column in columns)
    lines = [f'{"":<{label_width}}  {"median":>9}  {"p95":>9}  {"max":>9}']
    for column in columns:
        values = np.array([row[column] for row in rows])
        lines.append(
            f'{column:<{label_width}}  {np.median(values):9.1f}  {np.percentile(values, 95):9.1f}  {values.max():9.1f}'
        )
    return '\n'.join(lines)


@cfn.config(
    dataset=positronic.cfg.ds.local,
    episode=0,
    requests=20,
    server_host=None,
    server_wire='websocket_tls',
    server_port=443,
    server_model='',
    server_query='',
    frames=25,
    rate_hz=15.0,
    width=1024,
    height=288,
    cameras=(keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE),
    chunk_rows=24,
    compress_images=True,
    out=None,
)
def main(
    dataset: Dataset,
    episode: int,
    requests: int,
    server_host: str | None,
    server_wire: str,
    server_port: int,
    server_model: str,
    server_query: str,
    frames: int,
    rate_hz: float,
    width: int,
    height: int,
    cameras: Sequence[str],
    chunk_rows: int,
    compress_images: bool,
    out: str | None,
):
    # configuronic hands the CLI token through as a string.
    out_path = Path(out) if out is not None else None
    model = InstantChunk(chunk_rows, 1.0 / rate_hz)

    chosen = dataset[episode]
    assert isinstance(chosen, Episode), 'name one episode, not a slice of them'

    opened = (
        against_server(server_wire, server_host, server_port, server_model, server_query)
        if server_host
        else against_loopback(rig_stack(cameras, frames, rate_hz, width, height), compress_images, model)
    )
    with opened as measured:
        print(f'stack: {json.dumps(measured.stack.to_spec())}')
        # The declared stack chooses for a named server; the flags do it here, so nothing unasked-for is sent.
        selected = None if server_host else cameras
        payloads = capture(observations(chosen, rate_hz, selected), measured.stack, model, requests)
        if not payloads:
            raise ValueError(f'episode {episode} is shorter than one {chunk_rows}-row chunk; nothing was sent')
        print(f'captured {len(payloads)} payload(s) off episode {episode}')
        replay(measured.session, payloads[:1], measured.compress_images)  # warm up, so no first touch is timed
        rows = replay(measured.session, payloads, measured.compress_images)

    source = 'declared by the server' if server_host else 'built from the flags'
    print(
        f'\n{len(rows)} requests against {measured.target}, stack {source}, '
        f'compress_images={measured.compress_images}\n'
    )
    print(report(rows))
    if out_path is not None:
        out_path.write_text(json.dumps(rows, indent=1))
        print(f'\nper-request rows -> {out_path}')


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
