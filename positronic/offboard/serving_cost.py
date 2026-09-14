"""What one inference costs the serving path itself, with no model behind it.

Replays a recorded episode against a real ``PolicyServer`` on loopback whose model answers a fixed
chunk instantly, so every millisecond reported is serving cost, divided by the phases the server
reports. The default stack is the one a DROID endpoint declares: 25 frames of two cameras and the
arm's pose, bounded to 1024x288, JPEG-encoded per frame and re-queried every 24 rows.

Usage
    uv run --locked python -m positronic.offboard.serving_cost \\
        --dataset.path=<episode root> --requests=20
    ... --compress_images=False           # send raw stacks instead of per-frame JPEG
    ... --frames=25 --rate_hz=15 --width=1024 --height=288 --chunk_rows=24 --out=rows.json
"""

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
import pos3
import uvicorn

import positronic.cfg.ds
from pimm.logging import init_logging
from positronic import keys
from positronic.dataset.dataset import Dataset
from positronic.dataset.episode import Episode
from positronic.offboard import protocol
from positronic.offboard.client import InferenceClient, InferenceSession
from positronic.offboard.server import WS_IMPL, PolicyServer
from positronic.policy.base import DelegatingPolicy, DelegatingSession, Layer, Policy, Session
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, StopOnFault, TemporalStack
from positronic.policy.remote import prepare_obs
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
    """The rig-side half a DROID endpoint declares, with the stack depth and image bound the caller names."""
    offsets = tuple(-(frames - 1 - step) / rate_hz for step in range(frames))
    stacked = (*cameras, keys.EE_POSE, keys.GRIP)
    return (
        StopOnFault()
        | TemporalStack(stacked, offsets)
        | ChunkedSchedule()
        | RestrictImageSize(width=width, height=height)
    )


# What a rig observation carries beside its cameras. The episode holds much more — the arm's URDF, its
# meshes, every recorded command — and none of that crosses the wire.
STATE_KEYS = (keys.JOINTS, keys.JOINT_VEL, keys.EE_POSE, keys.GRIP, keys.ROBOT_STATUS)


def observations(episode: Episode, cameras: Sequence[str], rate_hz: float) -> Iterator[dict[str, Any]]:
    """The episode as the harness hands it to the stack: one observation per control tick."""
    period_ns = int(1e9 / rate_hz)
    for ts in range(episode.start_ts, episode.last_ts + 1, period_ns):
        sample = episode.time[ts]
        obs = {key: sample[key] for key in (*STATE_KEYS, *cameras) if key in sample}
        if keys.TASK in sample:
            obs[keys.TASK] = sample[keys.TASK]
        yield {**obs, keys.OBS_TIME_NS: ts, keys.WALL_TIME_NS: ts}


def capture(ticks: Iterable[dict[str, Any]], stack: Layer, model: Policy, requests: int) -> list[dict[str, Any]]:
    """Run the rig-side stack over ``ticks`` and collect the first ``requests`` payloads it sends."""
    wire = CapturingWire(model)
    session = stack.wrap(wire).new_session()
    try:
        for obs in ticks:
            session(obs, obs[keys.OBS_TIME_NS])
            if len(wire.sent) >= requests:
                break
    finally:
        session.close()
    return wire.sent


def serve(pipeline) -> tuple[uvicorn.Server, threading.Thread, int]:
    """Serve ``pipeline`` on a free loopback port, and hand back what stops it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as free_port:
        free_port.bind(('127.0.0.1', 0))
        port = free_port.getsockname()[1]
    policy_server = PolicyServer(pipeline, host='127.0.0.1', port=port)
    served = uvicorn.Server(
        uvicorn.Config(policy_server.app, host='127.0.0.1', port=port, log_level='warning', ws=WS_IMPL)
    )

    async def run():
        await policy_server._startup()
        await served.serve()

    thread = threading.Thread(target=asyncio.run, args=(run(),), daemon=True)
    thread.start()
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.2):
                return served, thread, port
        except OSError:
            time.sleep(0.05)
    raise RuntimeError('the probe server never came up')


def replay(session: InferenceSession, payloads: list[dict[str, Any]], compress_images: bool) -> list[dict[str, float]]:
    """Send each payload, and report what its round trip cost beside what the server reports spending."""
    rows = []
    for obs in payloads:
        started = time.perf_counter()
        prepared = prepare_obs(obs, compress_images)
        encoded = time.perf_counter()
        # The same pack ``infer`` does next, timed on its own so the round trip below divides.
        message = protocol.serialise(prepared)
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
            **served,
            # What the round trip spends outside the server's own span: the socket both ways, and the
            # websocket receive the server pays before ``served_ms`` opens.
            'transfer_ms': round_trip_ms - pack_ms - served.get(protocol.TIMING_SERVED, 0.0),
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
    frames: int,
    rate_hz: float,
    width: int,
    height: int,
    cameras: Sequence[str],
    chunk_rows: int,
    compress_images: bool,
    out: str | None,
):
    model = InstantChunk(chunk_rows, 1.0 / rate_hz)
    stack = rig_stack(cameras, frames, rate_hz, width, height)

    chosen = dataset[episode]
    assert isinstance(chosen, Episode), 'name one episode, not a slice of them'
    payloads = capture(observations(chosen, cameras, rate_hz), stack, model, requests)
    if not payloads:
        raise ValueError(f'episode {episode} is shorter than one {chunk_rows}-row chunk; nothing was sent')
    print(f'captured {len(payloads)} payload(s) off episode {episode}')

    served, thread, port = serve(stack | remote(compress_images=compress_images) | PolicySource(model))
    try:
        session = InferenceClient(f'ws://127.0.0.1:{port}').new_session()
        try:
            replay(session, payloads[:1], compress_images)  # warm up, so no first touch is timed
            rows = replay(session, payloads, compress_images)
        finally:
            session.close()
    finally:
        served.should_exit = True
        thread.join(timeout=10.0)

    print(
        f'\n{len(rows)} requests, {frames} frames x {len(cameras)} cameras, bound {width}x{height}, '
        f'compress_images={compress_images}, no model behind the server\n'
    )
    print(report(rows))
    if out is not None:
        Path(out).write_text(json.dumps(rows, indent=1))
        print(f'\nper-request rows -> {out}')


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
