"""Measure what one inference costs when the frames ride a shared-memory ring, and when they do not.

The server runs in this process on a Unix socket, and the served session reads every pixel it is
given, as a codec does. So each arm pays for touching the frames, and the boundary costs the
difference between them.

Usage
  uv run --locked python -m positronic.offboard.tests.bench_frame_ring
  uv run --locked python -m positronic.offboard.tests.bench_frame_ring --calls 100 --cameras 3
"""

import argparse
import statistics
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from positronic_model_server import server_wire, websocket_wire
from positronic_wire import registry, wire

from positronic import keys
from positronic.offboard import client
from positronic.offboard import frame_ring as frames
from positronic.offboard.client import InferenceClient
from positronic.offboard.server import PolicyServer
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy.processors import ChunkedSchedule

HD720 = (720, 1280, 3)


class _ReadEveryPixel(Model):
    """A model that reads each frame once and answers."""

    def __call__(self, obs: Mapping[str, Any], *, session_id: str) -> list[dict[str, Any]]:
        for value in obs.values():
            if isinstance(value, np.ndarray):
                int(value.sum())
        return [{keys.GRIP: 0.0}]


def _serve(socket_path: Path, frame_ring: bool) -> tuple[PolicyServer, threading.Thread]:
    """A server on ``socket_path``, over the wire production serves."""
    deployment = PolicyDeployment(ChunkedSchedule(fps=15), compress_images=False)
    server = PolicyServer(_ReadEveryPixel, deployment, frame_ring=frame_ring)
    wires: list[server_wire.Wire] = [websocket_wire.WebsocketWire(websocket_wire.ServedUnixSocket(socket_path))]
    ready = threading.Event()
    thread = threading.Thread(target=server.serve, args=(wires, ready.set), daemon=True)
    thread.start()
    if not ready.wait(timeout=30.0):
        raise RuntimeError('the server did not bind')
    return server, thread


class _Measurement(NamedTuple):
    """One arm's milliseconds per call, its largest message in bytes, and whether the server declared a ring."""

    times_ms: list[float]
    message_bytes: int
    declared: bool


def _measure(socket_path: Path, frame_ring: bool, calls: int, cameras: int) -> _Measurement:
    server, thread = _serve(socket_path, frame_ring)
    sent: list[int] = []
    packer = client.serialise
    client.serialise = lambda obj: _record(packer(obj), sent)
    try:
        times_ms, declared = _time_calls(socket_path, calls, cameras)
    finally:
        client.serialise = packer
        server.shutdown()
        thread.join(timeout=10.0)
    return _Measurement(times_ms, max(sent), declared)


def _time_calls(socket_path: Path, calls: int, cameras: int) -> tuple[list[float], bool]:
    """Milliseconds per call over one session on ``socket_path``, after a warm-up, and whether it declared a ring."""
    address = wire.UnixSocketAddress(socket_path, wire.SESSION_PATH, '')
    session = InferenceClient(registry.client_wire('websocket_unix'), address).new_session()
    declared = frames.HANDSHAKE_KEY in session.metadata
    obs: dict[str, Any] = {
        f'image.{i}': np.random.default_rng(i).integers(0, 256, HD720, dtype=np.uint8) for i in range(cameras)
    }
    obs[keys.GRIP] = 0.5
    for _ in range(5):
        session.infer(obs)
    times_ms = []
    for _ in range(calls):
        start = time.perf_counter()
        session.infer(obs)
        times_ms.append((time.perf_counter() - start) * 1e3)
    session.close()
    return times_ms, declared


def _record(message: bytes, sent: list[int]) -> bytes:
    sent.append(len(message))
    return message


def _report(name: str, measured: _Measurement) -> None:
    ordered = sorted(measured.times_ms)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    print(
        f'{name:<12} median {statistics.median(ordered):7.2f} ms   p95 {p95:7.2f} ms   '
        f'min {ordered[0]:7.2f} ms   message {measured.message_bytes / 1e6:6.2f} MB   declared {measured.declared}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calls', type=int, default=50)
    parser.add_argument('--cameras', type=int, default=3)
    args = parser.parse_args()

    frame = np.prod(HD720) * args.cameras
    print(f'{args.cameras} frames of {HD720} per inference, {frame / 1e6:.1f} MB raw, {args.calls} calls')
    with tempfile.TemporaryDirectory(dir='/tmp') as directory:
        for name, frame_ring in (('message', False), ('ring', True)):
            path = Path(directory) / f'{name}.sock'
            _report(name, _measure(path, frame_ring, args.calls, args.cameras))


if __name__ == '__main__':
    main()
