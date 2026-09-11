"""Measure what one inference costs when the frames ride a shared-memory ring, and when they do not.

The server runs in this process on a Unix socket, and the served session reads every pixel it is
given, as a codec does. So each arm pays for touching the frames, and the boundary costs the
difference between them.

Usage
  uv run --locked python -m positronic.offboard.tests.bench_frame_ring
  uv run --locked python -m positronic.offboard.tests.bench_frame_ring --calls 100 --cameras 3
"""

import argparse
import asyncio
import statistics
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn

from positronic import keys
from positronic.offboard import client
from positronic.offboard import keys as offboard_keys
from positronic.offboard.client import InferenceClient
from positronic.offboard.server import PolicyServer
from positronic.policy import Policy, Session
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.spec import PolicySource, remote

HD720 = (720, 1280, 3)


class _ReadEveryPixel(Session):
    """A session that reads each frame once and answers, so both arms pay the same read."""

    def __call__(self, obs: Mapping[str, Any], time_ns: int) -> list[dict[str, Any]]:
        for value in obs.values():
            if isinstance(value, np.ndarray):
                int(value.sum())
        return [{keys.ACTION_TIMESTAMP: 0.0}]

    @property
    def meta(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


class _StubPolicy(Policy):
    def new_session(self, context=None, rt=None) -> Session:
        return _ReadEveryPixel()

    @property
    def functions(self):
        return {}

    def close(self) -> None:
        pass


def _serve(socket_path: str, frame_ring: bool) -> tuple[PolicyServer, uvicorn.Server, threading.Thread]:
    server = PolicyServer(
        ChunkedSchedule() | remote | PolicySource(_StubPolicy()), uds=socket_path, frame_ring=frame_ring
    )
    PolicyServer.claim_socket_path(socket_path)
    uv_server = uvicorn.Server(uvicorn.Config(server.app, uds=socket_path, log_level='error'))

    async def run():
        await server._startup()
        await uv_server.serve()

    thread = threading.Thread(target=asyncio.run, args=(run(),), daemon=True)
    thread.start()
    while not uv_server.started:
        time.sleep(0.02)
    return server, uv_server, thread


def _measure(socket_path: str, frame_ring: bool, calls: int, cameras: int) -> tuple[list[float], int, bool]:
    server, uv_server, thread = _serve(socket_path, frame_ring)
    sent: list[int] = []
    packer = client.serialise
    client.serialise = lambda obj: _record(packer(obj), sent)
    try:
        session = InferenceClient(f'unix://{socket_path}').new_session()
        declared = offboard_keys.FRAME_RING in session.metadata
        obs: dict[str, Any] = {
            f'image.{i}': np.random.default_rng(i).integers(0, 256, HD720, dtype=np.uint8) for i in range(cameras)
        }
        obs[keys.GRIP] = 0.5
        for _ in range(5):
            session.infer(obs)
        times = []
        for _ in range(calls):
            start = time.perf_counter()
            session.infer(obs)
            times.append((time.perf_counter() - start) * 1e3)
        message = max(sent)
        session.close()
    finally:
        client.serialise = packer
        uv_server.should_exit = True
        thread.join(timeout=5.0)
        if server._frames is not None:
            server._frames.close()
    return times, message, declared


def _record(message: bytes, sent: list[int]) -> bytes:
    sent.append(len(message))
    return message


def _report(name: str, times: list[float], message: int, declared: bool) -> None:
    ordered = sorted(times)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    print(
        f'{name:<12} median {statistics.median(times):7.2f} ms   p95 {p95:7.2f} ms   '
        f'min {ordered[0]:7.2f} ms   message {message / 1e6:6.2f} MB   declared {declared}'
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
            path = str(Path(directory) / f'{name}.sock')
            _report(name, *_measure(path, frame_ring, args.calls, args.cameras))


if __name__ == '__main__':
    main()
