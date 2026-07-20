"""Synchronous client for the env server: the lockstep ``reset``/``step``/``close`` round-trips.

Positronic-free (``websockets`` + the wire codec). ``RemoteEnvControlSystem`` wraps this as a pimm
control system; tests use it directly to compare a socket rollout against an in-process one.
"""

import time
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

# HACK: scratch pi-timing instrumentation (internal#55) — breaks this module's positronic-free claim; never merge.
from positronic.utils import timing_stats

from .protocol import decode, encode


class EnvConnection:
    """One websocket to an ``EnvServer``, opened with retry. ``reset``/``step`` block on the round-trip.

    There is no handshake: the first ``reset`` constructs the env server-side and returns
    ``{'obs', 'meta', 'control_dt'}``.

    The connect deadline must cover a first boot on a fresh machine: a heavy simulator can spend many minutes
    bringing its runtime up — compiling shaders, loading assets — before it binds the port.
    """

    def __init__(self, host: str, port: int, *, open_timeout: float = 10.0, connect_deadline: float = 1800.0):
        uri = f'ws://{host}:{port}/'
        deadline = time.monotonic() + connect_deadline
        backoff = 0.5
        while True:
            try:
                # Camera + full-state observations routinely exceed websockets' 1 MiB default frame size.
                # HACK (internal#55): ping_interval=None — a multi-minute Isaac scene build starves the
                # server's pong thread (native code holds the GIL), so the client's keepalive kills a healthy
                # connection mid-reset (observed: 10-min CleanUpToys->next build, close 1011). Never merge as-is.
                self._ws = connect(uri, open_timeout=open_timeout, max_size=None, ping_interval=None)
                break
            except (TimeoutError, OSError) as e:
                if time.monotonic() >= deadline:
                    raise type(e)(f'{e} (connecting to {host}:{port})') from e
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

    def reset(self, token: Any) -> dict[str, Any]:
        start = time.monotonic()
        result = self._request({'cmd': 'reset', 'token': token})
        timing_stats.record('env.reset', time.monotonic() - start)
        return result

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        result = self._request({'cmd': 'step', 'action': action})
        timing_stats.record('env.step', time.monotonic() - start)
        return result

    def _request(self, msg: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(encode(msg))
        result = decode(self._ws.recv())
        if 'error' in result:
            raise RuntimeError(f'env server: {result["error"]}')
        return result

    def close(self) -> None:
        # Best-effort: ask the server to release, but a peer that is already gone (a crashed or killed server) is
        # success too — the socket is closed regardless.
        try:
            self._ws.send(encode({'cmd': 'close'}))
            self._ws.recv()
        except ConnectionClosed:
            pass
        finally:
            self._ws.close()
