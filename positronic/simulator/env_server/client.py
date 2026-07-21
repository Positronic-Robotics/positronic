"""Synchronous client for the env server: the lockstep ``reset``/``step``/``close`` round-trips.

Positronic-free (``websockets`` + the wire codec). ``RemoteEnvControlSystem`` wraps this as a pimm
control system; tests use it directly to compare a socket rollout against an in-process one.
"""

import time
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from .protocol import decode, encode


class EnvConnection:
    """One websocket to an ``EnvServer``, opened with retry. ``reset``/``step`` block on the round-trip.

    There is no handshake: the first ``reset`` constructs the env server-side and returns
    ``{'obs', 'meta', 'control_dt'}``.

    The connect deadline must cover a first boot on a fresh machine: a heavy simulator can spend many minutes
    bringing its runtime up — compiling shaders, loading assets — before it binds the port.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        open_timeout: float = 10.0,
        connect_deadline: float = 1800.0,
        step_timeout: float = 120.0,
        reset_timeout: float = 900.0,
    ):
        uri = f'ws://{host}:{port}/'
        # A single step is sub-second; a reset builds the scene (a RoboLab Isaac build runs into the minutes).
        # Each bound is generous over its legitimate case, so it fires only on a wedged server, not a slow one.
        self._step_timeout = step_timeout
        self._reset_timeout = reset_timeout
        deadline = time.monotonic() + connect_deadline
        backoff = 0.5
        while True:
            try:
                # Camera + full-state observations routinely exceed websockets' 1 MiB default frame size.
                # Keepalive stays off: a multi-minute Isaac scene build holds the server's GIL in native code,
                # starving its pong thread, so a client ping would kill a healthy connection mid-reset (observed
                # as a 1011 close during a 10-minute RoboLab scene build). A wedged (alive but unresponsive)
                # server is bounded instead per request by ``_request``'s recv timeout, and the connection's
                # own liveness by the connect deadline.
                self._ws = connect(uri, open_timeout=open_timeout, max_size=None, ping_interval=None)
                break
            except (TimeoutError, OSError) as e:
                if time.monotonic() >= deadline:
                    raise type(e)(f'{e} (connecting to {host}:{port})') from e
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

    def reset(self, token: Any) -> dict[str, Any]:
        return self._request({'cmd': 'reset', 'token': token}, self._reset_timeout)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._request({'cmd': 'step', 'action': action}, self._step_timeout)

    def _request(self, msg: dict[str, Any], timeout: float) -> dict[str, Any]:
        self._ws.send(encode(msg))
        try:
            raw = self._ws.recv(timeout=timeout)
        except TimeoutError as e:
            raise TimeoutError(f'env server did not respond to {msg["cmd"]!r} within {timeout:.0f}s') from e
        result = decode(raw)
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
