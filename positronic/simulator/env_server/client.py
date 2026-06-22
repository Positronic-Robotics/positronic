"""Synchronous client for the env server: the lockstep ``reset``/``step``/``close`` round-trips.

Positronic-free (``websockets`` + the wire codec). ``RemoteEnvControlSystem`` wraps this as a pimm
control system; tests use it directly to compare a socket rollout against an in-process one.
"""

import time
from typing import Any

from websockets.sync.client import connect

from .protocol import decode, encode


class EnvConnection:
    """One websocket to an ``EnvServer``, opened with retry. ``reset``/``step`` block on the round-trip.

    There is no handshake: the first ``reset`` constructs the env server-side and returns
    ``{'obs', 'meta', 'control_dt'}``.
    """

    def __init__(self, host: str, port: int, *, open_timeout: float = 10.0, connect_deadline: float = 60.0):
        uri = f'ws://{host}:{port}/'
        deadline = time.monotonic() + connect_deadline
        backoff = 0.5
        while True:
            try:
                # Camera + full-state observations routinely exceed websockets' 1 MiB default frame size.
                self._ws = connect(uri, open_timeout=open_timeout, max_size=None)
                break
            except (TimeoutError, OSError) as e:
                if time.monotonic() >= deadline:
                    raise type(e)(f'{e} (connecting to {host}:{port})') from e
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

    def reset(self, token: Any) -> dict[str, Any]:
        return self._request({'cmd': 'reset', 'token': token})

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._request({'cmd': 'step', 'action': action})

    def _request(self, msg: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(encode(msg))
        result = decode(self._ws.recv())
        if 'error' in result:
            raise RuntimeError(f'env server: {result["error"]}')
        return result

    def close(self) -> None:
        try:
            self._ws.send(encode({'cmd': 'close'}))
            self._ws.recv()
        finally:
            self._ws.close()
