"""Synchronous client for the env server: the lockstep ``tasks``/``reset``/``step``/``close`` round-trips.

Positronic-free (``websockets`` + the wire codec). ``RemoteEnvControlSystem`` wraps this as a pimm
control system; tests use it directly to compare a socket rollout against an in-process one.
"""

import logging
import time
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from . import protocol
from .protocol import decode, encode

logger = logging.getLogger(__name__)

# How long ``close`` waits to be acknowledged. Teardown often runs while the peer is on its way out, and a
# simulator wedged in its own destructor holds the socket open without ever answering — an unbounded wait there
# hangs the run in place of ending it.
_CLOSE_ACK_TIMEOUT = 5.0


class EnvConnection:
    """One websocket to an ``EnvServer``, opened with retry. Every command blocks on the round-trip.

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
                self._ws = connect(uri, open_timeout=open_timeout, max_size=None)
                break
            except (TimeoutError, OSError) as e:
                if time.monotonic() >= deadline:
                    raise type(e)(f'{e} (connecting to {host}:{port})') from e
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

    def tasks(self, spec: Any) -> list[dict[str, Any]]:
        req = {protocol.REQUEST_CMD: protocol.CMD_TASKS, protocol.REQUEST_SPEC: spec}
        return self._request(req)[protocol.RESPONSE_TASKS]

    def reset(self, token: Any) -> dict[str, Any]:
        return self._request({protocol.REQUEST_CMD: protocol.CMD_RESET, protocol.REQUEST_TOKEN: token})

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._request({protocol.REQUEST_CMD: protocol.CMD_STEP, protocol.REQUEST_ACTION: action})

    def _request(self, msg: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(encode(msg))
        result = decode(self._ws.recv())
        if protocol.RESPONSE_ERROR in result:
            raise RuntimeError(f'env server: {result[protocol.RESPONSE_ERROR]}')
        return result

    def close(self) -> None:
        try:
            self._ws.send(encode({protocol.REQUEST_CMD: protocol.CMD_CLOSE}))
            self._ws.recv(timeout=_CLOSE_ACK_TIMEOUT)
        except ConnectionClosed:
            pass  # a peer already gone has released whatever the acknowledgement would have reported
        except TimeoutError:
            # Abandoning it is still better than hanging the run here, but a server that took the request and
            # never answered is wedged rather than finished, and its resources are nobody's to reclaim now.
            logger.error('Env server did not acknowledge close within %.1fs; abandoning it', _CLOSE_ACK_TIMEOUT)
        finally:
            self._ws.close()
