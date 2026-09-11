"""Synchronous env-server client with no dependencies on Positronic."""

import logging
import time
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from . import protocol

logger = logging.getLogger(__name__)

# Bound cleanup time when the server keeps the socket open without answering.
_CLOSE_ACK_TIMEOUT = 5.0


class EnvConnection:
    """Connect to an ``EnvServer`` with retry; each command blocks for its response.

    Requests need no application handshake; ``reset`` returns the initial scene frame.
    The connect deadline must cover simulator startup, which can take many minutes on a fresh machine.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        open_timeout: float = 10.0,
        connect_deadline: float = 1800.0,
        ping_timeout: float = 600.0,
    ):
        uri = f'ws://{host}:{port}/'
        deadline = time.monotonic() + connect_deadline
        backoff = 0.5
        while True:
            try:
                # Camera + full-state observations routinely exceed websockets' 1 MiB default frame size.
                # Native scene loading can block the server's heartbeat replies for minutes.
                self._ws = connect(uri, open_timeout=open_timeout, max_size=None, ping_timeout=ping_timeout)
                break
            except (TimeoutError, OSError) as e:
                if time.monotonic() >= deadline:
                    raise type(e)(f'{e} (connecting to {host}:{port})') from e
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

    def tasks(self, spec: Any) -> list[dict[str, Any]]:
        return self._request({protocol.CMD: protocol.Command.TASKS.value, protocol.SPEC: spec})[protocol.TASKS]

    def reset(self, token: Any) -> dict[str, Any]:
        return self._request({protocol.CMD: protocol.Command.RESET.value, protocol.TOKEN: token})

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._request({protocol.CMD: protocol.Command.STEP.value, protocol.ACTION: action})

    def _request(self, msg: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(protocol.encode(msg))
        result = protocol.decode(self._ws.recv())
        if protocol.ERROR in result:
            raise RuntimeError(f'env server: {result[protocol.ERROR]}')
        return result

    def close(self) -> None:
        try:
            self._ws.send(protocol.encode({protocol.CMD: protocol.Command.CLOSE.value}))
            self._ws.recv(timeout=_CLOSE_ACK_TIMEOUT)
        except ConnectionClosed:
            pass  # A closed peer cannot acknowledge the close request.
        except TimeoutError:
            logger.error('Env server did not acknowledge close within %.1fs; abandoning it', _CLOSE_ACK_TIMEOUT)
        finally:
            self._ws.close()
