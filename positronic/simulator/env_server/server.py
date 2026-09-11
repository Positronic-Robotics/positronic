"""Synchronous env server for one client with one outstanding request.

This module must work in an isolated interpreter without Positronic installed.

Protocol (msgpack frames, see ``protocol``):
  client ``{'cmd': 'tasks', 'spec': ...}``    -> server ``{'tasks': [{...}, ...]}``
  client ``{'cmd': 'reset', 'token': ...}``   -> server ``{'obs', 'meta', 'robot_meta', 'control_dt', 'horizon'?}``
  client ``{'cmd': 'step', 'action': {...}}`` -> server ``{'obs', 'done', 'control_dt'}``
  client ``{'cmd': 'close'}``                 -> server ``{'ok': True}``
Command handling failures return ``{'error': str}`` without closing the session; the client re-raises them.

``control_dt`` is the control period in seconds and can vary per step.
``meta`` identifies the scene; ``robot_meta`` identifies the robot model.
Either metadata dict can be empty when the client supplies that information.
"""

from abc import ABC, abstractmethod
from typing import Any

from websockets.sync.server import ServerConnection, serve

# Isolated interpreters can import these modules as a package or as flat files.
try:
    from . import protocol
except ImportError:
    import protocol


class EnvProtocol(ABC):
    """An environment exchanging raw arrays and plain data with no Positronic dependencies.

    Canonical observation and command conversion belongs in the client's ``EnvAdapter``.
    Implementations own environment construction and caching across resets.
    """

    @abstractmethod
    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Task records selected by ``spec``, each with a ``name`` field.

        An absent key leaves that axis unrestricted; an unknown value raises.
        """

    @abstractmethod
    def reset(self, token: Any) -> dict[str, Any]:
        """Construct or reuse the environment and re-randomize it from an opaque token.

        Return ``obs``, scene ``meta``, ``robot_meta``, and ``control_dt`` in seconds.
        Either metadata dict may be empty when the client supplies it.

        ``horizon`` (optional) is the sim-enforced episode deadline in sim-seconds — the env's own time limit,
        which it delivers as a terminal ``done`` on expiry. It is reported for observability, so a run can be
        checked against the horizon the env actually resolved; omit it when the env enforces none.
        """

    @abstractmethod
    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply a raw action for one control period; return ``obs``, ``done``, and ``control_dt``.

        ``control_dt`` is the wait until the next step and may vary each step.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the env's resources."""


class EnvServer:
    """Serve one client on the calling thread; ``shutdown`` releases the owned environment.

    ``serve_forever`` ends when that client's session ends or shutdown is requested.
    """

    def __init__(self, env: EnvProtocol, host: str, port: int):
        self._env = env
        self._host = host
        self._port = port
        self._server = None
        self._served = False
        self._shutdown = False

    def _handle(self, connection: ServerConnection) -> None:
        self._served = True
        for raw in connection:
            msg = protocol.decode(raw)
            try:
                match protocol.Command(msg[protocol.CMD]):
                    case protocol.Command.CLOSE:
                        connection.send(protocol.encode({protocol.OK: True}))
                        return
                    case protocol.Command.TASKS:
                        result = {protocol.TASKS: self._env.tasks(msg[protocol.SPEC])}
                    case protocol.Command.RESET:
                        result = self._env.reset(msg[protocol.TOKEN])
                    case protocol.Command.STEP:
                        result = self._env.step(msg[protocol.ACTION])
            except Exception as e:
                result = {protocol.ERROR: f'{type(e).__name__}: {e}'}
            connection.send(protocol.encode(result))

    def serve_forever(self) -> None:
        # Reset tokens can exceed the default frame size.
        # Accept and handle connections inline: macOS GLFW requires environment calls on the main thread.
        # Bare TCP probes do not count as the one websocket session.
        with serve(self._handle, self._host, self._port, max_size=None) as server:
            self._server = server
            # Closing the socket from another thread does not reliably wake a blocking accept.
            server.socket.settimeout(0.5)
            while not self._served and not self._shutdown:
                try:
                    sock, addr = server.socket.accept()
                except TimeoutError:
                    continue
                except OSError:
                    return  # ``shutdown`` closed the listening socket
                sock.settimeout(None)  # the accepted socket runs the long-lived session in blocking mode
                server.handler(sock, addr)

    def shutdown(self) -> None:
        self._shutdown = True
        if self._server is not None:
            self._server.shutdown()
        self._env.close()
