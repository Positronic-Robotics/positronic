"""Synchronous env server for one client with one outstanding request.

This module must work in an isolated interpreter without Positronic installed.

Protocol (msgpack frames, see ``protocol``):
  client ``{'cmd': 'tasks', 'spec': ...}``    -> server ``{'tasks': [{...}, ...]}``
  client ``{'cmd': 'reset', 'token': ...}``   -> server ``{'obs', 'meta', 'robot_meta', 'control_dt'}``
  client ``{'cmd': 'step', 'action': {...}}`` -> server ``{'obs', 'done', 'control_dt'}``
  client ``{'cmd': 'close'}``                 -> server ``{'ok': True}``
Any command whose handling raises returns ``{'error': str}`` instead, which the client re-raises.

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
    """An environment with task selection, reset, step, and cleanup operations."""

    @abstractmethod
    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Task records selected by ``spec``, each with a ``name`` field.

        An absent key leaves that axis unrestricted; an unknown value raises.
        """

    @abstractmethod
    def reset(self, token: Any) -> dict[str, Any]:
        """The initial observation and metadata for a reset token."""

    @abstractmethod
    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """The observation and terminal state after one action period."""

    @abstractmethod
    def close(self) -> None:
        """Release the env's resources."""


class EnvServer:
    """A synchronous websocket server that owns one environment."""

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
        # macOS GLFW requires environment calls on the main thread.
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
