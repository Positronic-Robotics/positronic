"""Synchronous env server for one client with one outstanding request.

This module must work in an isolated interpreter without Positronic installed.

Protocol (msgpack frames, see ``protocol``):
  client ``{'cmd': 'tasks', 'spec': ...}``      -> server ``{'tasks': [{...}, ...]}``
  client ``{'cmd': 'reset', 'token': ...}``     -> server ``{'slots': [{'obs'}], 'meta', 'robot_meta',
                                                            'control_dt'}``
  client ``{'cmd': 'step', 'actions': [{...}]}``-> server ``{'slots': [{'obs', 'done'}], 'control_dt'}``
  client ``{'cmd': 'close'}``                   -> server ``{'ok': True}``
Command handling failures return ``{'error': str}`` without closing the session; the client re-raises them.

An env serves a fixed number of slots — independent episodes it steps together. Most serve one; a benchmark
that clones its scene serves several.

* ``slots`` carries one entry per slot, in slot order. An episode's own fields live inside its entry:
  ``obs`` from a reset, and ``obs``, ``done`` and ``success`` from a step.
* ``meta``, ``robot_meta`` and ``control_dt`` sit beside ``slots`` and describe the whole batch.
* ``actions`` is the mirror of ``slots``: one action per slot, same order.
* A client reads the slot count off the width of ``slots``; nothing announces it.

``control_dt`` is the control period in seconds and can vary per step.
``meta`` identifies the scene; ``robot_meta`` identifies the robot model.
Either metadata dict can be empty when the client supplies that information.
"""

from abc import ABC, abstractmethod
from typing import Any

from websockets.sync.server import ServerConnection, serve

# Isolated interpreters can import these modules as a package or as flat files.
if __package__:
    from . import protocol
else:
    import protocol


class EnvProtocol(ABC):
    """An environment exchanging raw arrays and plain data with no Positronic dependencies.

    Implementations convert wire commands to native actions and own environment construction.
    """

    @abstractmethod
    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Task records selected by ``spec``, each with a ``name`` field.

        An absent key leaves that axis unrestricted; an unknown value raises.
        """

    @abstractmethod
    def reset(self, token: Any) -> dict[str, Any]:
        """Construct or reuse the environment and re-randomize it from an opaque token.

        Return ``slots`` (one ``{'obs': ...}`` per slot), scene ``meta``, ``robot_meta``, and ``control_dt``
        in seconds. Either metadata dict may be empty when the client supplies it.
        """

    @abstractmethod
    def step(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply one raw action per slot for one control period; return ``slots`` and ``control_dt``.

        ``actions`` carries one entry per slot, in slot order, and ``slots`` answers with one ``{'obs',
        'done'}`` each. An env whose slot count differs from the actions it is handed raises.
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
                        result = self._env.step(msg[protocol.ACTIONS])
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
