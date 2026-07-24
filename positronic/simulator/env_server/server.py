"""The dumb remote env-server: a benchmark env behind ``reset``/``step``/``close`` over websockets.

Positronic-free by contract — it depends only on ``websockets``, plus the wire codec and the
``EnvProtocol`` an env implements. A benchmark runs this in its own isolated interpreter (where
positronic can't be installed); the native fixture runs it in-process against ``MujocoSim``. The
server is lockstep request-response — one client, one outstanding request — so the World's virtual
clock advances unchanged while a step round-trips.

There is no build phase: ``reset`` constructs (or reuses a cached) env from its opaque token and
re-randomizes it; ``control_dt`` rides every observation (``reset`` and each ``step``), so a benchmark
may even vary its control period per step. Heavy construction is the env's own concern (cache it).

Protocol (msgpack frames, see ``protocol``):
  client ``{'cmd': 'reset', 'token': ...}``   -> server ``{'obs', 'meta', 'robot_meta', 'control_dt'}``
  client ``{'cmd': 'step', 'action': {...}}`` -> server ``{'obs', 'done', 'control_dt'}``
  client ``{'cmd': 'close'}``                 -> server ``{'ok': True}``
Any command whose handling raises returns ``{'error': str}`` instead, which the client re-raises.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from websockets.sync.server import ServerConnection, serve

# This module + ``protocol`` copy into a benchmark's isolated interpreter as a self-contained unit
# (importing ``positronic.*`` would run the package's installed-version lookup and fail there).
# Relative when they land as a package, top-level when copied in flat.
try:
    from .protocol import decode, encode
except ImportError:
    from protocol import decode, encode

logger = logging.getLogger(__name__)

# Float slack tolerated on the phase-sum invariant: the measured spans and the whole-step wall are separate
# ``perf_counter`` reads, so their difference can dip a hair below zero without any real double-count.
_PHASE_SLACK_S = 1e-6


def disjoint_step_phases(wall_s: float, *, physics_s: float, render_s: float) -> dict[str, float]:
    """An env server's disjoint step decomposition summing to ``wall_s``.

    ``physics_s`` and ``render_s`` are the measured native phases; ``server_other_s`` is the residual — the
    step wall outside them (managers, IK, plumbing). The three are additive by construction, so a client can
    normalise each against the step wall. Raises when the measured phases exceed the whole-step wall (a phase
    timed inside another — e.g. rendering inside the physics loop — double-counts that wall), which would make
    the residual negative and the split untrustworthy; a sub-nanosecond float dip is clamped, not raised.
    """
    server_other_s = wall_s - physics_s - render_s
    if server_other_s < -_PHASE_SLACK_S:
        raise ValueError(
            f'env step wall {wall_s:.6f}s is below physics {physics_s:.6f}s + render {render_s:.6f}s: a phase '
            f'is double-counted (timed inside another), so the step decomposition is not disjoint'
        )
    return {'physics_s': physics_s, 'render_s': render_s, 'server_other_s': max(server_other_s, 0.0)}


class EnvProtocol(ABC):
    """A benchmark env behind the three methods the server exposes; the wire contract, positronic-free.

    ``reset`` and ``step`` exchange raw plain-data dicts (numpy arrays + scalars) — the canonical<->raw
    mapping is the client's ``EnvAdapter``, never the server's. Heavy, per-task construction is the
    env's own concern (cache it, keyed by the token's structural part); the protocol has no build phase.
    """

    @abstractmethod
    def reset(self, token: Any) -> dict[str, Any]:
        """Construct (cached) + re-randomize from a token; returns ``obs``, ``meta``, ``robot_meta``, ``control_dt``.

        ``control_dt`` is the env's control period: the client paces one ``step`` per ``control_dt`` and advances
        the World's virtual clock by it each step. ``meta`` is the scene identity the policy reads its instruction
        from (the language goal, scene ids); ``robot_meta`` is the robot model identity (URDF / joint names /
        control frame) recorded into the episode. Either is ``{}`` when the client owns that side — a static
        instruction, or an embodiment that ships its own model.
        """

    @abstractmethod
    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply a raw action for one control period; returns ``{'obs', 'done', 'control_dt'}``.

        ``control_dt`` is re-reported every step (the wait until the next one), so it can vary.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the env's resources."""


class EnvServer:
    """Serves one ``EnvProtocol`` over a synchronous websocket — one client per server, lockstep.

    The single client is accepted and served on the thread that calls ``serve_forever`` (the subprocess main
    thread), not a per-connection websocket thread: a render backend can be thread-affine — macOS GLFW must
    initialize on the main thread — and a sim is single-threaded anyway. The env lives for the server's
    lifetime; ``shutdown`` releases it.
    """

    def __init__(self, env: EnvProtocol, host: str, port: int):
        self._env = env
        self._host = host
        self._port = port
        self._server = None
        self._served = False
        self._shutdown = False

    def _handle(self, connection: ServerConnection) -> None:
        # Reaching here means a client completed the websocket handshake: it is the one client this server
        # serves, so ``serve_forever`` exits once this returns. A failure handling a command (a rejected action,
        # a sim blow-up, an unknown command) crosses back as an error frame the client re-raises, rather than
        # killing the connection.
        self._served = True
        for raw in connection:
            msg = decode(raw)
            try:
                match msg['cmd']:
                    case 'close':
                        connection.send(encode({'ok': True}))
                        return
                    case 'reset':
                        result = self._env.reset(msg['token'])
                    case 'step':
                        result = self._env.step(msg['action'])
                    case other:
                        raise ValueError(f'Unknown command: {other!r}')
            except Exception as e:
                result = {'error': f'{type(e).__name__}: {e}'}
            connection.send(encode(result))

    def serve_forever(self) -> None:
        # A reset token can be a large opaque blob (e.g. an exact-replay scene), so don't cap frame size.
        # ``serve`` would spawn a thread per connection; instead the accept loop runs here and calls the
        # handshake + ``_handle`` inline, so the env runs on this thread (the subprocess main thread — macOS GLFW
        # must init there). The loop skips bare TCP probes (which never handshake) and exits once the one client
        # has been served.
        with serve(self._handle, self._host, self._port, max_size=None) as server:
            self._server = server
            # Time out ``accept`` so the loop periodically observes ``shutdown`` even with no client connecting —
            # closing the listening socket from another thread does not reliably wake a blocking ``accept``.
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
        # Stop accepting (the flag breaks the timed accept loop) and release the env.
        self._shutdown = True
        if self._server is not None:
            self._server.shutdown()
        self._env.close()
