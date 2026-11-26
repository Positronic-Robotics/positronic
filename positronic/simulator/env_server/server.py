"""Synchronous env server: one client per slot, and every slot stepped in one env call.

This module must work in an isolated interpreter without Positronic installed.

Protocol (msgpack frames, see ``protocol``), between a client and its slot:
  client ``{'cmd': 'tasks', 'spec': ...}``    -> server ``{'tasks': [{...}, ...]}``
  client ``{'cmd': 'reset', 'token': ...}``   -> server ``{'obs', 'meta', 'robot_meta', 'control_dt'}``, opt. ``goal``
    (a reset can also carry ``'batch_size': n`` and ``'refill': True``)
  client ``{'cmd': 'step', 'action': {...}}`` -> server ``{'obs', 'done', 'control_dt'}``, opt. ``goal``
  client ``{'cmd': 'close'}``                 -> server ``{'ok': True}``
Command handling failures return ``{'error': str}`` without closing the session; the client re-raises them.

``control_dt`` is the control period in seconds and can vary per step.
``meta`` identifies the scene; ``robot_meta`` identifies the robot model. Either metadata dict can be empty when the
client supplies that information. A task record's ``instruction`` is the text the policy gets for that task. A frame
carries ``goal`` when the env moves the goal away from that text, and a later ``goal`` replaces an earlier one.

An env serves at most ``num_slots`` slots: independent episodes that one env call steps together. Most envs serve
one. Each client that connects takes the next free slot and drives only that slot, so N clients can run N different
policies in one env:

* The slots that reset together are a batch. All of them reset with one token, and the env resets every slot.
* The first reset of a batch carries the batch size, at most ``num_slots``. A reset with no batch size asks for
  ``num_slots``. The env resets for a batch of that size, and each slot of the batch drives one env slot.
* A batch's first reset waits for as many slots as the batch size, but not longer than ``join_deadline`` seconds
  after the first slot asks while no batch runs. An env slot that no client takes by then sits out the batch, and
  the env gets no action for it. A slot that asks for another batch size than the slots that wait is refused. A
  slot past the batch size waits for the next batch.
* A later reset of a batch answers when every slot of the batch has asked for one or has left. A batch whose reset
  failed stays a batch, so its retry is a later reset.
* A step answers when every slot in an episode has sent its action. The env gets no action for the other slots.
  A slot that reports ``done`` stops holding the others back; a client may still step it until its next reset.
* A reset that carries ``refill`` resets its slot alone into the batch that runs, between two steps, and answers at
  once. Its token must be the token of the batch, and its batch size, if it gives one, the size of the batch. A slot
  of the batch takes its own env slot again; a slot outside the batch takes an env slot that no slot of the batch
  holds. The env must reset one slot alone (``EnvProtocol.reset_slot``). A refill is refused while a slot waits to
  join the next batch, so that the batch ends for it.
* A slot leaves when its client closes or disconnects, also while it waits on an answer, and a new client can
  then take it. A slot outside the batch that asks for a reset waits until every slot of the batch has left, and
  then joins the next batch.
* The server serves batch after batch until ``shutdown``.
"""

import inspect
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from websockets.protocol import State
from websockets.sync.server import ServerConnection, serve

if TYPE_CHECKING:
    # websockets 12 names this class otherwise.
    from websockets.sync.server import Server

# Isolated interpreters can import these modules as a package or as flat files.
if __package__:
    from . import protocol
else:
    import protocol

# How long the first reset of a batch waits for the slots that have not asked for it, in seconds.
DEFAULT_JOIN_DEADLINE = 600.0
# How often the server loop wakes with no request, to check ``shutdown``, in seconds.
_IDLE_CHECK_INTERVAL = 0.5
# How often a client thread that waits on an answer checks that its client is still connected, in seconds.
_CONNECTION_CHECK_INTERVAL = 0.1


class EnvProtocol(ABC):
    """An environment exchanging raw arrays and plain data with no Positronic dependencies.

    Implementations convert wire commands to native actions and own environment construction.
    """

    @abstractmethod
    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Task records selected by ``spec``, each with a ``name`` field and an ``instruction`` field.

        An absent key leaves that axis unrestricted; an unknown value raises.
        """

    @property
    def num_slots(self) -> int:
        """The most episodes the env steps together, each driven by a client of its own."""
        return 1

    @abstractmethod
    def reset(self, token: Any, slots: int | None = None) -> dict[str, Any]:
        """Construct or reuse the environment for ``slots`` slots, and re-randomize every slot from an opaque token.

        ``slots`` is at most ``num_slots``, and ``None`` asks for ``num_slots``. An env that builds its slots at reset
        builds that many; an env with one slot ignores it.
        Return the ``slots`` field (one ``{'obs': ...}`` per slot), scene ``meta``, ``robot_meta``, and ``control_dt``
        in seconds. Either metadata dict may be empty when the client supplies it. A slot whose scene draws a goal
        other than its task's listed instruction also carries ``goal``.
        """

    def reset_slot(self, token: Any, slot: int) -> dict[str, Any]:
        """Re-randomize env slot ``slot`` alone, while the other slots of the batch keep their state.

        ``token`` is the token of the last reset of the batch. Return the fields of ``reset``; ``slots`` holds at least
        the entry of ``slot``. An env that cannot reset one slot alone raises.
        """
        raise NotImplementedError(f'{type(self).__name__} resets no slot alone')

    @abstractmethod
    def step(self, actions: dict[int, dict[str, Any]]) -> dict[str, Any]:
        """Apply each slot's raw action for one control period; return ``slots`` and ``control_dt``.

        ``actions`` maps a slot index to its action and leaves out every slot that is not in an episode; the env
        holds such a slot. ``slots`` answers with one ``{'obs', 'done'}`` per slot, in slot order. A slot may
        also carry ``goal``, the new goal text of a task whose goal moves in the episode.
        ``control_dt`` is the wait until the next step and may vary each step.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the env's resources."""


class _Phase(Enum):
    VACANT = 'vacant'  # no client holds the slot
    IDLE = 'idle'  # a client, and no episode yet
    WAITING = 'waiting'  # in the batch, and asked for its next reset
    JOINING = 'joining'  # outside the batch, and asked to join the next one
    RUNNING = 'running'  # in an episode
    ENDED = 'ended'  # its episode reported ``done``
    RESET_FAILED = 'reset failed'  # in the batch, and its last reset failed


@dataclass
class _Request:
    slot: int
    msg: dict[str, Any]
    reply: 'queue.Queue[dict[str, Any]]'


@dataclass
class _Disconnect:
    slot: int


@dataclass
class _Slot:
    phase: _Phase = _Phase.VACANT
    pending: _Request | None = None  # the reset or step this slot waits on


def _error(e: Exception) -> dict[str, Any]:
    return {protocol.ERROR: f'{type(e).__name__}: {e}'}


class EnvServer:
    """Serve one client per slot of ``env``; ``shutdown`` releases the owned environment.

    The env runs on the thread that calls ``serve_forever``: macOS GLFW requires environment calls on the main
    thread. ``serve_forever`` returns on shutdown.
    """

    def __init__(
        self,
        env: EnvProtocol,
        host: str,
        port: int,
        *,
        join_deadline: float = DEFAULT_JOIN_DEADLINE,
        ping_timeout: float = protocol.HEARTBEAT_TIMEOUT_S,
    ):
        self._env = env
        self._host = host
        self._port = port
        self._join_deadline = join_deadline
        self._ping_timeout = ping_timeout
        self._join_started: float | None = None  # when the first slot asked for the reset that starts a batch
        self._slots = [_Slot() for _ in range(env.num_slots)]
        self._batch_size = env.num_slots
        self._env_slots: dict[int, int] = {}  # the env slot of each slot of the batch
        self._batch_token: bytes | None = None  # the encoded token of the batch, while the env holds its scene
        self._requests: queue.Queue[_Request | _Disconnect] = queue.Queue()
        self._lock = threading.Lock()  # guards slot claims and ``_stopped``
        self._stopped = False
        self._server = None
        self._shutdown = False

    def _claim(self) -> int | None:
        with self._lock:
            for index, slot in enumerate(self._slots):
                if slot.phase is _Phase.VACANT:
                    slot.phase = _Phase.IDLE
                    return index
        return None

    def _submit(self, request: _Request, connection: ServerConnection) -> dict[str, Any] | None:
        """The answer to ``request``, or ``None`` when its client disconnects first."""
        with self._lock:
            if self._stopped:
                return {protocol.ERROR: 'RuntimeError: the env server stopped'}
            self._requests.put(request)
        # ``connection.state`` needs websockets 13; a simulator's venv can carry 12, which has only this path.
        while connection.protocol.state is State.OPEN:
            try:
                return request.reply.get(timeout=_CONNECTION_CHECK_INTERVAL)
            except queue.Empty:
                pass
        return None

    def _refuse(self, connection: ServerConnection) -> None:
        """Answer a client that found no free slot with an error per request, until it closes."""
        for raw in connection:
            closing = protocol.decode(raw).get(protocol.CMD) == protocol.Command.CLOSE.value
            refused = {protocol.ERROR: f'RuntimeError: all {len(self._slots)} slots are taken'}
            connection.send(protocol.encode({protocol.OK: True} if closing else refused))
            if closing:
                return

    def _handle(self, connection: ServerConnection) -> None:
        slot = self._claim()
        if slot is None:
            self._refuse(connection)
            return
        closed = False  # a close frees the slot itself
        try:
            for raw in connection:
                msg = protocol.decode(raw)
                answer = self._submit(_Request(slot, msg, queue.Queue(maxsize=1)), connection)
                if answer is None:
                    return
                closed = msg.get(protocol.CMD) == protocol.Command.CLOSE.value
                connection.send(protocol.encode(answer))
                if closed:
                    return
        finally:
            with self._lock:
                if not self._stopped and not closed:
                    self._requests.put(_Disconnect(slot))

    def _take(self, request: _Request | _Disconnect) -> None:
        """Answer ``request`` now, or park it on its slot until the batch can run."""
        slot = self._slots[request.slot]
        if isinstance(request, _Disconnect):
            slot.phase, slot.pending = _Phase.VACANT, None
            return
        try:
            match protocol.Command(request.msg[protocol.CMD]):
                case protocol.Command.CLOSE:
                    slot.phase = _Phase.VACANT
                    request.reply.put({protocol.OK: True})
                case protocol.Command.TASKS:
                    request.reply.put({protocol.TASKS: self._env.tasks(request.msg[protocol.SPEC])})
                case protocol.Command.RESET if request.msg.get(protocol.REFILL):
                    self._refill(request)
                case protocol.Command.RESET:  # from a running slot too: its client gave up on the episode
                    in_batch = slot.phase in (_Phase.RUNNING, _Phase.ENDED, _Phase.RESET_FAILED)
                    if not in_batch:
                        self._check_batch_size(request)
                    slot.phase, slot.pending = (_Phase.WAITING if in_batch else _Phase.JOINING), request
                case protocol.Command.STEP:
                    if slot.phase not in (_Phase.RUNNING, _Phase.ENDED) or slot.pending is not None:
                        raise RuntimeError(f'slot {request.slot} stepped while {slot.phase.value}')
                    slot.phase, slot.pending = _Phase.RUNNING, request
        except Exception as e:
            request.reply.put(_error(e))

    def _refill(self, request: _Request) -> None:
        """Reset the slot of ``request`` alone into the batch that runs, and answer it."""
        index, slot = request.slot, self._slots[request.slot]
        token = request.msg[protocol.TOKEN]
        if self._batch_token is None or protocol.encode(token) != self._batch_token:
            raise ValueError('a refill resets a slot into the batch that runs, with the token of that batch')
        if request.msg.get(protocol.BATCH_SIZE, self._batch_size) != self._batch_size:
            raise ValueError(f'the batch that runs has {self._batch_size} slots, but this refill asks for another size')
        if self._joining_batch_size() is not None:
            raise RuntimeError('a slot waits to join the next batch, so this batch takes no refill')
        if slot.phase is _Phase.RUNNING:
            raise RuntimeError('a refill resets a clone whose episode ended, and this slot still runs its episode')
        in_batch = slot.phase in (_Phase.ENDED, _Phase.RESET_FAILED)
        env_slot = self._env_slots[index] if in_batch else self._free_env_slot()
        try:
            answer = self._env.reset_slot(token, env_slot)
        except Exception:
            if in_batch:
                slot.phase = _Phase.RESET_FAILED
            raise
        self._env_slots[index] = env_slot
        slot.phase = _Phase.RUNNING
        request.reply.put(protocol.slot_frame(answer, env_slot))

    def _free_env_slot(self) -> int:
        """The first env slot of the batch that no slot of the batch holds."""
        in_batch = (_Phase.WAITING, _Phase.RUNNING, _Phase.ENDED, _Phase.RESET_FAILED)
        held = {self._env_slots[index] for index, slot in enumerate(self._slots) if slot.phase in in_batch}
        free = [env_slot for env_slot in range(self._batch_size) if env_slot not in held]
        if not free:
            raise RuntimeError(f'every env slot of the batch of {self._batch_size} is held')
        return free[0]

    def _asked_batch_size(self, request: _Request) -> int:
        return request.msg.get(protocol.BATCH_SIZE, len(self._slots))

    def _joining_batch_size(self) -> int | None:
        """The batch size the slots that wait to join ask for, or ``None`` when no slot waits to join."""
        for slot in self._slots:
            if slot.phase is _Phase.JOINING and slot.pending is not None:
                return self._asked_batch_size(slot.pending)
        return None

    def _check_batch_size(self, request: _Request) -> None:
        """Raise unless ``request`` asks for a batch size this server serves, and the size the waiting slots ask for."""
        size = self._asked_batch_size(request)
        if type(size) is not int:  # a float or a bool slices the slot list wrong and stops the whole server
            raise ValueError(f'a batch size is a whole number, but this reset asks for {size!r}')
        if not 1 <= size <= len(self._slots):
            raise ValueError(f'a batch of {size} slots is asked for, but this env serves 1 to {len(self._slots)} slots')
        waiting = self._joining_batch_size()
        if waiting is not None and size != waiting:
            raise ValueError(f'the next batch has {waiting} slots, but this reset asks for a batch of {size}')

    def _take_pending(self, indices: list[int]) -> dict[int, _Request]:
        """The parked request of each slot in ``indices``, by slot index, cleared from its slot."""
        taken = {}
        for index in indices:
            slot = self._slots[index]
            if slot.pending is not None:
                taken[index], slot.pending = slot.pending, None
        return taken

    def _reset(self, indices: list[int]) -> None:
        """Reset the env for the slots ``indices``, which then are the batch."""
        requests = self._take_pending(indices)
        try:
            tokens = {protocol.encode(request.msg[protocol.TOKEN]) for request in requests.values()}
            if len(tokens) > 1:
                raise ValueError(
                    f'{len(requests)} slots asked for {len(tokens)} different resets at once, but all slots of one env '
                    'server share one scene and one task: run every client of this server in one task order'
                )
            self._batch_token = None  # a reset that raises leaves the env in no known scene
            answer = self._env.reset(next(iter(requests.values())).msg[protocol.TOKEN], self._batch_size)
        except Exception as e:
            for index, request in requests.items():
                self._slots[index].phase = _Phase.RESET_FAILED
                request.reply.put(_error(e))
            return
        self._batch_token = next(iter(tokens))
        for index, request in requests.items():
            self._slots[index].phase = _Phase.RUNNING
            request.reply.put(protocol.slot_frame(answer, self._env_slots[index]))

    def _start_batch(self, indices: list[int], size: int) -> None:
        """Reset the env for a batch of ``size`` env slots, driven by the slots ``indices``, in that order."""
        self._batch_size = size
        self._env_slots = {index: env_slot for env_slot, index in enumerate(indices)}
        self._reset(indices)

    def _step(self, indices: list[int]) -> None:
        requests = self._take_pending(indices)
        try:
            actions = {self._env_slots[index]: request.msg[protocol.ACTION] for index, request in requests.items()}
            answer = self._env.step(actions)
        except Exception as e:
            for request in requests.values():
                request.reply.put(_error(e))
            return
        for index, request in requests.items():
            frame = protocol.slot_frame(answer, self._env_slots[index])
            if frame[protocol.FRAME_DONE]:
                self._slots[index].phase = _Phase.ENDED
            request.reply.put(frame)

    def _advance(self) -> None:
        """Run the batch step or the batch reset, or start a batch, if every slot it waits for is ready."""
        in_batch = (_Phase.WAITING, _Phase.RUNNING, _Phase.ENDED, _Phase.RESET_FAILED)
        batch = [index for index, slot in enumerate(self._slots) if slot.phase in in_batch]
        if batch:
            running = [index for index in batch if self._slots[index].phase is _Phase.RUNNING]
            if running:
                if all(self._slots[index].pending is not None for index in running):
                    self._step(running)
            elif all(self._slots[index].phase is _Phase.WAITING for index in batch):
                self._reset(batch)
            return
        joining = [index for index, slot in enumerate(self._slots) if slot.phase is _Phase.JOINING]
        size = self._joining_batch_size()
        if size is None:
            self._join_started = None
            return
        if self._join_started is None:
            self._join_started = time.monotonic()
        if len(joining) >= size or time.monotonic() - self._join_started >= self._join_deadline:
            self._join_started = None
            self._start_batch(joining[:size], size)

    def _seconds_to_next_check(self) -> float:
        if self._join_started is None:
            return _IDLE_CHECK_INTERVAL
        return max(0.0, min(_IDLE_CHECK_INTERVAL, self._join_started + self._join_deadline - time.monotonic()))

    def _stop(self) -> None:
        """Fail every request still open, so no client thread waits on an answer that never comes."""
        with self._lock:
            self._stopped = True
        stopped = {protocol.ERROR: 'RuntimeError: the env server stopped'}
        for slot in self._slots:
            if slot.pending is not None:
                slot.pending.reply.put(stopped)
                slot.pending = None
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return
            if isinstance(request, _Request):
                request.reply.put(stopped)

    def _serve(self) -> 'Server':
        """A websockets server for ``_handle`` with the settings of the env server.

        Reset tokens can exceed the default frame size. No deflate: see "The env-server wire" in
        positronic/simulator/README.md. A native scene load blocks the server's keepalive too, so the server waits
        ``ping_timeout`` for a pong. websockets 12, which a simulator's venv can carry, has no keepalive here.
        """
        if 'ping_timeout' not in inspect.signature(serve).parameters:
            return serve(self._handle, self._host, self._port, max_size=None, compression=None)
        return serve(
            self._handle, self._host, self._port, max_size=None, compression=None, ping_timeout=self._ping_timeout
        )

    def serve_forever(self) -> None:
        # Bare TCP probes never reach ``_handle``, so they take no slot.
        with self._serve() as server:
            self._server = server
            accept = threading.Thread(target=server.serve_forever, daemon=True)
            accept.start()
            try:
                while not self._shutdown:
                    try:  # wake with no request too, to check the join deadline and ``shutdown``
                        request = self._requests.get(timeout=self._seconds_to_next_check())
                    except queue.Empty:
                        request = None
                    if request is not None:
                        self._take(request)
                    self._advance()
            finally:
                self._stop()
                server.shutdown()

    def shutdown(self) -> None:
        self._shutdown = True
        if self._server is not None:
            self._server.shutdown()
        self._env.close()
