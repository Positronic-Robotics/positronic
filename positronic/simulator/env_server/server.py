"""Synchronous env server: one client per slot, and every slot stepped in one env call.

This module must work in an isolated interpreter without Positronic installed.

Protocol (msgpack frames, see ``protocol``), between a client and its slot:
  client ``{'cmd': 'tasks', 'spec': ...}``    -> server ``{'tasks': [{...}, ...]}``
  client ``{'cmd': 'reset', 'token': ...}``   -> server ``{'obs', 'meta', 'robot_meta', 'control_dt'}``, opt. ``goal``
  client ``{'cmd': 'step', 'action': {...}}`` -> server ``{'obs', 'done', 'control_dt'}``, opt. ``goal``
  client ``{'cmd': 'close'}``                 -> server ``{'ok': True}``
Command handling failures return ``{'error': str}`` without closing the session; the client re-raises them.

``control_dt`` is the control period in seconds and can vary per step.
``meta`` identifies the scene; ``robot_meta`` identifies the robot model. Either metadata dict can be empty when the
client supplies that information. A task record's ``instruction`` is the text the policy gets for that task. A frame
carries ``goal`` when the env moves the goal away from that text, and a later ``goal`` replaces an earlier one.

An env serves a fixed number of slots, ``num_slots``: independent episodes that one env call steps together. Most
envs serve one. Each client that connects takes the next free slot and drives only that slot, so N clients can run N
different policies in one env:

* The slots that reset together are a batch. All of them reset with one token, and the env resets every slot.
* A batch's first reset waits for every slot, but not longer than ``join_deadline`` seconds after the first slot
  asks while no batch runs. A slot that has not asked by then sits out the batch, and the env gets no action for it.
* A later reset of a batch answers when every slot of the batch has asked for one or has left.
* A step answers when every slot in an episode has sent its action. The env gets no action for the other slots.
  A slot that reports ``done`` stops holding the others back; a client may still step it until its next reset.
* A slot leaves when its client closes or disconnects, and a new client can then take it. A slot outside the batch
  that asks for a reset waits until every slot of the batch has left, and then joins the next batch.
* The server serves batch after batch until ``shutdown``.
"""

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from websockets.sync.server import ServerConnection, serve

# Isolated interpreters can import these modules as a package or as flat files.
if __package__:
    from . import protocol
else:
    import protocol

# How long the first reset of a batch waits for the slots that have not asked for it, in seconds.
DEFAULT_JOIN_DEADLINE = 600.0


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
        """How many episodes the env steps together, each driven by a client of its own."""
        return 1

    @abstractmethod
    def reset(self, token: Any) -> dict[str, Any]:
        """Construct or reuse the environment and re-randomize every slot from an opaque token.

        Return ``slots`` (one ``{'obs': ...}`` per slot), scene ``meta``, ``robot_meta``, and ``control_dt``
        in seconds. Either metadata dict may be empty when the client supplies it. A slot whose scene draws a goal
        other than its task's listed instruction also carries ``goal``.
        """

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

    def __init__(self, env: EnvProtocol, host: str, port: int, *, join_deadline: float = DEFAULT_JOIN_DEADLINE):
        self._env = env
        self._host = host
        self._port = port
        self._join_deadline = join_deadline
        self._join_started: float | None = None  # when the first slot asked for the reset that starts a batch
        self._slots = [_Slot() for _ in range(env.num_slots)]
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

    def _submit(self, request: _Request) -> dict[str, Any]:
        with self._lock:
            if self._stopped:
                return {protocol.ERROR: 'RuntimeError: the env server stopped'}
            self._requests.put(request)
        return request.reply.get()

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
                answer = self._submit(_Request(slot, msg, queue.Queue(maxsize=1)))
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
            slot.phase = _Phase.VACANT
            return
        try:
            match protocol.Command(request.msg[protocol.CMD]):
                case protocol.Command.CLOSE:
                    slot.phase = _Phase.VACANT
                    request.reply.put({protocol.OK: True})
                case protocol.Command.TASKS:
                    request.reply.put({protocol.TASKS: self._env.tasks(request.msg[protocol.SPEC])})
                case protocol.Command.RESET:  # from a running slot too: its client gave up on the episode
                    in_episode = slot.phase in (_Phase.RUNNING, _Phase.ENDED)
                    slot.phase, slot.pending = (_Phase.WAITING if in_episode else _Phase.JOINING), request
                case protocol.Command.STEP:
                    if slot.phase not in (_Phase.RUNNING, _Phase.ENDED) or slot.pending is not None:
                        raise RuntimeError(f'slot {request.slot} stepped while {slot.phase.value}')
                    slot.phase, slot.pending = _Phase.RUNNING, request
        except Exception as e:
            request.reply.put(_error(e))

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
        for index in requests:
            self._slots[index].phase = _Phase.IDLE
        try:
            tokens = {protocol.encode(request.msg[protocol.TOKEN]) for request in requests.values()}
            if len(tokens) > 1:
                raise ValueError(
                    f'{len(requests)} slots asked for {len(tokens)} different resets at once, but all slots of one env '
                    'server share one scene and one task: run every client of this server in one task order'
                )
            answer = self._env.reset(next(iter(requests.values())).msg[protocol.TOKEN])
        except Exception as e:
            for request in requests.values():
                request.reply.put(_error(e))
            return
        for index, request in requests.items():
            self._slots[index].phase = _Phase.RUNNING
            request.reply.put(protocol.slot_frame(answer, index))

    def _step(self, indices: list[int]) -> None:
        requests = self._take_pending(indices)
        try:
            answer = self._env.step({index: request.msg[protocol.ACTION] for index, request in requests.items()})
        except Exception as e:
            for request in requests.values():
                request.reply.put(_error(e))
            return
        for index, request in requests.items():
            frame = protocol.slot_frame(answer, index)
            if frame[protocol.FRAME_DONE]:
                self._slots[index].phase = _Phase.ENDED
            request.reply.put(frame)

    def _advance(self) -> None:
        """Run the batch step or the batch reset, or start a batch, if every slot it waits for is ready."""
        in_batch = (_Phase.WAITING, _Phase.RUNNING, _Phase.ENDED)
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
        if not joining:
            self._join_started = None
            return
        if self._join_started is None:
            self._join_started = time.monotonic()
        if len(joining) == len(self._slots) or time.monotonic() - self._join_started >= self._join_deadline:
            self._join_started = None
            self._reset(joining)

    def _seconds_to_next_check(self) -> float:
        if self._join_started is None:
            return 0.5
        return max(0.0, min(0.5, self._join_started + self._join_deadline - time.monotonic()))

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

    def serve_forever(self) -> None:
        # Reset tokens can exceed the default frame size.
        # Bare TCP probes never reach ``_handle``, so they take no slot.
        # No deflate: see "The env-server wire" in positronic/simulator/README.md.
        with serve(self._handle, self._host, self._port, max_size=None, compression=None) as server:
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
