"""The teleop service: the headset on one side, one ``TeleopPolicy`` session on the other.

``positronic teleop`` runs it beside a run that uses the ``teleop`` policy. The headset page shows the camera image
of the run and the state of the trial. The teleoperator uses the buttons of the right controller.
"""

import asyncio
import logging
import os
import stat
import threading
import time
from collections.abc import Coroutine, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from positronic_model_server import server_wire
from positronic_model_server.serialization import deserialise, serialise
from positronic_model_server.websocket_wire import ServedUnixSocket, WebsocketWire
from positronic_wire.wire import PeerDisconnected

from positronic import geom

from .session import (
    ATTEMPT_STARTED,
    CONTROL,
    DISPLAY,
    EVENTS,
    GRIP,
    IMAGE,
    KIND,
    MESSAGE,
    POSE,
    TASK,
    Event,
    Reply,
    Request,
)

logger = logging.getLogger(__name__)

# The index of each button in the values that the headset page sends for a controller: the xr-standard gamepad
# layout, which `BUTTON` names in the WebXR pages.
_TRIGGER, _STICK, _A, _B = 0, 3, 4, 5


class ButtonMap:
    """The buttons of the right controller as the replies and the events of a session.

    - A: control on or off.
    - B: Ready while the service waits for it, then Start task. A hold of B after the start is Complete.
    - The stick: Cancel while the service waits for Ready. A hold of the stick is Give up.
    - The trigger: the grip.
    """

    def __init__(self, hold_sec: float, press_threshold: float = 0.6):
        self._hold_ns = round(hold_sec * 1e9)
        # A button whose value is at least this is down.
        self._press_threshold = press_threshold
        # The time at which each button went down, while it is down.
        self._down_since: dict[int, int] = {}
        # The buttons whose press did something already, until they go up.
        self._used: set[int] = set()

    def _new_presses(self, buttons: Sequence[float], now_ns: int) -> set[int]:
        """Record the state of the buttons, and return the buttons that went down since the previous sample."""
        new_presses = set()
        for button in (_STICK, _A, _B):
            if buttons[button] < self._press_threshold:
                self._down_since.pop(button, None)
                self._used.discard(button)
            elif button not in self._down_since:
                self._down_since[button] = now_ns
                new_presses.add(button)
        return new_presses

    def observe(self, buttons: Sequence[float], now_ns: int) -> None:
        """Record the state of the buttons, outside a trial. A button that is down here does not press in the next."""
        self._used.update(self._new_presses(buttons, now_ns))

    def _is_held(self, button: int, now_ns: int) -> bool:
        since = self._down_since.get(button)
        return since is not None and button not in self._used and now_ns - since >= self._hold_ns

    def ready_reply(self, buttons: Sequence[float], now_ns: int) -> Reply | None:
        new_presses = self._new_presses(buttons, now_ns)
        if _B in new_presses:
            self._used.add(_B)
            return Reply.READY
        if _STICK in new_presses:
            self._used.add(_STICK)
            return Reply.CANCELLED
        return None

    def events(self, buttons: Sequence[float], now_ns: int, attempt_started: bool) -> list[Event]:
        new_presses = self._new_presses(buttons, now_ns)
        events = []
        if _A in new_presses:
            events.append(Event.CLUTCH)
        if _B in new_presses and not attempt_started:
            self._used.add(_B)
            events.append(Event.START_TASK)
        # Complete comes only after Start task.
        if attempt_started and self._is_held(_B, now_ns):
            self._used.add(_B)
            events.append(Event.COMPLETE)
        if self._is_held(_STICK, now_ns):
            self._used.add(_STICK)
            events.append(Event.GIVE_UP)
        return events

    @staticmethod
    def grip(buttons: Sequence[float]) -> float:
        return float(buttons[_TRIGGER])


class Phase(StrEnum):
    """Where the service is in a trial."""

    IDLE = 'idle'
    READY_WAIT = 'ready_wait'
    RUNNING = 'running'
    ENDED = 'ended'


class NotWaiting(Exception):
    """No trial waits for Ready."""


@dataclass(frozen=True)
class TrialState:
    """What the window on the station screen shows."""

    phase: Phase
    task: str
    control: bool
    attempt_started: bool


@dataclass(frozen=True)
class _Sample:
    """What the headset sent last, and its number in the order of the samples."""

    number: int
    pose: geom.Transform3D | None
    buttons: Sequence[float] | None


class TeleopService:
    """Serve one ``TeleopPolicy`` session at a time on the Unix socket ``uds``.

    ``take_sample`` takes each sample of the right controller. A frame request waits for the next sample. When none
    comes within ``silence_limit``, the frame has no pose ("not tracked"). The ready request waits for Ready or Cancel,
    and answers an error after ``ready_limit``.
    """

    def __init__(
        self, uds: Path, ready_limit: float, silence_limit: float, hold_sec: float, press_threshold: float = 0.6
    ):
        self._uds = uds
        self._ready_limit = ready_limit
        self._silence_limit = silence_limit
        self._buttons = ButtonMap(hold_sec, press_threshold)
        self._wire = WebsocketWire(ServedUnixSocket(uds))
        self._changed = threading.Condition()
        self._sample = _Sample(0, None, None)
        self._phase = Phase.IDLE
        self._ready: Reply | None = None
        self._events: list[Event] = []
        self._display: dict[str, Any] = {}
        self._open = False
        # A wait ends at once when its session ended, or when the service stops.
        self._session_ended = False
        self._stopping = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _prepare_private_directory(directory: Path) -> None:
        """Create ``directory`` for this user only, or check that it is private to this user.

        Any account that can write to the directory of the socket can bind the socket first, and get the sessions.
        """
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        mode = directory.stat()
        if mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) & 0o077:
            raise PermissionError(f'{directory} must belong to this user, with no access for others (mode 700)')

    def __enter__(self) -> 'TeleopService':
        self._prepare_private_directory(self._uds.parent)
        started: Future[None] = Future()

        async def serve() -> None:
            self._loop = asyncio.get_running_loop()
            try:
                await self._wire.start(self._session, lambda: None, lambda _headers: True)
            except BaseException as failed:
                started.set_exception(failed)
                raise
            started.set_result(None)
            await self._wire.serve()

        self._thread = threading.Thread(target=asyncio.run, args=(serve(),), name='teleop-service', daemon=True)
        self._thread.start()
        started.result(timeout=10)
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._loop is not None and self._thread is not None
        with self._changed:
            self._stopping = True
            self._changed.notify_all()
        asyncio.run_coroutine_threadsafe(self._wire.stop(), self._loop).result(timeout=10)
        self._thread.join(timeout=10)

    def _end_session(self) -> None:
        with self._changed:
            self._session_ended = True
            self._changed.notify_all()

    def _waits_must_end(self) -> bool:
        return self._session_ended or self._stopping

    def take_sample(self, pose: geom.Transform3D | None, buttons: Sequence[float] | None) -> None:
        """Take one sample of the right controller. ``None`` is a part that the headset does not track."""
        now_ns = time.monotonic_ns()
        with self._changed:
            self._sample = _Sample(self._sample.number + 1, pose, buttons)
            if buttons is None:
                pass
            elif self._phase is Phase.READY_WAIT and self._ready is None:
                self._ready = self._buttons.ready_reply(buttons, now_ns)
            elif self._phase is Phase.RUNNING:
                self._events += self._buttons.events(buttons, now_ns, bool(self._display.get(ATTEMPT_STARTED)))
            else:
                self._buttons.observe(buttons, now_ns)
            self._changed.notify_all()

    @property
    def phase(self) -> Phase:
        with self._changed:
            return self._phase

    def state(self) -> TrialState:
        with self._changed:
            phase, display = self._phase, self._display
            return TrialState(
                phase, str(display.get(TASK, '')), bool(display.get(CONTROL)), bool(display.get(ATTEMPT_STARTED))
            )

    def answer_ready(self, reply: Literal[Reply.READY, Reply.CANCELLED]) -> None:
        """Ready or Cancel from the window on the station screen, as the buttons of the headset give them."""
        with self._changed:
            if self._phase is not Phase.READY_WAIT or self._ready is not None:
                raise NotWaiting('no trial waits for Ready')
            self._ready = reply
            self._changed.notify_all()

    @staticmethod
    def _view(image: np.ndarray | None, lines: Sequence[str]) -> np.ndarray:
        """``image``, or a black image, with ``lines`` of text at its top."""
        blank = (480, 640, 3)
        view = np.zeros(blank, dtype=np.uint8) if image is None else np.ascontiguousarray(image, dtype=np.uint8)
        for row, line in enumerate(lines):
            origin = (10, 30 + 30 * row)
            cv2.putText(view, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    def view(self) -> np.ndarray:
        """The image for the headset: the camera image of the run, with the state of the trial."""
        with self._changed:
            phase, display = self._phase, dict(self._display)
        match phase:
            case Phase.IDLE:
                return self._view(None, ['Waiting for a trial'])
            case Phase.READY_WAIT:
                return self._view(None, ['Human trial', 'B: Ready', 'Stick: Cancel'])
            case Phase.ENDED:
                return self._view(None, ['The trial ended', 'Give the verdict on the console'])
        control = 'Control on (A: off)' if display.get(CONTROL) else 'Control off (A: on)'
        task = 'Hold B: Complete. Hold the stick: Give up' if display.get(ATTEMPT_STARTED) else 'B: Start task'
        return self._view(display.get(IMAGE), [str(display.get(TASK, '')), control, task])

    def _await_ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + self._ready_limit
        with self._changed:
            self._phase, self._ready, self._events, self._display = Phase.READY_WAIT, None, [], {}
            while self._ready is None:
                left = deadline - time.monotonic()
                if self._waits_must_end():
                    self._phase = Phase.IDLE
                    return {KIND: Reply.ERROR, MESSAGE: 'the teleop service ended the session'}
                if left <= 0:
                    self._phase = Phase.IDLE
                    return {KIND: Reply.ERROR, MESSAGE: f'the teleoperator was not ready within {self._ready_limit} s'}
                self._changed.wait(left)
            self._phase = Phase.RUNNING if self._ready is Reply.READY else Phase.IDLE
            return {KIND: self._ready}

    def _next_frame(self, display: dict[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + self._silence_limit
        with self._changed:
            self._display = display
            seen = self._sample.number
            while (
                self._sample.number == seen and not self._waits_must_end() and (left := deadline - time.monotonic()) > 0
            ):
                self._changed.wait(left)
            sample = self._sample if self._sample.number != seen else _Sample(seen, None, None)
            events, self._events = self._events, []
            if Event.COMPLETE in events or Event.GIVE_UP in events:
                self._phase = Phase.ENDED
        frame: dict[str, Any] = {
            KIND: Reply.FRAME,
            GRIP: 0.0 if sample.buttons is None else ButtonMap.grip(sample.buttons),
            EVENTS: [str(event) for event in events],
        }
        if sample.pose is not None:
            frame[POSE] = sample.pose.as_vector(geom.Rotation.Representation.QUAT)
        return frame

    async def _session(self, connection: server_wire.ServerConnection) -> None:
        if self._open:
            await connection.refuse('the teleop service serves one session at a time')
            return
        self._open = True
        with self._changed:
            self._session_ended = False
        try:
            await self._serve(connection)
        finally:
            with self._changed:
                if self._phase is not Phase.ENDED:
                    self._phase = Phase.IDLE
            self._open = False

    async def _serve(self, connection: server_wire.ServerConnection) -> None:
        while True:
            try:
                request = deserialise(await connection.receive())
            except PeerDisconnected:
                return  # The policy closed its end of the session.
            match Request(request[KIND]):
                case Request.READY:
                    reply = await self._answer_unless_the_peer_speaks(connection, asyncio.to_thread(self._await_ready))
                case Request.FRAME:
                    reply = await asyncio.to_thread(self._next_frame, request[DISPLAY])
            if reply is None:
                return
            await connection.send(serialise(reply))

    async def _answer_unless_the_peer_speaks(
        self, connection: server_wire.ServerConnection, wait: Coroutine[Any, Any, dict[str, Any]]
    ) -> dict[str, Any] | None:
        """The answer of ``wait``, or ``None`` when the policy disconnects or sends again first.

        The policy sends nothing while it waits for an answer, so a message then is a protocol error, and the session
        ends.
        """
        answer = asyncio.ensure_future(wait)
        peer = asyncio.ensure_future(connection.receive())
        await asyncio.wait({answer, peer}, return_when=asyncio.FIRST_COMPLETED)
        if answer.done():
            peer.cancel()
            return answer.result()
        peer.exception()
        self._end_session()
        await answer
        return None
