"""The teleop service: the headset on one side, one teleop session of a ``RemotePolicy`` on the other.

``positronic teleop`` runs it beside a run whose policy dials its socket. It is a model server: each session declares
``TeleopClient`` in its client stack. The service keeps the trial state and the clutch, so the input of the
teleoperator stays in this process. The headset page shows the camera image of the run and the state of the trial.
The teleoperator uses the buttons of the right controller.
"""

import _thread
import logging
import math
import os
import stat
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from positronic_model_server import spec
from positronic_model_server.server import Model, ModelServer, Session
from positronic_model_server.websocket_wire import ServedUnixSocket, WebsocketWire

from positronic import geom
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd
from positronic.policy.codecs import EncodeImages
from positronic.policy.teleop import TeleopClient

from .clutch import Tracker
from .protocol import EE_POSE, END, GRIP, IMAGES, KIND, START_TASK, TARGET, TASK, Reply, Request, SessionParams

logger = logging.getLogger(__name__)

QUAT = geom.Rotation.Representation.QUAT

# The index of each button in the values that the headset page sends for a controller, in the xr-standard gamepad
# layout.
_TRIGGER, _STICK, _A, _B = 0, 3, 4, 5


class Event(StrEnum):
    """A press that the teleoperator made since the previous frame."""

    # Turn control on or off.
    CLUTCH = 'clutch'
    # The real attempt starts. The moves before it are trial moves.
    START_TASK = 'start_task'
    # End the episode: the teleoperator thinks that the task is done.
    COMPLETE = 'complete'
    # End the episode: the teleoperator stops before the task is done.
    GIVE_UP = 'give_up'


class ButtonMap:
    """The buttons of the right controller as the replies and the events of a session.

    - A: control on or off.
    - B: Ready while the service waits for it, then Start task. A hold of B after the start is Complete.
    - The stick: Cancel while the service waits for Ready. A hold of the stick is Give up.
    - The trigger: the grip.
    """

    def __init__(self, hold_sec: float, press_threshold: float = 0.6):
        if not 0 < press_threshold <= 1:
            raise ValueError(f'press_threshold must be in (0, 1], got {press_threshold}')
        if not 0 < hold_sec < math.inf:
            raise ValueError(f'hold_sec must be finite and positive, got {hold_sec}')
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
    cameras: tuple[str, ...]


@dataclass(frozen=True)
class _Sample:
    """What the headset sent last, and its number in the order of the samples."""

    number: int
    pose: geom.Transform3D | None
    buttons: Sequence[float] | None


_DEFAULT_SESSION_PARAMS = SessionParams()


class TeleopService:
    """Serve one teleop session at a time on the Unix socket ``uds``, as a model server.

    ``take_sample`` takes each sample of the right controller. A frame request waits for the next sample. When none
    comes within ``silence_limit``, the frame has no pose ("not tracked"), and control goes off. A ready request waits
    at most ``READY_SLICE_SEC`` for Ready or Cancel. The ready limit counts from the first ready request of a session.
    ``session_defaults`` holds the parameters of a session that sets none.
    """

    READY_SLICE_SEC = 1.0
    # The client stops waiting for Ready this long after the ready limit, when the service does not answer.
    READY_BACKSTOP_MARGIN_SEC = 10.0

    def __init__(
        self,
        uds: Path,
        ready_limit: float,
        silence_limit: float,
        hold_sec: float,
        press_threshold: float = 0.6,
        headset_camera: str | None = None,
        session_defaults: SessionParams = _DEFAULT_SESSION_PARAMS,
    ):
        for name, value in (('ready_limit', ready_limit), ('silence_limit', silence_limit)):
            if not 0 < value < math.inf:
                raise ValueError(f'{name} must be finite and positive, got {value}')
        self._uds = uds
        self.ready_limit = ready_limit
        self.silence_limit = silence_limit
        self._session_defaults = session_defaults
        self._buttons = ButtonMap(hold_sec, press_threshold)
        self._changed = threading.Condition()
        self._sample = _Sample(0, None, None)
        self._phase = Phase.IDLE
        self._ready: Reply | None = None
        self._events: list[Event] = []
        self._task = ''
        self._control = False
        # Whether the service emitted Start task in this trial.
        self._attempt_started = False
        # The last camera images of the run, by name, in the order that the client sends them.
        self._images: dict[str, np.ndarray] = {}
        # The camera that the headset shows. ``None`` is the first camera of the run.
        self._headset_camera = headset_camera
        # A wait ends at once when the service stops.
        self._stopping = False
        self._server: ModelServer | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _prepare_private_directory(directory: Path) -> None:
        """Create ``directory`` for this user only, or check that it is private to this user.

        Any account that can write to the directory of the socket can bind the socket first, and get the sessions.
        """
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # `lstat`, so a symlink that another account put there does not pass as the directory it points to.
        mode = os.lstat(directory)
        if not stat.S_ISDIR(mode.st_mode):
            raise PermissionError(f'{directory} must be a directory, not a link or a file')
        if mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) & 0o077:
            raise PermissionError(f'{directory} must belong to this user, with no access for others (mode 700)')

    def __enter__(self) -> 'TeleopService':
        self._prepare_private_directory(self._uds.parent)
        server = ModelServer(lambda: TeleopModel(self, self._session_defaults))
        wires = [WebsocketWire(ServedUnixSocket(self._uds))]
        started: Future[None] = Future()

        def serve() -> None:
            try:
                server.serve(wires, on_ready=lambda: started.set_result(None))
            except BaseException as failed:
                if not started.done():
                    started.set_exception(failed)
                    return
                # Stop the process, so that the page and the window do not stay up with no socket behind them.
                logger.exception('The model server of the teleop service stopped')
                _thread.interrupt_main()

        self._server = server
        self._thread = threading.Thread(target=serve, name='teleop-service', daemon=True)
        self._thread.start()
        started.result(timeout=10)
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._server is not None and self._thread is not None
        with self._changed:
            self._stopping = True
            self._changed.notify_all()
        self._server.shutdown()
        self._thread.join(timeout=10)

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
                events = self._buttons.events(buttons, now_ns, self._attempt_started)
                self._attempt_started = self._attempt_started or Event.START_TASK in events
                self._events += events
            else:
                self._buttons.observe(buttons, now_ns)
            self._changed.notify_all()

    @property
    def phase(self) -> Phase:
        with self._changed:
            return self._phase

    def state(self) -> TrialState:
        with self._changed:
            return TrialState(self._phase, self._task, self._control, self._attempt_started, tuple(self._images))

    def camera_image(self, name: str) -> np.ndarray:
        """The last image of camera ``name``. Raise ``KeyError`` for a camera that the run does not send."""
        with self._changed:
            return self._images[name]

    def cancel_ready_wait(self) -> None:
        """Cancel the trial that waits for Ready, from the window on the station screen.

        Ready comes only from the headset or the phone, because it confirms that the person holds the controller.
        """
        with self._changed:
            if self._phase is not Phase.READY_WAIT or self._ready is not None:
                raise NotWaiting('no trial waits for Ready')
            self._ready = Reply.CANCELLED
            self._changed.notify_all()

    @staticmethod
    def _view(image: np.ndarray | None, lines: Sequence[str]) -> np.ndarray:
        """``image``, or a black image, with ``lines`` of text at its top."""
        blank = (480, 640, 3)
        # A copy: the text goes on the view, not on the image that the service holds.
        view = np.zeros(blank, dtype=np.uint8) if image is None else np.array(image, dtype=np.uint8, order='C')
        for row, line in enumerate(lines):
            origin = (10, 30 + 30 * row)
            cv2.putText(view, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    def view(self) -> np.ndarray:
        """The image for the headset: the camera image of the run, with the state of the trial."""
        with self._changed:
            state = self.state()
            image = self._headset_image()
        match state.phase:
            case Phase.IDLE:
                return self._view(None, ['Waiting for a trial'])
            case Phase.READY_WAIT:
                return self._view(None, ['Human trial', 'B: Ready', 'Stick: Cancel'])
            case Phase.ENDED:
                return self._view(None, ['The trial ended', 'Give the verdict on the console'])
        control = 'Control on (A: off)' if state.control else 'Control off (A: on)'
        task = 'Hold B: Complete. Hold the stick: Give up' if state.attempt_started else 'B: Start task'
        return self._view(image, [state.task, control, task])

    def _headset_image(self) -> np.ndarray | None:
        if self._headset_camera is not None:
            return self._images.get(self._headset_camera)
        return next(iter(self._images.values()), None)

    def begin_ready_wait(self) -> None:
        """Start a new trial that waits for Ready."""
        with self._changed:
            self._clear_trial()
            self._phase, self._ready = Phase.READY_WAIT, None

    def await_ready(self, deadline: float) -> Reply:
        """Wait at most ``READY_SLICE_SEC`` for Ready or Cancel. Answer ``EXPIRED`` at the ``time.monotonic()``
        instant ``deadline``, and ``WAITING`` when the slice ends first."""
        slice_end = min(time.monotonic() + self.READY_SLICE_SEC, deadline)
        with self._changed:
            if self._phase is not Phase.READY_WAIT:
                raise ValueError(f'a ready request came in the {self._phase} phase')
            while self._ready is None and not self._stopping and (left := slice_end - time.monotonic()) > 0:
                self._changed.wait(left)
            if self._stopping:
                self._phase = Phase.IDLE
                raise RuntimeError('the teleop service stops')
            if self._ready is None:
                if time.monotonic() < deadline:
                    return Reply.WAITING
                self._phase = Phase.IDLE
                return Reply.EXPIRED
            self._phase = Phase.RUNNING if self._ready is Reply.READY else Phase.IDLE
            return self._ready

    def _next_sample(self, seen: int, deadline: float) -> _Sample:
        """The sample after sample number ``seen``, or a sample with no pose when none comes before ``deadline``. Hold
        the lock."""
        while self._sample.number == seen and not self._stopping and (left := deadline - time.monotonic()) > 0:
            self._changed.wait(left)
        return self._sample if self._sample.number != seen else _Sample(seen, None, None)

    def next_frame(self, request: Mapping[str, Any], tracker: Tracker) -> dict[str, Any]:
        """Take the state of the robot in ``request``, and answer with the next frame of the teleoperator input."""
        robot_pose = geom.Transform3D.from_vector(np.asarray(request[EE_POSE], dtype=float), QUAT)
        deadline = time.monotonic() + self.silence_limit
        with self._changed:
            if self._phase is not Phase.RUNNING:
                raise ValueError(f'a frame request came in the {self._phase} phase')
            self._task = str(request[TASK])
            self._images = dict(request.get(IMAGES, self._images))
            sample = self._next_sample(self._sample.number, deadline)
            events, self._events = self._events, []
            reply = self._apply(sample, events, tracker, robot_pose)
            self._control = tracker.on
            if END in reply:
                self._phase = Phase.ENDED
            return reply

    @staticmethod
    def _apply(
        sample: _Sample, events: Sequence[Event], tracker: Tracker, robot_pose: geom.Transform3D
    ) -> dict[str, Any]:
        """The frame reply of ``sample`` and ``events``. Control starts from ``robot_pose``, so the robot does not
        jump."""
        reply: dict[str, Any] = {KIND: Reply.FRAME, START_TASK: Event.START_TASK in events}
        target = None if sample.pose is None else tracker.update(sample.pose)
        for event in events:
            match event:
                case Event.CLUTCH if tracker.on:
                    tracker.turn_off()
                case Event.CLUTCH if sample.pose is not None:
                    tracker.turn_on(robot_pose, sample.pose)
                    target = tracker.update(sample.pose)
                case Event.COMPLETE:
                    reply[END] = PolicyEnd.COMPLETE.value
                    return reply
                case Event.GIVE_UP:
                    reply[END] = PolicyEnd.GIVE_UP.value
                    return reply
        if target is None and tracker.on:
            tracker.turn_off()
        if tracker.on and target is not None:
            reply[TARGET] = target.as_vector(QUAT)
            reply[GRIP] = 0.0 if sample.buttons is None else ButtonMap.grip(sample.buttons)
        return reply

    def _clear_trial(self) -> None:
        """Show no trial. Hold the lock."""
        self._phase, self._events = Phase.IDLE, []
        self._task, self._control, self._attempt_started, self._images = '', False, False, {}

    def end_session(self) -> None:
        with self._changed:
            self._clear_trial()


class TeleopSession(Session):
    """One trial of the teleop service: the ready wait, then the frames."""

    def __init__(
        self, service: TeleopService, tracker: Tracker, client_stack: dict[str, Any], on_close: Callable[[], None]
    ):
        super().__init__(client_stack)
        self._service = service
        self._tracker = tracker
        self._on_close = on_close
        # The `time.monotonic()` instant at which the ready limit ends. It starts at the first ready request.
        self._ready_deadline: float | None = None

    def infer(self, observation: Any) -> dict[str, Any]:
        match Request(observation[KIND]):
            case Request.READY:
                if self._ready_deadline is None:
                    self._ready_deadline = time.monotonic() + self._service.ready_limit
                    self._service.begin_ready_wait()
                return {KIND: self._service.await_ready(self._ready_deadline)}
            case Request.FRAME:
                return self._service.next_frame(observation, self._tracker)

    def close(self) -> None:
        self._service.end_session()
        self._on_close()


class TeleopModel(Model):
    """The teleop service as a model server serves it. It prepares one session at a time.

    ``ModelServer`` calls it on one worker thread only.
    """

    def __init__(self, service: TeleopService, defaults: SessionParams):
        super().__init__(
            parameters=defaults.to_json(),
            metadata={
                policy_keys.TYPE: 'teleop',
                'ready_limit_sec': service.ready_limit,
                'silence_limit_sec': service.silence_limit,
            },
        )
        self._service = service
        self._open = False

    def prepare_session(self, params: dict[str, Any]) -> TeleopSession:
        if self._open:
            raise RuntimeError('the teleop service serves one session at a time')
        settings = SessionParams.from_json(**params)
        client = TeleopClient(
            ready_backstop_sec=self._service.ready_limit + TeleopService.READY_BACKSTOP_MARGIN_SEC,
            cameras=settings.cameras,
            image_fps=settings.image_fps,
        )
        stack = spec.sequence(client.to_spec(), EncodeImages(quality=settings.jpeg_quality).to_spec())
        session = TeleopSession(self._service, Tracker(settings.operator()), stack, self._release)
        self._open = True
        return session

    def _release(self) -> None:
        self._open = False
