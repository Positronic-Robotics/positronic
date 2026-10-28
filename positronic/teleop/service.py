"""The teleop service: the headset on one side, one teleop session of a ``RemotePolicy`` on the other.

``positronic teleop`` runs it beside a run whose policy dials its socket. It is a model server: the preparation of a
session waits for Ready, and each frame request gets the next target of the teleoperator. Each session declares stock
components only in its client stack. The service keeps the trial state and the clutch, so the input of the
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
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from positronic_model_server import spec
from positronic_model_server.server import Model, ModelServer, PreparationCancelled, Session
from positronic_model_server.server_wire import ServedAddress
from positronic_model_server.websocket_wire import ServedUnixSocket, WebsocketWire

from positronic import geom, keys
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame
from positronic.offboard.client import DEFAULT_RECONNECT_DEADLINE
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd
from positronic.policy.codecs import (
    AbsolutePositionAction,
    ChangeEEFrame,
    EncodeImages,
    EndCommand,
    StateObservationCodec,
)
from positronic.policy.codecs.image import image_observation
from positronic.policy.codecs.metadata import ACTION
from positronic.policy.keys import END
from positronic.policy.processors import ChunkedSchedule, ConstantDuration, EndOnField

from .clutch import Tracker
from .protocol import EE_POSE, GRIP, TASK, VIEW_SIZE, SessionParams

logger = logging.getLogger(__name__)

QUAT = geom.Rotation.Representation.QUAT

# The index of each button in the values that the headset page sends for a controller, in the xr-standard gamepad
# layout.
_TRIGGER, _STICK, _A, _B = 0, 3, 4, 5


class Event(StrEnum):
    """A press that the teleoperator made since the previous frame."""

    # Turn control on or off.
    CLUTCH = 'clutch'
    # End the episode: the teleoperator thinks that the task is done.
    COMPLETE = 'complete'
    # End the episode: the teleoperator stops before the task is done.
    GIVE_UP = 'give_up'


class ReadyAnswer(Enum):
    """The answer of the teleoperator to the wait for Ready."""

    READY = 'ready'
    CANCELLED = 'cancelled'


class ResumeOutcome(Enum):
    """The outcome of a try to resume a dropped trial."""

    RESUMED = 'resumed'
    NO_DROPPED_TRIAL = 'no_dropped_trial'


class ButtonMap:
    """The buttons of the right controller as the answers and the events of a session.

    - A: control on or off.
    - B: Ready while the service waits for it. A hold of B in the trial is Complete.
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
        self._new_presses(buttons, now_ns)
        self._used.update(self._down_since)

    def _is_held(self, button: int, now_ns: int) -> bool:
        since = self._down_since.get(button)
        return since is not None and button not in self._used and now_ns - since >= self._hold_ns

    def ready_answer(self, buttons: Sequence[float], now_ns: int) -> ReadyAnswer | None:
        new_presses = self._new_presses(buttons, now_ns)
        # A button that is down before the trial runs does not hold in it.
        self._used.update(self._down_since)
        if _B in new_presses:
            return ReadyAnswer.READY
        if _STICK in new_presses:
            return ReadyAnswer.CANCELLED
        return None

    def events(self, buttons: Sequence[float], now_ns: int) -> list[Event]:
        new_presses = self._new_presses(buttons, now_ns)
        events = []
        if _A in new_presses:
            events.append(Event.CLUTCH)
        if self._is_held(_B, now_ns):
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


class TrialNotStarted(Exception):
    """The teleoperator cancelled the trial, or was not ready within the ready limit of the service."""


@dataclass(frozen=True)
class TrialState:
    """What the window on the station screen shows."""

    phase: Phase
    task: str
    control: bool
    cameras: tuple[str, ...]


@dataclass(frozen=True)
class _Sample:
    """What the headset sent last, and its number in the order of the samples."""

    number: int
    pose: geom.Transform3D | None
    buttons: Sequence[float] | None


class TeleopService:
    """Serve one teleop session at a time at ``served_address``, as a model server.

    With no ``auth_token``, a host and a port serve open to each peer that reaches them. The Unix socket is in a
    directory of this user only.

    ``take_sample`` takes each sample of the right controller. A frame request waits for the next sample. When none
    comes within ``silence_limit``, the frame has no pose ("not tracked"), and control goes off. The preparation of a
    session waits for Ready or Cancel up to ``ready_limit``.
    """

    # How often a wait for Ready checks that the model server cancelled the preparation.
    CANCEL_POLL_SEC = 0.1
    # A client opens a session again within its reconnect deadline when the connection drops before its first answer.
    RESUME_LIMIT_SEC = DEFAULT_RECONNECT_DEADLINE

    def __init__(
        self,
        served_address: ServedAddress,
        ready_limit: float,
        silence_limit: float,
        hold_sec: float,
        press_threshold: float = 0.6,
        auth_token: str | None = None,
    ):
        for name, value in (('ready_limit', ready_limit), ('silence_limit', silence_limit)):
            if not 0 < value < math.inf:
                raise ValueError(f'{name} must be finite and positive, got {value}')
        self._binds = served_address
        self._auth_token = auth_token
        self.ready_limit = ready_limit
        self.silence_limit = silence_limit
        self._buttons = ButtonMap(hold_sec, press_threshold)
        self._changed = threading.Condition()
        self._sample = _Sample(0, None, None)
        self._phase = Phase.IDLE
        self._ready: ReadyAnswer | None = None
        self._events: list[Event] = []
        self._task = ''
        # The clutch of the trial, from Ready on. A session that resumes the trial keeps it.
        self._tracker: Tracker | None = None
        # The last camera images of the run, by name, in the order that the client sends them.
        self._images: dict[str, np.ndarray] = {}
        # The camera that the headset shows in this trial. ``None`` is the first camera of the run.
        self._headset_camera: str | None = None
        # The target pose and grip that each frame sends while control is off: the last target, or at first the state
        # of the robot. So every command signal starts at the start of the episode, and the robot stays still.
        self._hold: tuple[geom.Transform3D, float] | None = None
        # The frames that the service answered in this trial.
        self._frames = 0
        # The element that ended the trial. Each later frame request gets it again, because the client can drop it
        # before it plays.
        self._end: dict[str, Any] | None = None
        # The `time.monotonic()` instant until which a new session resumes this trial, after its session dropped
        # before its first answer.
        self._resumable_until: float | None = None
        # A wait ends at once when the service stops.
        self._stopping = False
        self._wire: WebsocketWire | None = None
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

    @property
    def served_address(self) -> ServedAddress:
        """Where the running service serves: for a port of 0, the port that it bound."""
        assert self._wire is not None, 'The teleop service has not started'
        return self._wire.served_address

    def __enter__(self) -> 'TeleopService':
        if isinstance(self._binds, ServedUnixSocket):
            self._prepare_private_directory(self._binds.uds.parent)
        server = ModelServer(lambda: TeleopModel(self), auth_token=self._auth_token)
        self._wire = WebsocketWire(self._binds)
        wires = [self._wire]
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
                self._ready = self._buttons.ready_answer(buttons, now_ns)
            elif self._phase is Phase.RUNNING:
                self._events += self._buttons.events(buttons, now_ns)
            else:
                self._buttons.observe(buttons, now_ns)
            self._changed.notify_all()

    @property
    def phase(self) -> Phase:
        with self._changed:
            self._forget_an_old_dropped_trial()
            return self._phase

    def state(self) -> TrialState:
        with self._changed:
            self._forget_an_old_dropped_trial()
            control = self._tracker is not None and self._tracker.on
            return TrialState(self._phase, self._task, control, tuple(self._images))

    def _forget_an_old_dropped_trial(self) -> None:
        """Show no trial once no session can resume it. Hold the lock."""
        if self._resumable_until is not None and time.monotonic() >= self._resumable_until:
            self._clear_trial()

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
            self._ready = ReadyAnswer.CANCELLED
            self._changed.notify_all()

    @staticmethod
    def _view(image: np.ndarray | None, lines: Sequence[str]) -> np.ndarray:
        """``image``, or a black image, with ``lines`` of text at its top."""
        width, height = VIEW_SIZE
        blank = (height, width, 3)
        # A copy: the text goes on the view, not on the image that the service holds.
        view = np.zeros(blank, dtype=np.uint8) if image is None else np.array(image, dtype=np.uint8, order='C')
        for row, line in enumerate(lines):
            origin = (10, 30 + 36 * row)
            cv2.putText(view, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(view, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
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
        return self._view(image, [state.task, control, 'Hold B: Complete. Hold the stick: Give up'])

    def _headset_image(self) -> np.ndarray | None:
        if self._headset_camera is not None:
            return self._images.get(self._headset_camera)
        return next(iter(self._images.values()), None)

    def resume_dropped_trial(self, headset_camera: str | None) -> ResumeOutcome:
        """Resume the trial whose session dropped before its first answer, as the next session of its client.

        The client opens such a session again within its reconnect deadline, and the teleoperator pressed Ready already.
        """
        with self._changed:
            self._forget_an_old_dropped_trial()
            if self._resumable_until is None:
                return ResumeOutcome.NO_DROPPED_TRIAL
            self._resumable_until, self._headset_camera = None, headset_camera
            return ResumeOutcome.RESUMED

    def wait_for_ready(
        self, operator: geom.Transform3D | None, headset_camera: str | None, cancelled: threading.Event
    ) -> None:
        """Start a new trial, and wait up to the ready limit for Ready. The clutch maps the hand with ``operator``, and
        the headset shows ``headset_camera``, or the first camera for ``None``.

        Raise ``TrialNotStarted`` for Cancel or for no Ready within the ready limit, and ``PreparationCancelled`` when
        ``cancelled`` is set or the service stops.
        """
        deadline = time.monotonic() + self.ready_limit
        with self._changed:
            self._clear_trial()
            self._phase, self._ready, self._headset_camera = Phase.READY_WAIT, None, headset_camera
            while (
                self._ready is None
                and not self._stopping
                and not cancelled.is_set()
                and (left := deadline - time.monotonic()) > 0
            ):
                self._changed.wait(min(left, self.CANCEL_POLL_SEC))
            ready, self._phase = self._ready, Phase.RUNNING if self._ready is ReadyAnswer.READY else Phase.IDLE
            if ready is ReadyAnswer.READY:
                self._tracker = Tracker(operator)
        if ready is ReadyAnswer.READY:
            return
        if self._stopping or cancelled.is_set():
            raise PreparationCancelled('the session closed before the teleoperator was ready')
        if ready is ReadyAnswer.CANCELLED:
            raise TrialNotStarted('the teleoperator cancelled the trial')
        raise TrialNotStarted('the teleoperator was not ready within the ready limit of the service')

    def _next_sample(self, seen: int, deadline: float) -> _Sample:
        """The sample after sample number ``seen``, or a sample with no pose when none comes before ``deadline``. Hold
        the lock."""
        while self._sample.number == seen and not self._stopping and (left := deadline - time.monotonic()) > 0:
            self._changed.wait(left)
        return self._sample if self._sample.number != seen else _Sample(seen, None, None)

    def next_frame(self, request: Mapping[str, Any], cameras: Sequence[str]) -> dict[str, Any]:
        """Take the state of the robot and the images of ``cameras`` in ``request``, and answer with the element of the
        next frame: the target of the teleoperator, with the end of the trial once the teleoperator ends it."""
        robot_pose = geom.Transform3D.from_vector(np.asarray(request[EE_POSE], dtype=float), QUAT)
        deadline = time.monotonic() + self.silence_limit
        with self._changed:
            if self._phase is Phase.ENDED:
                assert self._end is not None
                return self._end
            if self._phase is not Phase.RUNNING:
                raise ValueError(f'a frame request came in the {self._phase} phase')
            self._task = str(request[TASK])
            self._images = {name: request[name] for name in cameras}
            if self._hold is None:
                self._hold = robot_pose, float(np.asarray(request[GRIP], dtype=float).reshape(-1)[0])
            sample = self._next_sample(self._sample.number, deadline)
            events, self._events = self._events, []
            assert self._tracker is not None, 'a running trial has the clutch of its Ready'
            element = self._apply(sample, events, self._tracker, robot_pose)
            self._frames += 1
            if END in element:
                self._phase, self._end = Phase.ENDED, element
            return element

    def _apply(
        self, sample: _Sample, events: Sequence[Event], tracker: Tracker, robot_pose: geom.Transform3D
    ) -> dict[str, Any]:
        """The frame element of ``sample`` and ``events``. Control starts from ``robot_pose``, so the robot does not
        jump. Hold the lock."""
        target = None if sample.pose is None else tracker.update(sample.pose)
        for event in events:
            match event:
                case Event.CLUTCH if tracker.on:
                    tracker.turn_off()
                case Event.CLUTCH if sample.pose is not None:
                    tracker.turn_on(robot_pose, sample.pose)
                    target = tracker.update(sample.pose)
                case Event.COMPLETE:
                    return {**self._hold_element(), END: PolicyEnd.COMPLETE.value}
                case Event.GIVE_UP:
                    return {**self._hold_element(), END: PolicyEnd.GIVE_UP.value}
        if target is None and tracker.on:
            tracker.turn_off()
        if tracker.on and target is not None:
            self._hold = target, 0.0 if sample.buttons is None else ButtonMap.grip(sample.buttons)
        return self._hold_element()

    def _hold_element(self) -> dict[str, Any]:
        """The element that sends the hold target. Hold the lock."""
        assert self._hold is not None
        pose, grip = self._hold
        return {ACTION: np.append(pose.as_vector(QUAT), grip)}

    def _clear_trial(self) -> None:
        """Show no trial. Hold the lock."""
        self._phase, self._events = Phase.IDLE, []
        self._task, self._tracker, self._images = '', None, {}
        self._headset_camera, self._hold, self._frames, self._resumable_until = None, None, 0, None
        self._end = None

    def end_session(self) -> None:
        """End the session of the trial. A trial that answered at most one frame stays for a client that opens its
        session again, because the client cannot know that the one answer came."""
        # TODO: `Session.close` does not tell how the session ended, so a clean end after at most one frame also keeps
        #  the trial. A new session within `RESUME_LIMIT_SEC` then resumes it and skips Ready, so in UMI mode the robot
        #  follows the hand at once. We think this is improbable. Fix it when it occurs, or when we think that it is
        #  more probable.
        with self._changed:
            if self._phase is Phase.RUNNING and self._frames <= 1:
                self._resumable_until = time.monotonic() + self.RESUME_LIMIT_SEC
                return
            self._clear_trial()


class TeleopSession(Session):
    """One trial of the teleop service: each inference call is one frame."""

    def __init__(
        self,
        service: TeleopService,
        settings: SessionParams,
        client_stack: dict[str, Any],
        on_close: Callable[[], None],
    ):
        super().__init__(client_stack)
        self._service = service
        self._cameras = settings.cameras
        self._on_close = on_close

    def infer(self, observation: Any) -> list[dict[str, Any]]:
        return [self._service.next_frame(observation, self._cameras)]

    def close(self) -> None:
        self._service.end_session()
        self._on_close()


class TeleopModel(Model):
    """The teleop service as a model server serves it. It prepares one session at a time.

    ``ModelServer`` calls it on one worker thread only.
    """

    # The schedule plays each answer of one element at once, and asks for the next frame at once.
    FRAME_FPS = 100.0

    def __init__(self, service: TeleopService):
        super().__init__(
            parameters=SessionParams().to_json(),
            metadata={
                policy_keys.TYPE: 'teleop',
                'ready_limit_sec': service.ready_limit,
                'silence_limit_sec': service.silence_limit,
            },
        )
        self._service = service
        self._open = False

    @classmethod
    def client_stack(cls, settings: SessionParams) -> dict[str, Any]:
        """The client stack of a session: one frame for each call, the state and the images of the robot at
        ``GRIPPER_SITE``, the target as a command, and the end of the trial."""
        # TODO: A sim run that does not charge inference time holds its clock during each call, so a person cannot
        #  control it. Refuse such a run.
        schedule = ChunkedSchedule(
            fps=cls.FRAME_FPS, record_stats=False, inference_lead=ConstantDuration(1 / cls.FRAME_FPS)
        )
        observation = StateObservationCodec({EE_POSE: {keys.EE_POSE: 7}, GRIP: {keys.GRIP: 1}}, task_field=TASK) & (
            image_observation({name: (name, VIEW_SIZE) for name in settings.cameras})
        )
        action = AbsolutePositionAction(keys.TARGET_EE_POSE, keys.TARGET_GRIP)
        # A client stack that states no frame gets no pose.
        frame = ChangeEEFrame(EEFrame(site=GRIPPER_SITE))
        codec = frame | (observation & action & EndCommand()) | EncodeImages(quality=settings.jpeg_quality)
        return spec.sequence(EndOnField().to_spec(), schedule.to_spec(), codec.to_spec())

    def prepare_session(self, params: dict[str, Any]) -> TeleopSession:
        if self._open:
            raise RuntimeError('the teleop service serves one session at a time')
        settings = SessionParams.from_json(**params)
        self._open = True
        try:
            if self._service.resume_dropped_trial(settings.headset_camera) is ResumeOutcome.NO_DROPPED_TRIAL:
                self._service.wait_for_ready(settings.operator(), settings.headset_camera, self.preparation_cancelled)
        except BaseException:
            self._open = False
            raise
        return TeleopSession(self._service, settings, self.client_stack(settings), self._release)

    def _release(self) -> None:
        self._open = False
