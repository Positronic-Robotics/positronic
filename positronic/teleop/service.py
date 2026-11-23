"""The teleop service: the headset on one side, one teleop session of a ``RemotePolicy`` on the other.

``positronic teleop`` runs it beside a run whose policy dials its socket. It is a model server: the preparation of a
session waits for Ready, and each frame request gets the next target of the teleoperator. Each session declares stock
components only in its client stack. The service keeps the trial state and the clutch, so the input of the
teleoperator stays in this process. The headset page shows the camera images of the run and the state of the trial.
The teleoperator uses the buttons of the right controller. In a bimanual session, the left controller drives the left
arm.
"""

import _thread
import logging
import os
import stat
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, replace
from enum import Enum, IntEnum, StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from positronic_model_server.server import Model, ModelServer, PreparationCancelled, Session
from positronic_model_server.server_wire import ServedAddress
from positronic_model_server.websocket_wire import ServedUnixSocket, WebsocketWire

from positronic import geom, keys
from positronic.dataset.transforms.image import Resample, ResizeMode, transform_frame
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame
from positronic.drivers.webxr import LEFT, RIGHT
from positronic.gui.video_stream import AnnexBDecoder
from positronic.offboard.client import DEFAULT_RECONNECT_DEADLINE
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd
from positronic.policy.codecs import (
    AbsolutePositionAction,
    BimanualCartesianAction,
    ChangeEEFrame,
    EncodeImages,
    EncodeVideo,
    EndCommand,
    StateObservationCodec,
)
from positronic.policy.codecs.image import image_observation
from positronic.policy.codecs.metadata import ACTION
from positronic.policy.keys import END
from positronic.policy.processors import (
    ConstantDuration,
    EndOnField,
    InferenceSchedule,
    RequestTiming,
    TrajectoryMerge,
    TrajectoryPlayer,
)
from positronic.policy.sequential import Sequential
from positronic.utils import check_finite_positive

from .clutch import CONTROLLER_AXES, Tracker
from .protocol import EE_POSE, GRIP, TASK, VIEW_SIZE, Arms, SessionParams, camera_names

logger = logging.getLogger(__name__)

QUAT = geom.Rotation.Representation.QUAT


class Button(IntEnum):
    """The index of each button in the values that the headset page sends for a controller, in the xr-standard
    gamepad layout."""

    TRIGGER = 0
    STICK = 3
    A = 4
    B = 5
    # The left controller has X and Y at the indices of A and B.
    X = 4


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
    """The buttons of one controller as the answers and the events of a session.

    - A: control on or off.
    - B: Ready while the service waits for it. A hold of B in the trial is Complete.
    - The stick: Cancel while the service waits for Ready. A hold of the stick is Give up.
    - The trigger: the grip.

    It reads only ``buttons``: the left controller reads only X, because the headset page uses its Y.
    """

    def __init__(
        self,
        hold_sec: float,
        press_threshold: float = 0.6,
        buttons: Sequence[Button] = (Button.STICK, Button.A, Button.B),
    ):
        if not 0 < press_threshold <= 1:
            raise ValueError(f'press_threshold must be in (0, 1], got {press_threshold}')
        check_finite_positive(hold_sec, 'hold_sec')
        self._hold_ns = round(hold_sec * 1e9)
        # A button whose value is at least this is down.
        self._press_threshold = press_threshold
        # The time at which each button went down, while it is down.
        self._down_since: dict[Button, int] = {}
        # The buttons whose press did something already, until they go up.
        self._used: set[Button] = set()
        self._read = tuple(buttons)

    def _new_presses(self, buttons: Sequence[float], now_ns: int) -> set[Button]:
        """Record the state of the buttons, and return the buttons that went down since the previous sample."""
        new_presses = set()
        for button in self._read:
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

    def cancel_holds(self) -> None:
        """Cancel the holds of the buttons that are down. Each of them presses again only after it goes up."""
        self._used.update(self._down_since)

    def _is_held(self, button: Button, now_ns: int) -> bool:
        since = self._down_since.get(button)
        return since is not None and button not in self._used and now_ns - since >= self._hold_ns

    def ready_answer(self, buttons: Sequence[float], now_ns: int) -> ReadyAnswer | None:
        new_presses = self._new_presses(buttons, now_ns)
        # A button that is down before the trial runs does not hold in it.
        self._used.update(self._down_since)
        if Button.B in new_presses:
            return ReadyAnswer.READY
        if Button.STICK in new_presses:
            return ReadyAnswer.CANCELLED
        return None

    def events(self, buttons: Sequence[float], now_ns: int) -> list[Event]:
        new_presses = self._new_presses(buttons, now_ns)
        events = []
        if Button.A in new_presses:
            events.append(Event.CLUTCH)
        if self._is_held(Button.B, now_ns):
            self._used.add(Button.B)
            events.append(Event.COMPLETE)
        if self._is_held(Button.STICK, now_ns):
            self._used.add(Button.STICK)
            events.append(Event.GIVE_UP)
        return events

    @staticmethod
    def grip(buttons: Sequence[float]) -> float:
        return float(buttons[Button.TRIGGER])


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


# The use of Y on the left controller, at the bottom of the view
CAMERA_HINT = 'Y: next camera. Hold Y: panel'
# The width of the column of the view that shows the other cameras of the session, half the width of the main camera
OTHERS_WIDTH = VIEW_SIZE[0] // 2

# The controller that drives each arm, by the arm name of its channels. The one arm of a one-arm session has none.
ARM_CONTROLLERS: dict[str | None, str] = {None: RIGHT, keys.LEFT_ARM: LEFT, keys.RIGHT_ARM: RIGHT}


@dataclass(frozen=True)
class CameraState:
    """The cameras that the headset page offers: the cameras of the run, or of the station before a run sends one;
    the ones whose images come now; the one that the headset shows; and the one that the teleoperator chose, or
    ``None`` for the camera of the session."""

    names: tuple[str, ...]
    live: tuple[str, ...]
    shown: str | None
    chosen: str | None


@dataclass(frozen=True)
class _Sample:
    """What the headset sent last, by controller, and its number in the order of the samples. A controller that the
    headset does not track is absent or ``None``."""

    number: int
    poses: Mapping[str, geom.Transform3D | None]
    buttons: Mapping[str, Sequence[float] | None]


class TeleopService:
    """Serve one teleop session at a time at ``served_address``, as a model server.

    With no ``auth_token``, a host and a port serve open to each peer that reaches them. The Unix socket is in a
    directory of this user only.

    ``take_sample`` takes each sample of the controllers. A frame request waits for the next sample. When none comes
    within ``silence_limit``, the frame has no pose ("not tracked"), and control goes off. The preparation of a
    session waits for Ready or Cancel up to ``ready_limit``.
    """

    # How often a wait for Ready checks that the model server cancelled the preparation.
    CANCEL_POLL_SEC = 0.1
    # A camera is live while its last image is younger than this. Images come only while a trial runs, when the
    # teleoperator is in the headset, so the window spans the gap between trials for the page before Enter AR.
    LIVE_SEC = 30.0
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
        cameras: Sequence[str] = (),
        controller_axes: geom.Rotation | None = CONTROLLER_AXES,
    ):
        for name, value in (('ready_limit', ready_limit), ('silence_limit', silence_limit)):
            check_finite_positive(value, name)
        self._binds = served_address
        self._auth_token = auth_token
        # The cameras of the station: a session that names no cameras gets these
        self.station_cameras = camera_names(cameras)
        # The axes of the controller in the frame of the headset, or None for the axes of the headset (see `Tracker`)
        self.controller_axes = controller_axes
        self.ready_limit = ready_limit
        self.silence_limit = silence_limit
        self._buttons = {
            RIGHT: ButtonMap(hold_sec, press_threshold),
            LEFT: ButtonMap(hold_sec, press_threshold, (Button.X,)),
        }
        self._changed = threading.Condition()
        self._sample = _Sample(0, {}, {})
        self._phase = Phase.IDLE
        self._ready: ReadyAnswer | None = None
        # The events since the previous frame, each with the controller that made it
        self._events: list[tuple[str, Event]] = []
        self._task = ''
        # The clutch of each arm of the trial, from Ready on. A session that resumes the trial keeps them.
        self._trackers: dict[str | None, Tracker] = {}
        # The last camera images of the run, by name, in the order that the client sends them.
        self._images: dict[str, np.ndarray] = {}
        # The `time.monotonic()` instant of the last image of each camera that the run sent. It holds over trials.
        self._seen: dict[str, float] = {}
        # The cameras that the session of this trial sends, from the wait for Ready on
        self._trial_cameras: tuple[str, ...] = ()
        # The camera that the session of this trial names for the headset. ``None`` is the first camera of the run.
        self._headset_camera: str | None = None
        # The camera that the teleoperator chose on the headset page. It holds over trials, and goes before the camera
        # of the session while the run sends it.
        self._chosen_camera: str | None = None
        # The target pose and grip of each arm that each frame sends while control of the arm is off: the last target,
        # or at first the state of the arm. So every command signal starts at the start of the episode, and the robot
        # stays still.
        self._hold: dict[str | None, tuple[geom.Transform3D, float]] = {}
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

    def take_sample(
        self, poses: Mapping[str, geom.Transform3D | None], buttons: Mapping[str, Sequence[float] | None]
    ) -> None:
        """Take one sample of the controllers, by ``LEFT`` and ``RIGHT``. A controller that the headset does not track
        is absent or ``None``."""
        now_ns = time.monotonic_ns()
        # A controller with fewer button values than the maps read cannot press the clutch, so it drives no arm
        controllers = {name for name, values in buttons.items() if values is not None and len(values) > max(Button)}
        poses = {name: pose for name, pose in poses.items() if name in controllers}
        buttons = {name: buttons[name] for name in controllers}
        with self._changed:
            self._sample = _Sample(self._sample.number + 1, poses, buttons)
            for controller, button_map in self._buttons.items():
                values = buttons.get(controller)
                if values is None:
                    button_map.cancel_holds()
                    continue
                if self._phase is Phase.READY_WAIT and self._ready is None:
                    self._ready = button_map.ready_answer(values, now_ns)
                elif self._phase is Phase.RUNNING:
                    self._events += [(controller, event) for event in button_map.events(values, now_ns)]
                else:
                    button_map.observe(values, now_ns)
            self._changed.notify_all()

    @property
    def phase(self) -> Phase:
        with self._changed:
            self._forget_an_old_dropped_trial()
            return self._phase

    def state(self) -> TrialState:
        with self._changed:
            self._forget_an_old_dropped_trial()
            control = any(tracker.on for tracker in self._trackers.values())
            return TrialState(self._phase, self._task, control, tuple(self._images))

    def _forget_an_old_dropped_trial(self) -> None:
        """Show no trial once no session can resume it. Hold the lock."""
        if self._resumable_until is not None and time.monotonic() >= self._resumable_until:
            self._clear_trial()

    def cameras(self) -> CameraState:
        with self._changed:
            now = time.monotonic()
            recent = tuple(name for name, seen_at in self._seen.items() if now - seen_at < self.LIVE_SEC)
            # Between trials: the cameras of the station, and the cameras of the last run that are still live
            names = self._trial_cameras or tuple(dict.fromkeys((*self.station_cameras, *recent)))
            live = tuple(name for name in names if name in recent)
            return CameraState(names, live, self._shown_camera(), self._chosen_camera)

    def choose_camera(self, name: str | None) -> None:
        """Show camera ``name`` in the headset from now on, or the camera of the session for ``None``. Raise
        ``ValueError`` for a camera that neither the run nor the station has."""
        with self._changed:
            if name is not None and name not in self._trial_cameras and name not in self.station_cameras:
                known = sorted({*self._trial_cameras, *self.station_cameras})
                raise ValueError(f'no camera {name!r}: the run and the station have {known}')
            self._chosen_camera = name

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
    def _text(view: np.ndarray, origin: tuple[int, int], line: str, scale: float) -> None:
        cv2.putText(view, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)

    @classmethod
    def _view(
        cls,
        image: np.ndarray | None,
        lines: Sequence[str],
        footer: Sequence[str] = (),
        others: Sequence[tuple[str, np.ndarray | None]] = (),
    ) -> np.ndarray:
        """``image``, or a black image, with ``lines`` of text at its top and the smaller ``footer`` at its bottom.

        With ``others``, the view has a column on its right with one named cell for each of them, and ``image`` fits
        the left part of the view size.
        """
        width, height = VIEW_SIZE
        if others:
            view = np.zeros((height, width + OTHERS_WIDTH, 3), dtype=np.uint8)
            if image is not None:
                view[:, :width] = transform_frame(image, width, height, mode=ResizeMode.PAD, resample=Resample.AREA)
            cell_height = height // len(others)
            for row, (name, other) in enumerate(others):
                top = row * cell_height
                if other is not None:
                    cell = transform_frame(
                        other, OTHERS_WIDTH, cell_height, mode=ResizeMode.PAD, resample=Resample.AREA
                    )
                    view[top : top + cell_height, width:] = cell
                cls._text(view, (width + 6, top + 18), name.removeprefix(keys.IMAGE_PREFIX), 0.5)
        elif image is None:
            view = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # A copy: the text goes on the view, not on the image that the service holds.
            view = np.array(image, dtype=np.uint8, order='C')
        bottom = view.shape[0]
        for row, line in enumerate(lines):
            cls._text(view, (10, 26 + 30 * row), line, 0.75)
        for row, line in enumerate(reversed(footer)):
            cls._text(view, (10, bottom - 10 - 22 * row), line, 0.55)
        return view

    def view(self, camera: bool = True) -> np.ndarray:
        """The image for the headset: the camera image of the run, or black with no ``camera``, with the state of the
        trial. With more than one camera, a column on the right shows the other cameras."""
        with self._changed:
            state = self.state()
            shown = self._shown_camera()
            image = self._headset_image() if camera else None
            names = self.cameras().names
            main = shown if shown in names else next(iter(names), None)
            shows_images = state.phase is Phase.RUNNING and camera
            others = [(name, self._images.get(name) if shows_images else None) for name in names if name != main]
        footer = [CAMERA_HINT]
        match state.phase:
            case Phase.IDLE:
                return self._view(None, ['Waiting for a trial'], footer, others)
            case Phase.READY_WAIT:
                return self._view(None, ['Human trial', 'B: Ready', 'Stick: Cancel'], footer, others)
            case Phase.ENDED:
                return self._view(None, ['The trial ended', 'Give the verdict on the console'], footer, others)
        if shown is not None:
            footer = [f'Camera: {shown}', *footer]
        control = self._control_line()
        return self._view(image, [state.task, control, 'Hold B: Complete. Hold the stick: Give up'], footer, others)

    def _control_line(self) -> str:
        """The state of the clutch of each arm, and the button that changes it."""
        clutch_labels = {LEFT: 'X', RIGHT: 'A'}
        parts = []
        with self._changed:
            for arm, tracker in self._trackers.items():
                name = 'Control' if arm is None else arm.capitalize()
                clutch = clutch_labels[ARM_CONTROLLERS[arm]]
                parts.append(f'{name} on ({clutch}: off)' if tracker.on else f'{name} off ({clutch}: on)')
        return '. '.join(parts)

    def _shown_camera(self) -> str | None:
        """The camera that the headset shows. Hold the lock."""
        for name in (self._chosen_camera, self._headset_camera):
            if name is not None and name in self._trial_cameras:
                return name
        return next(iter(self._trial_cameras), None)

    def _headset_image(self) -> np.ndarray | None:
        shown = self._shown_camera()
        return None if shown is None else self._images.get(shown)

    def resume_dropped_trial(self, settings: SessionParams) -> ResumeOutcome:
        """Resume the trial whose session dropped before its first answer, as the next session of its client.

        The client opens such a session again within its reconnect deadline, and the teleoperator pressed Ready already.
        """
        with self._changed:
            self._forget_an_old_dropped_trial()
            if self._resumable_until is None:
                return ResumeOutcome.NO_DROPPED_TRIAL
            self._resumable_until, self._trial_cameras = None, settings.cameras
            self._headset_camera = settings.headset_camera
            return ResumeOutcome.RESUMED

    def wait_for_ready(self, settings: SessionParams, cancelled: threading.Event) -> None:
        """Start a new trial of a session with ``settings``, and wait up to the ready limit for Ready.

        Raise ``TrialNotStarted`` for Cancel or for no Ready within the ready limit, and ``PreparationCancelled`` when
        ``cancelled`` is set or the service stops.
        """
        deadline = time.monotonic() + self.ready_limit
        with self._changed:
            self._clear_trial()
            self._phase, self._ready = Phase.READY_WAIT, None
            self._trial_cameras, self._headset_camera = settings.cameras, settings.headset_camera
            while (
                self._ready is None
                and not self._stopping
                and not cancelled.is_set()
                and (left := deadline - time.monotonic()) > 0
            ):
                self._changed.wait(min(left, self.CANCEL_POLL_SEC))
            ready = self._ready
            if ready is ReadyAnswer.READY:
                self._phase = Phase.RUNNING
                self._trackers = {
                    arm: Tracker(settings.operator(), self.controller_axes) for arm in settings.arms.value
                }
            else:
                self._clear_trial()
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
        return self._sample if self._sample.number != seen else _Sample(seen, {}, {})

    def next_frame(self, request: Mapping[str, Any], cameras: Sequence[str]) -> dict[str, Any]:
        """Take the state of the arms and the images of ``cameras`` in ``request``, and answer with the element of the
        next frame: the target of the teleoperator, with the end of the trial once the teleoperator ends it."""
        deadline = time.monotonic() + self.silence_limit
        with self._changed:
            if self._phase is Phase.ENDED:
                assert self._end is not None
                return self._end
            if self._phase is not Phase.RUNNING:
                raise ValueError(f'a frame request came in the {self._phase} phase')
            self._task = str(request[TASK])
            self._images = {name: request[name] for name in cameras}
            self._seen.update(dict.fromkeys(cameras, time.monotonic()))
            assert self._trackers, 'a running trial has the clutches of its Ready'
            robot = {arm: self._arm_state(request, arm) for arm in self._trackers}
            if not self._hold:
                self._hold = dict(robot)
            sample = self._next_sample(self._sample.number, deadline)
            events, self._events = self._events, []
            element = self._apply(sample, events, {arm: pose for arm, (pose, _) in robot.items()})
            self._frames += 1
            if END in element:
                self._phase, self._end = Phase.ENDED, element
            return element

    @staticmethod
    def _arm_state(request: Mapping[str, Any], arm: str | None) -> tuple[geom.Transform3D, float]:
        """The pose and the grip of ``arm`` in a frame request."""
        pose = np.asarray(request[keys.arm_channel(EE_POSE, arm)], dtype=float)
        grip = np.asarray(request[keys.arm_channel(GRIP, arm)], dtype=float).reshape(-1)[0]
        return geom.Transform3D.from_vector(pose, QUAT), float(grip)

    def _apply(
        self, sample: _Sample, events: Sequence[tuple[str, Event]], robot_poses: Mapping[str | None, geom.Transform3D]
    ) -> dict[str, Any]:
        """The frame element of ``sample`` and ``events``. Control of each arm starts from its pose in
        ``robot_poses``, so the robot does not jump. Hold the lock."""
        for _, event in events:
            match event:
                case Event.COMPLETE:
                    return {**self._hold_element(), END: PolicyEnd.COMPLETE.value}
                case Event.GIVE_UP:
                    return {**self._hold_element(), END: PolicyEnd.GIVE_UP.value}
        for arm in self._trackers:
            self._drive_arm(arm, sample, events, robot_poses[arm])
        return self._hold_element()

    def _drive_arm(
        self, arm: str | None, sample: _Sample, events: Sequence[tuple[str, Event]], robot_pose: geom.Transform3D
    ) -> None:
        """Apply the clutch presses of the controller of ``arm`` in ``events``, and move the hold target of ``arm`` to
        the hand. Hold the lock."""
        tracker, controller = self._trackers[arm], ARM_CONTROLLERS[arm]
        pose, buttons = sample.poses.get(controller), sample.buttons.get(controller)
        target = None if pose is None else tracker.update(pose)
        for _ in [event for source, event in events if source == controller and event is Event.CLUTCH]:
            if tracker.on:
                tracker.turn_off()
            elif pose is not None:
                tracker.turn_on(robot_pose, pose)
                target = tracker.update(pose)
        if target is None and tracker.on:
            tracker.turn_off()
        if tracker.on and target is not None:
            self._hold[arm] = target, 0.0 if buttons is None else ButtonMap.grip(buttons)

    def _hold_element(self) -> dict[str, Any]:
        """The element that sends the hold targets, arm after arm. Hold the lock."""
        assert self._hold, 'the first frame request sets the hold targets'
        return {ACTION: np.concatenate([np.append(pose.as_vector(QUAT), grip) for pose, grip in self._hold.values()])}

    def _clear_trial(self) -> None:
        """Show no trial. Hold the lock."""
        self._phase, self._events = Phase.IDLE, []
        self._task, self._trackers, self._images, self._trial_cameras = '', {}, {}, ()
        self._headset_camera, self._hold, self._frames, self._resumable_until = None, {}, 0, None
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
        # The decoder of each camera stream, for a client that sends H.264
        self._decoders = {} if settings.video_bitrate is None else {name: AnnexBDecoder() for name in settings.cameras}

    def infer(self, observation: Any) -> list[dict[str, Any]]:
        images = {name: decoder.decode(observation[name]) for name, decoder in self._decoders.items()}
        return [self._service.next_frame({**observation, **images}, self._cameras)]

    def close(self) -> None:
        self._service.end_session()
        self._on_close()


class TeleopModel(Model):
    """The teleop service as a model server serves it. It prepares one session at a time.

    ``ModelServer`` calls it on one worker thread only.
    """

    # The schedule plays each answer of one element at once. It asks for the next frame at once, or on the frame grid
    # of an H.264 session.
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
        """The client stack of a session: one frame for each call, the state of the arms at ``GRIPPER_SITE`` and the
        images of the robot, the targets as commands, and the end of the trial."""
        # TODO: A sim run that does not charge inference time holds its clock during each call, so a person cannot
        #  control it. Refuse such a run.
        if settings.video_bitrate is None:
            images = EncodeImages(quality=settings.jpeg_quality)
            timing = RequestTiming.before_end()
        else:
            images = EncodeVideo([[name] for name in settings.cameras], settings.video_bitrate, settings.video_fps)
            # The encoder gives each call one frame of a stream at `video_fps`, so calls come at most that often.
            timing = RequestTiming.every(1 / settings.video_fps)
        schedule = Sequential(
            TrajectoryPlayer(replan_on_error=True, record_stats=False),
            InferenceSchedule(
                cls.FRAME_FPS,
                request_timing=timing,
                merge=TrajectoryMerge.after_previous(),
                latency_estimate=ConstantDuration(1 / cls.FRAME_FPS),
            ),
        )
        state, poses = {}, []
        for arm in settings.arms.value:
            pose = f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.EE_POSE_SUFFIX}'
            command = keys.arm_channel(keys.ROBOT_COMMAND, arm)
            state[keys.arm_channel(EE_POSE, arm)] = {pose: 7}
            state[keys.arm_channel(GRIP, arm)] = {keys.arm_channel(keys.GRIP, arm): 1}
            poses += [pose, f'{command}{keys.POSE_SUFFIX}', command]
        observation = StateObservationCodec(state, task_field=TASK) & (
            image_observation({name: (name, VIEW_SIZE) for name in settings.cameras})
        )
        match settings.arms:
            case Arms.ONE:
                action = AbsolutePositionAction(keys.TARGET_EE_POSE, keys.TARGET_GRIP)
            case Arms.BIMANUAL:
                action = BimanualCartesianAction()
        frame = ChangeEEFrame(EEFrame(site=GRIPPER_SITE), tuple(poses))
        codec = frame | (observation & action & EndCommand()) | images
        return Sequential(EndOnField(), schedule, codec).to_spec()

    def prepare_session(self, params: dict[str, Any]) -> TeleopSession:
        if self._open:
            raise RuntimeError('the teleop service serves one session at a time')
        settings = SessionParams.from_json(**params)
        if not settings.cameras:
            settings = replace(settings, cameras=self._service.station_cameras)
        self._open = True
        try:
            if self._service.resume_dropped_trial(settings) is ResumeOutcome.NO_DROPPED_TRIAL:
                self._service.wait_for_ready(settings, self.preparation_cancelled)
        except BaseException:
            self._open = False
            raise
        return TeleopSession(self._service, settings, self.client_stack(settings), self._release)

    def _release(self) -> None:
        self._open = False
