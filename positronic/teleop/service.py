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
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
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
from positronic.drivers.webxr import LEFT, RIGHT, ControllerButtons
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

    It reads only ``buttons``: the left controller reads only X, because the headset page uses its Y. A press is a step
    of the press count of the button, and a hold is a value at ``hold_threshold`` or above for ``hold_sec``.
    """

    def __init__(
        self,
        hold_sec: float,
        hold_threshold: float = 0.6,
        buttons: Sequence[Button] = (Button.STICK, Button.A, Button.B),
    ):
        if not 0 < hold_threshold <= 1:
            raise ValueError(f'hold_threshold must be in (0, 1], got {hold_threshold}')
        check_finite_positive(hold_sec, 'hold_sec')
        self._hold_ns = round(hold_sec * 1e9)
        # A button whose value is at least this is down for a hold. A press comes from the press count alone.
        self._hold_threshold = hold_threshold
        # The time at which each button went down, while it is down.
        self._down_since: dict[Button, int] = {}
        # The buttons whose press did something already, until they go up.
        self._used: set[Button] = set()
        # The press counts of the previous sample. The first sample sets them, so a page that pressed before it
        # connected presses nothing.
        self._presses: tuple[int, ...] | None = None
        self._read = tuple(buttons)

    def _new_presses(self, buttons: ControllerButtons, now_ns: int) -> dict[Button, int]:
        """Record the state of the buttons, and return the number of presses of each button since the previous
        sample. A sample with a count that went down, or with another number of counts, is of a page that loaded
        again: all its counts become the counts to count from, and it presses nothing."""
        for button in self._read:
            if buttons.values[button] < self._hold_threshold:
                self._down_since.pop(button, None)
                self._used.discard(button)
            elif button not in self._down_since:
                self._down_since[button] = now_ns
        previous, self._presses = self._presses, buttons.presses
        if previous is None or len(previous) != len(buttons.presses):
            return {}
        if any(now < before for now, before in zip(buttons.presses, previous, strict=True)):
            return {}
        return {
            button: buttons.presses[button] - previous[button]
            for button in self._read
            if buttons.presses[button] > previous[button]
        }

    def observe(self, buttons: ControllerButtons, now_ns: int) -> None:
        """Record the state of the buttons, outside a trial. A press here does nothing, and a button that is down here
        does not hold in the next."""
        self._new_presses(buttons, now_ns)
        self._used.update(self._down_since)

    def cancel_holds(self) -> None:
        """Cancel the holds of the buttons that are down. Each of them holds again only after it goes up."""
        self._used.update(self._down_since)

    def _is_held(self, button: Button, now_ns: int) -> bool:
        since = self._down_since.get(button)
        return since is not None and button not in self._used and now_ns - since >= self._hold_ns

    def ready_answer(self, buttons: ControllerButtons, now_ns: int) -> ReadyAnswer | None:
        new_presses = self._new_presses(buttons, now_ns)
        # A button that is down before the trial runs does not hold in it.
        self._used.update(self._down_since)
        if Button.B in new_presses:
            return ReadyAnswer.READY
        if Button.STICK in new_presses:
            return ReadyAnswer.CANCELLED
        return None

    def events(self, buttons: ControllerButtons, now_ns: int) -> list[Event]:
        """The events of the sample ``buttons``. An odd number of presses of A since the previous sample gives one
        clutch event. An even number gives two, so control that is on turns off and on again, at the new place of the
        hand."""
        presses_of_a = self._new_presses(buttons, now_ns).get(Button.A, 0)
        events = [Event.CLUTCH] * (0 if presses_of_a == 0 else 1 if presses_of_a % 2 else 2)
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
    # A run with a teleop episode started, and its trial does not wait for Ready yet.
    HEADS_UP = 'heads_up'
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
    # A sample that tracks a controller came within the silence limit
    tracked: bool


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
class CameraFeed:
    """The images of one camera that reached the service: their rate in Hz over the last ``TeleopService.FEED_SEC``,
    and the seconds since the last one."""

    rate_hz: float
    age_sec: float


@dataclass(frozen=True)
class _Sample:
    """What the headset sent last, by controller, and its `time.monotonic_ns()` instant. A controller that the headset
    does not track is absent or ``None``."""

    taken_ns: int
    poses: Mapping[str, geom.Transform3D | None]
    buttons: Mapping[str, ControllerButtons]


@dataclass(frozen=True)
class Badge:
    """A line of the view in black on a box of ``colour``, an RGB colour."""

    text: str
    colour: tuple[int, int, int]


@dataclass(frozen=True)
class ViewText:
    """The text of the headset view: the lines at its top, each plain text or a badge, and the smaller lines at its
    bottom."""

    lines: tuple[str | Badge, ...]
    footer: tuple[str, ...]


class TeleopService:
    """Serve one teleop session at a time at ``served_address``, as a model server.

    With no ``auth_token``, a host and a port serve open to each peer that reaches them. The Unix socket is in a
    directory of this user only.

    ``take_sample`` takes each sample of the controllers. A frame request answers at once with the newest sample.
    When that sample is older than ``silence_limit``, the frame has no pose ("not tracked"), and control goes off. The
    preparation of a session waits for Ready or Cancel up to ``ready_limit``. A warm-up session opens at once and starts
    the heads-up of its run, up to ``heads_up_limit`` or until a trial waits for Ready. Only the notice shows it.
    """

    # How often a wait for Ready checks that the model server cancelled the preparation.
    CANCEL_POLL_SEC = 0.1
    # A camera is live while its last image is younger than this. Images come only while a trial runs, when the
    # teleoperator is in the headset, so the window spans the gap between trials for the page before Enter AR.
    LIVE_SEC = 30.0
    # A client opens a session again within its reconnect deadline when the connection drops before its first answer.
    RESUME_LIMIT_SEC = DEFAULT_RECONNECT_DEADLINE
    # The span over which `camera_feeds` counts the images of a camera.
    FEED_SEC = 2.0
    # The most recent images of one camera that `camera_feeds` counts.
    FEED_MAX_IMAGES = 512

    def __init__(
        self,
        served_address: ServedAddress,
        ready_limit: float,
        silence_limit: float,
        hold_sec: float,
        hold_threshold: float = 0.6,
        auth_token: str | None = None,
        cameras: Sequence[str] = (),
        controller_axes: geom.Rotation | None = CONTROLLER_AXES,
        heads_up_limit: float = 30 * 60.0,
    ):
        limits = (('ready_limit', ready_limit), ('silence_limit', silence_limit), ('heads_up_limit', heads_up_limit))
        for name, value in limits:
            check_finite_positive(value, name)
        self._binds = served_address
        self._auth_token = auth_token
        # The cameras of the station: a session that names no cameras gets these
        self.station_cameras = camera_names(cameras)
        # The axes of the controller in the frame of the headset, or None for the axes of the headset (see `Tracker`)
        self.controller_axes = controller_axes
        self.ready_limit = ready_limit
        # The longest time that the heads-up of a run shows, so the heads-up of a run that stopped goes.
        self.heads_up_limit = heads_up_limit
        self.silence_limit = silence_limit
        self._buttons = {
            RIGHT: ButtonMap(hold_sec, hold_threshold),
            LEFT: ButtonMap(hold_sec, hold_threshold, (Button.X,)),
        }
        self._changed = threading.Condition()
        # The log lines of the changes made under `_changed`, which `_locked` writes after it releases the lock
        self._log_lines: list[str] = []
        self._sample = _Sample(0, {}, {})
        # The `time.monotonic_ns()` instant of the last sample that tracks a controller, or None before the first
        self._tracked_at_ns: int | None = None
        self._phase = Phase.IDLE
        self._ready: ReadyAnswer | None = None
        # The events since the previous frame, each with the controller that made it
        self._events: list[tuple[str, Event]] = []
        self._task = ''
        # The clutch of each arm of the trial, from Ready on. A session that resumes the trial keeps them.
        self._trackers: dict[str | None, Tracker] = {}
        # The arms whose control went off because no sample came within the silence limit, until control comes on
        self._connection_lost: set[str | None] = set()
        # The last camera images of the run, by name, in the order that the client sends them.
        self._images: dict[str, np.ndarray] = {}
        # The `time.monotonic()` instant of the last image of each camera that the run sent. It holds over trials.
        self._seen: dict[str, float] = {}
        # The `time.monotonic()` instants of the last images of each camera, under `_feed_lock`.
        self._arrivals: dict[str, deque[float]] = {}
        self._feed_lock = threading.Lock()
        # The images of the shown camera that the service took. A new one changes the view.
        self._shown_image_count = 0
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
        # The `time.monotonic()` instant at which the heads-up of a run goes.
        self._heads_up_until: float | None = None
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
        self, poses: Mapping[str, geom.Transform3D | None], buttons: Mapping[str, ControllerButtons | None]
    ) -> None:
        """Take one sample of the controllers, by ``LEFT`` and ``RIGHT``. A controller that the headset does not track
        is absent or ``None``."""
        now_ns = time.monotonic_ns()
        # A controller with fewer button values than the maps read cannot press the clutch, so it drives no arm
        tracked = {
            name: state for name, state in buttons.items() if state is not None and len(state.values) > max(Button)
        }
        poses = {name: pose for name, pose in poses.items() if name in tracked}
        with self._changed:
            self._sample = _Sample(now_ns, poses, tracked)
            if tracked:
                self._tracked_at_ns = now_ns
            for controller, button_map in self._buttons.items():
                state = tracked.get(controller)
                if state is None:
                    button_map.cancel_holds()
                    continue
                if self._phase is Phase.READY_WAIT and self._ready is None:
                    self._ready = button_map.ready_answer(state, now_ns)
                elif self._phase is Phase.RUNNING:
                    self._events += [(controller, event) for event in button_map.events(state, now_ns)]
                else:
                    button_map.observe(state, now_ns)
            self._changed.notify_all()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the lock, then write the log lines of the changes after the release. Take it only where this thread
        does not hold the lock already, so that no line is written under the lock."""
        with self._changed:
            yield
            lines, self._log_lines = self._log_lines, []
        for line in lines:
            logger.info(line)

    @property
    def phase(self) -> Phase:
        with self._locked():
            self._forget_what_expired()
            return self._phase

    def _state(self) -> TrialState:
        """Hold the lock."""
        self._forget_what_expired()
        control = any(tracker.on for tracker in self._trackers.values())
        tracked = (
            self._tracked_at_ns is not None and time.monotonic_ns() - self._tracked_at_ns < self.silence_limit * 1e9
        )
        return TrialState(self._phase, self._task, control, tuple(self._images), tracked)

    def state(self) -> TrialState:
        with self._locked():
            return self._state()

    @staticmethod
    def _arm_name(arm: str | None) -> str:
        """The name of ``arm`` in the view and the log. The one arm of a one-arm session is 'Control'."""
        return 'Control' if arm is None else arm.capitalize()

    def _log(self, line: str) -> None:
        """Keep ``line`` for the log, until `_locked` releases the lock. Hold the lock."""
        self._log_lines.append(line)

    def _set_phase(self, phase: Phase, cause: str) -> None:
        """Go to ``phase``, and log a change with its ``cause``. Hold the lock."""
        if phase is not self._phase:
            self._log(f'Teleop trial: {self._phase} -> {phase} ({cause})')
        self._phase = phase

    def _clear_trial(self, cause: str) -> None:
        """Show no trial, and log that control of each arm whose control is on goes off for ``cause``. Hold the
        lock."""
        for arm in [arm for arm, tracker in self._trackers.items() if tracker.on]:
            self._log(f'{self._arm_name(arm)} off ({cause})')
        self._set_phase(Phase.IDLE, cause)
        self._events, self._connection_lost = [], set()
        self._task, self._trackers, self._images, self._trial_cameras = '', {}, {}, ()
        self._headset_camera, self._hold, self._frames, self._resumable_until = None, {}, 0, None
        self._end, self._heads_up_until = None, None

    def _forget_what_expired(self) -> None:
        """Show no trial once no session can resume it, and no heads-up once it expired. Hold the lock."""
        now = time.monotonic()
        if self._resumable_until is not None and now >= self._resumable_until:
            self._clear_trial('no session resumed the dropped trial')
        elif self._heads_up_until is not None and now >= self._heads_up_until:
            self._clear_trial('the heads-up expired')

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

    def take_images(self, images: Mapping[str, np.ndarray]) -> None:
        """Take camera images of the run, by camera name. Only a running trial takes them."""
        now = time.monotonic()
        # Atomic with the phase check: one entry of each of two dicts for each camera, and one count increment.
        with self._changed:
            if self._phase is not Phase.RUNNING:
                return
            self._images.update(images)
            self._seen.update(dict.fromkeys(images, now))
            if self._shown_camera() in images:
                self._shown_image_count += 1
        # One append for each camera, to a deque of at most `FEED_MAX_IMAGES`.
        with self._feed_lock:
            for name in images:
                self._arrivals.setdefault(name, deque(maxlen=self.FEED_MAX_IMAGES)).append(now)

    def shown_image_count(self) -> int:
        """The count of the images of the shown camera that the service took. It changes with each new one."""
        with self._changed:  # One int read, ordered with the increment in `take_images`.
            return self._shown_image_count

    def camera_feeds(self) -> dict[str, CameraFeed]:
        """The feed of each camera whose images reached the service in the last ``FEED_SEC``."""
        now = time.monotonic()
        since = now - self.FEED_SEC
        # Drops each camera with no image since `since`, then copies at most `FEED_MAX_IMAGES` instants per camera.
        with self._feed_lock:
            for name in [name for name, arrivals in self._arrivals.items() if arrivals[-1] <= since]:
                del self._arrivals[name]
            recent = {name: tuple(arrivals) for name, arrivals in self._arrivals.items()}
        return {
            name: CameraFeed(sum(arrived > since for arrived in arrivals) / self.FEED_SEC, now - arrivals[-1])
            for name, arrivals in recent.items()
        }

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

    @staticmethod
    def _badge(view: np.ndarray, origin: tuple[int, int], badge: Badge, scale: float) -> None:
        (width, height), baseline = cv2.getTextSize(badge.text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x, y = origin
        cv2.rectangle(view, (x - 6, y - height - 5), (x + width + 6, y + baseline), badge.colour, cv2.FILLED)
        cv2.putText(view, badge.text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)

    @staticmethod
    def _wrapped(line: str, scale: float, width: int) -> list[str]:
        """``line`` in at most three rows that each fit in ``width`` pixels at ``scale``. The rows break between words,
        and a word wider than ``width`` breaks between its characters. A line that needs more rows ends its third row
        with '...'."""
        most_rows = 3

        def fits(text: str) -> bool:
            return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)[0][0] <= width

        def pieces(word: str) -> Iterator[str]:
            if fits(word):
                yield word
                return
            piece = ''
            for char in word:
                if piece and not fits(piece + char):
                    yield piece
                    piece = ''
                piece += char
            yield piece

        rows = ['']
        for piece in (part for word in line.split(' ') for part in pieces(word)):
            joined = f'{rows[-1]} {piece}' if rows[-1] else piece
            if rows[-1] and not fits(joined):
                rows.append(piece)
            else:
                rows[-1] = joined
        if len(rows) <= most_rows:
            return rows
        last = rows[most_rows - 1]
        while last and not fits(last + '...'):
            last = last.rpartition(' ')[0] if ' ' in last else last[:-1]
        return [*rows[: most_rows - 1], last + '...']

    @classmethod
    def _view(
        cls,
        image: np.ndarray | None,
        lines: Sequence[str | Badge],
        footer: Sequence[str] = (),
        others: Sequence[tuple[str, np.ndarray | None]] = (),
    ) -> np.ndarray:
        """``image``, or a black image, with ``lines`` of text or badges at its top and the smaller ``footer`` at its
        bottom. Each line of text wraps to the width of the main camera.

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
        bottom, main_width = view.shape[0], width if others else view.shape[1]
        rows = [
            row
            for line in lines
            for row in ([line] if isinstance(line, Badge) else cls._wrapped(line, 0.75, main_width - 20))
        ]
        for number, row in enumerate(rows):
            if isinstance(row, Badge):
                cls._badge(view, (10, 26 + 30 * number), row, 0.75)
            else:
                cls._text(view, (10, 26 + 30 * number), row, 0.75)
        for row, line in enumerate(reversed(footer)):
            cls._text(view, (10, bottom - 10 - 22 * row), line, 0.55)
        return view

    @dataclass(frozen=True)
    class _ViewSnapshot:
        """What the headset view shows, read in one hold of the lock."""

        state: TrialState
        shown: str | None
        answered: bool
        # Each arm, by the arm name of its channels, and whether its control is on
        controls: tuple[tuple[str | None, bool], ...]
        # The arms whose control went off because no controller data came
        connection_lost: frozenset[str | None]
        names: tuple[str, ...]
        images: Mapping[str, np.ndarray]

    def view_text(self, task: bool = True) -> ViewText:
        """The text of the headset view in the state of the trial. A running trial shows its task only with ``task``."""
        with self._locked():  # It copies the state that the view shows, with one image reference for each camera.
            snapshot = self._view_snapshot()
        return self._text_of(snapshot, task)

    def view(self, camera: bool = True, task: bool = True) -> np.ndarray:
        """The image for the headset: the camera image of a running trial, or black with no ``camera``, with
        `view_text` on it. With more than one camera, a column on the right shows the other cameras."""
        with self._locked():  # It copies the state that the view shows, with one image reference for each camera.
            snapshot = self._view_snapshot()
        text = self._text_of(snapshot, task)
        shows_images = snapshot.state.phase is Phase.RUNNING and camera
        shown, names = snapshot.shown, snapshot.names
        image = snapshot.images.get(shown) if shows_images and shown is not None else None
        main = shown if shown in names else next(iter(names), None)
        others = [(name, snapshot.images.get(name) if shows_images else None) for name in names if name != main]
        return self._view(image, text.lines, text.footer, others)

    def _view_snapshot(self) -> _ViewSnapshot:
        """Hold the lock."""
        # The state first, because the end of an old dropped trial clears the arms
        state, shown, names = self._state(), self._shown_camera(), self.cameras().names
        controls = tuple((arm, tracker.on) for arm, tracker in self._trackers.items())
        # The task and the state of the arms come with the first frame request.
        answered = self._frames > 0
        connection_lost = frozenset(self._connection_lost)
        return self._ViewSnapshot(state, shown, answered, controls, connection_lost, names, dict(self._images))

    def _control_lines(
        self, controls: Sequence[tuple[str | None, bool]], connection_lost: frozenset[str | None]
    ) -> list[Badge]:
        """The state of the clutch of each arm and the button that changes it, on the colour of the state. The line of
        an arm in ``connection_lost`` names that cause."""
        lines = []
        for arm, on in controls:
            name, clutch = self._arm_name(arm), self.CLUTCH_BUTTONS[ARM_CONTROLLERS[arm]]
            if on:
                lines.append(Badge(f'{name} on ({clutch}: off)', self.CONTROL_ON_COLOUR))
            else:
                cause = ': connection lost' if arm in connection_lost else ''
                lines.append(Badge(f'{name} off{cause} ({clutch}: on)', self.CONTROL_OFF_COLOUR))
        return lines

    def _text_of(self, snapshot: _ViewSnapshot, task: bool) -> ViewText:
        footer = (CAMERA_HINT,)
        match snapshot.state.phase:
            case Phase.IDLE | Phase.HEADS_UP:
                return ViewText(('Waiting for a trial',), footer)
            case Phase.READY_WAIT:
                return ViewText(('Human trial', 'B: Ready', 'Stick: Cancel'), footer)
            case Phase.ENDED:
                return ViewText(('The trial ended', 'Give the verdict on the console'), footer)
        if snapshot.shown is not None:
            footer = (f'Camera: {snapshot.shown}', *footer)
        if not snapshot.answered:
            return ViewText(('Trial starting: waiting for the robot',), footer)
        lines = (
            *self._control_lines(snapshot.controls, snapshot.connection_lost),
            'Hold B: Complete. Hold the stick: Give up',
        )
        return ViewText((snapshot.state.task, *lines) if task else lines, footer)

    # The RGB colour behind the control line of an arm whose control is on, and off
    CONTROL_ON_COLOUR = (6, 214, 160)
    CONTROL_OFF_COLOUR = (255, 209, 102)
    # The button that turns control on and off, by controller
    CLUTCH_BUTTONS = {LEFT: 'X', RIGHT: 'A'}

    def _shown_camera(self) -> str | None:
        """The camera that the headset shows. Hold the lock."""
        for name in (self._chosen_camera, self._headset_camera):
            if name is not None and name in self._trial_cameras:
                return name
        return next(iter(self._trial_cameras), None)

    def show_heads_up(self) -> None:
        """Show the heads-up of a run that has a teleop episode, in place of the trial that the service kept."""
        with self._locked():  # Field writes only: no reader sees the heads-up with the fields of the old trial.
            self._clear_trial('a warm-up session starts')
            self._set_phase(Phase.HEADS_UP, 'a warm-up session starts')
            self._heads_up_until = time.monotonic() + self.heads_up_limit

    def resume_dropped_trial(self, settings: SessionParams) -> ResumeOutcome:
        """Resume the trial whose session dropped after its first frame request, as the next session of its client.

        The client opens such a session again within its reconnect deadline, and the teleoperator pressed Ready already.
        """
        with self._locked():
            self._forget_what_expired()
            if self._resumable_until is None:
                return ResumeOutcome.NO_DROPPED_TRIAL
            self._resumable_until, self._trial_cameras = None, settings.cameras
            self._headset_camera = settings.headset_camera
            self._log('A new session resumed the dropped trial')
            return ResumeOutcome.RESUMED

    def wait_for_ready(self, settings: SessionParams, cancelled: threading.Event) -> None:
        """Start a new trial of a session with ``settings``, and wait up to the ready limit for Ready.

        Raise ``TrialNotStarted`` for Cancel or for no Ready within the ready limit, and ``PreparationCancelled`` when
        ``cancelled`` is set or the service stops.
        """
        deadline = time.monotonic() + self.ready_limit
        with self._locked():
            self._clear_trial('a new trial starts')
            self._set_phase(Phase.READY_WAIT, 'a session waits for Ready')
            self._ready = None
            self._trial_cameras, self._headset_camera = settings.cameras, settings.headset_camera
            tracked_at_before = self._tracked_at_ns
        # A second hold, so that the start of the wait is in the log while the wait runs
        with self._locked():
            while (
                self._ready is None
                and not self._stopping
                and not cancelled.is_set()
                and (left := deadline - time.monotonic()) > 0
            ):
                self._changed.wait(min(left, self.CANCEL_POLL_SEC))
            ready, tracked_in_wait = self._ready, self._tracked_at_ns != tracked_at_before
            if ready is ReadyAnswer.READY:
                self._set_phase(Phase.RUNNING, 'B: Ready')
                self._trackers = {
                    arm: Tracker(settings.operator(), self.controller_axes) for arm in settings.arms.value
                }
                # In UMI mode control comes on with the trackers
                for arm in [arm for arm, tracker in self._trackers.items() if tracker.on]:
                    self._log(f'{self._arm_name(arm)} on (B: Ready, in UMI mode)')
            elif self._stopping or cancelled.is_set():
                self._clear_trial('the session closed before Ready')
            elif ready is ReadyAnswer.CANCELLED:
                self._clear_trial('Cancel')
            else:
                self._clear_trial('no Ready within the ready limit')
        if ready is ReadyAnswer.READY:
            return
        if self._stopping or cancelled.is_set():
            raise PreparationCancelled('the session closed before the teleoperator was ready')
        if ready is ReadyAnswer.CANCELLED:
            raise TrialNotStarted('the teleoperator cancelled the trial')
        not_ready = 'the teleoperator was not ready within the ready limit of the service'
        raise TrialNotStarted(not_ready if tracked_in_wait else f'{not_ready}, and no controller data came in the wait')

    def _newest_sample(self) -> _Sample | None:
        """The newest sample, or None when it is older than the silence limit. Hold the lock."""
        if time.monotonic_ns() - self._sample.taken_ns > self.silence_limit * 1e9:
            return None
        return self._sample

    def next_frame(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Take the state of the arms in ``request``, and answer with the element of the next frame: the target of the
        teleoperator, with the end of the trial once the teleoperator ends it."""
        with self._locked():  # One frame: the events, and the clutch and the target of at most two arms.
            if self._phase is Phase.ENDED:
                assert self._end is not None
                return self._end
            if self._phase is not Phase.RUNNING:
                raise ValueError(f'a frame request came in the {self._phase} phase')
            self._task = str(request[TASK])
            assert self._trackers, 'a running trial has the clutches of its Ready'
            robot = {arm: self._arm_state(request, arm) for arm in self._trackers}
            if not self._hold:
                self._hold = dict(robot)
            sample = self._newest_sample()
            events, self._events = self._events, []
            element = self._apply(sample, events, {arm: pose for arm, (pose, _) in robot.items()})
            self._frames += 1
            return element

    @staticmethod
    def _arm_state(request: Mapping[str, Any], arm: str | None) -> tuple[geom.Transform3D, float]:
        """The pose and the grip of ``arm`` in a frame request."""
        pose = np.asarray(request[keys.arm_channel(EE_POSE, arm)], dtype=float)
        grip = np.asarray(request[keys.arm_channel(GRIP, arm)], dtype=float).reshape(-1)[0]
        return geom.Transform3D.from_vector(pose, QUAT), float(grip)

    def _end_trial(self, end: PolicyEnd, cause: str) -> dict[str, Any]:
        """End the trial with ``end``, and return the element that ends it. Hold the lock."""
        self._end = {**self._hold_element(), END: end.value}
        self._set_phase(Phase.ENDED, cause)
        return self._end

    def _drive_arm(
        self, arm: str | None, sample: _Sample | None, events: Sequence[tuple[str, Event]], robot_pose: geom.Transform3D
    ) -> None:
        """Apply the clutch presses of the controller of ``arm`` in ``events``, and move the hold target of ``arm`` to
        the hand. With no ``sample`` within the silence limit, control goes off. Hold the lock."""
        tracker, controller = self._trackers[arm], ARM_CONTROLLERS[arm]
        name, clutch = self._arm_name(arm), self.CLUTCH_BUTTONS[controller]
        pose = None if sample is None else sample.poses.get(controller)
        buttons = None if sample is None else sample.buttons.get(controller)
        target = None if pose is None else tracker.update(pose)
        # In UMI mode control is always on, and the clutch does nothing
        umi = tracker.is_umi_mode
        for _ in [event for source, event in events if source == controller and event is Event.CLUTCH and not umi]:
            if tracker.on:
                tracker.turn_off()
                self._log(f'{name} off ({clutch})')
            elif pose is not None:
                tracker.turn_on(robot_pose, pose)
                target = tracker.update(pose)
                self._connection_lost.discard(arm)
                self._log(f'{name} on ({clutch})')
        if target is None and tracker.on and not umi:
            tracker.turn_off()
            if sample is None:
                self._connection_lost.add(arm)
                silence = f'no controller data within the silence limit of {self.silence_limit} s'
                self._log(f'{name} off: connection lost, {silence}')
            else:
                self._log(f'{name} off: the headset does not track the controller')
        if tracker.on and target is not None:
            self._hold[arm] = target, 0.0 if buttons is None else ButtonMap.grip(buttons.values)

    def _apply(
        self,
        sample: _Sample | None,
        events: Sequence[tuple[str, Event]],
        robot_poses: Mapping[str | None, geom.Transform3D],
    ) -> dict[str, Any]:
        """The frame element of ``sample`` and ``events``. Control of each arm starts from its pose in
        ``robot_poses``, so the robot does not jump. Hold the lock."""
        for _, event in events:
            match event:
                case Event.COMPLETE:
                    return self._end_trial(PolicyEnd.COMPLETE, 'hold B: Complete')
                case Event.GIVE_UP:
                    return self._end_trial(PolicyEnd.GIVE_UP, 'hold the stick: Give up')
        for arm in self._trackers:
            self._drive_arm(arm, sample, events, robot_poses[arm])
        return self._hold_element()

    def _hold_element(self) -> dict[str, Any]:
        """The element that sends the hold targets, arm after arm. Hold the lock."""
        assert self._hold, 'the first frame request sets the hold targets'
        return {ACTION: np.concatenate([np.append(pose.as_vector(QUAT), grip) for pose, grip in self._hold.values()])}

    def end_session(self) -> None:
        """End the session of the trial. A trial that answered one frame stays for a client that opens its session
        again, because the client cannot know that the one answer came. A heads-up stays."""
        # TODO: `Session.close` does not tell how the session ended, so a clean end after one frame also keeps the
        #  trial. A new session within `RESUME_LIMIT_SEC` then resumes it and skips Ready, so in UMI mode the robot
        #  follows the hand at once. We think this is improbable. Fix it when it occurs, or when we think that it is
        #  more probable.
        with self._locked():
            if self._phase is Phase.HEADS_UP:
                return
            if self._phase is Phase.RUNNING and self._frames == 1:
                self._resumable_until = time.monotonic() + self.RESUME_LIMIT_SEC
                self._log(f'The session ended. A new session can resume the trial for {self.RESUME_LIMIT_SEC} s')
                return
            self._clear_trial('the session ended')


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
        if self._decoders:
            images = {name: decoder.decode(observation[name]) for name, decoder in self._decoders.items()}
        else:
            images = {name: observation[name] for name in self._cameras}
        self._service.take_images(images)
        return [self._service.next_frame(observation)]

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
            if settings.warm_up:
                self._service.show_heads_up()
            elif self._service.resume_dropped_trial(settings) is ResumeOutcome.NO_DROPPED_TRIAL:
                self._service.wait_for_ready(settings, self.preparation_cancelled)
        except BaseException:
            self._open = False
            raise
        return TeleopSession(self._service, settings, self.client_stack(settings), self._release)

    def _release(self) -> None:
        self._open = False
