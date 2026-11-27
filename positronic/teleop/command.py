"""The command ``positronic teleop``: the teleop service, the WebXR page of the headset, and the window."""

import logging
import math
import os
import subprocess
import sys
import webbrowser
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import configuronic as cfn
from platform_client.policy_container import AUTH_TOKEN_ENV
from positronic_model_server.server_wire import ServedAddress, ServedHostPort

import pimm
import positronic.cfg.webxr
from positronic import geom
from positronic.drivers.webxr import WebXR
from positronic.offboard.server import socket_at
from positronic.utils import check_finite_positive

from .clutch import CONTROLLER_AXES
from .protocol import DEFAULT_PORT, default_socket_path
from .service import HEADS_UP_TEXT, Phase, TeleopService
from .window import Window, window_app

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageState:
    """The state of the trial that the phone page and the headset page show. Its field names are the fields of the
    JSON object. ``cameras``, ``live``, ``shown`` and ``chosen`` are a ``CameraState``."""

    phase: Phase
    control: bool
    cameras: tuple[str, ...] = ()
    live: tuple[str, ...] = ()
    shown: str | None = None
    chosen: str | None = None
    # A sample that tracks a controller came within the silence limit of the service
    tracked: bool = False


# The field of a page input that chooses the camera of the headset: a camera name, or null for the camera of the
# session.
HEADSET_CAMERA = 'headset_camera'
# The field of a page input that asks for the text of the view with no camera image, so that a page that shows black
# as clear shows only the text: true or false.
TEXT_ONLY = 'headset_text_only'
# The field of a page input that asks for the view of a running trial with no task: true or false.
TASK_HIDDEN = 'headset_task_hidden'


class _Headset(pimm.ControlSystem):
    """Pass the samples of the headset to the service, and the view of the service to the headset.

    The bridge sends the view on a new image of the shown camera or a change that the page asks for, and else once each
    `REFRESH_PERIODS` view periods. It sends at most one view each view period, so a new image that comes sooner waits
    for the next one, and the view then shows the newest image.
    """

    # How often the bridge logs the camera feeds that reach the service.
    FEED_LOG_SEC = 10.0
    # How often the bridge reads the headset and looks for a new image of the shown camera. It also wakes at each slot.
    POLL_SEC = 0.005
    # How many view periods the bridge waits with no new image or page change before it sends the view again.
    REFRESH_PERIODS = 2

    def __init__(self, service: TeleopService, view_fps: float = 30.0):
        check_finite_positive(view_fps, 'view_fps')
        self._service = service
        self._view_period_sec = 1 / view_fps
        self.controller_positions = pimm.ControlSystemReceiver(self)
        self.buttons = pimm.ControlSystemReceiver(self)
        # What the headset page asks for, as a JSON object
        self.page_input = pimm.ControlSystemReceiver(self)
        self._text_only = False
        self._task_hidden = False
        self.view = pimm.ControlSystemEmitter(self)
        # The state of the trial, as a `PageState` mapping.
        self.page_state = pimm.ControlSystemEmitter(self)

    def _log_feeds(self) -> None:
        feeds = self._service.camera_feeds()
        if feeds:
            parts = [f'{name} at {f.rate_hz:.1f} Hz, the last {f.age_sec:.2f} s ago' for name, f in feeds.items()]
            logger.info('Camera images: %s', '; '.join(parts))

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        next_log = clock.now() + self.FEED_LOG_SEC
        last_view, sent_count, page_changed = -math.inf, None, False
        while not should_stop.value:
            if clock.now() >= next_log:
                next_log = clock.now() + self.FEED_LOG_SEC
                self._log_feeds()
            positions = pimm.read_updated(self.controller_positions)
            if positions is not None:
                buttons = self.buttons.read()
                self._service.take_sample(positions.data, {} if buttons is None else buttons.data)
            page_input = pimm.read_updated(self.page_input)
            if page_input is not None and HEADSET_CAMERA in page_input.data:
                camera = page_input.data[HEADSET_CAMERA]
                if camera is not None and not isinstance(camera, str):
                    logger.warning('The headset page chose no camera: %r is not a camera name', camera)
                else:
                    try:
                        self._service.choose_camera(camera)
                    except ValueError as refused:
                        logger.warning('The headset page chose no camera: %s', refused)
            if page_input is not None and TEXT_ONLY in page_input.data:
                self._text_only = page_input.data[TEXT_ONLY] is True
            if page_input is not None and TASK_HIDDEN in page_input.data:
                self._task_hidden = page_input.data[TASK_HIDDEN] is True
            page_changed = page_changed or page_input is not None
            image_count = self._service.shown_image_count()
            now = clock.now()
            changed = image_count != sent_count or page_changed
            if now >= last_view + self._view_period_sec and (
                changed or now >= last_view + self.REFRESH_PERIODS * self._view_period_sec
            ):
                last_view, sent_count, page_changed = now, image_count, False
                self._send_view()
            next_slot = last_view + self._view_period_sec
            yield pimm.Sleep(min(self.POLL_SEC, next_slot - now) if next_slot > now else self.POLL_SEC)

    def _send_view(self) -> None:
        self.view.emit(self._service.view(camera=not self._text_only, task=not self._task_hidden))
        state, cameras = self._service.state(), self._service.cameras()
        page_state = PageState(
            state.phase, state.control, cameras.names, cameras.live, cameras.shown, cameras.chosen, state.tracked
        )
        self.page_state.emit(asdict(page_state))


@dataclass(frozen=True)
class NoticeText:
    """The title and the message of a notice."""

    title: str
    message: str


class Notice(Protocol):
    """A notice to the teleoperator on the computer of the service."""

    def show(self, text: NoticeText) -> None: ...

    def hide(self) -> None: ...


class _PhaseNotice(pimm.ControlSystem):
    """Show the text of each phase in ``texts`` with ``notice`` while the trial is in that phase."""

    def __init__(self, notice: Notice, texts: Mapping[Phase, NoticeText]):
        self._notice = notice
        self._texts = dict(texts)
        # The state of the trial, as a `PageState` mapping.
        self.page_state = pimm.ControlSystemReceiver(self)

    def _switch(self, shown: NoticeText | None, text: NoticeText | None) -> None:
        """Hide the notice of ``shown``, and show ``text``. ``None`` is no notice."""
        if shown is not None:
            self._notice.hide()
        if text is not None:
            self._notice.show(text)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        shown: NoticeText | None = None
        try:
            while not should_stop.value:
                state = pimm.read_updated(self.page_state)
                if state is not None and (text := self._texts.get(PageState(**state.data).phase)) != shown:
                    self._switch(shown, text)
                    shown = text
                yield pimm.Sleep(0.1)
        finally:
            self._switch(shown, None)


_MACOS_SOUND, _LINUX_SOUND = '/System/Library/Sounds/Glass.aiff', '/usr/share/sounds/freedesktop/stereo/complete.oga'


class DesktopNotice:
    """A dialog with ``sound`` on the screen of this computer: ``osascript`` and ``afplay`` on macOS, else ``zenity`` on
    ``DISPLAY`` and ``paplay``."""

    def __init__(self, sound: str = _MACOS_SOUND if sys.platform == 'darwin' else _LINUX_SOUND):
        self._sound = Path(sound)
        self._dialog: subprocess.Popen | None = None

    @staticmethod
    def _applescript_string(text: str) -> str:
        return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'

    @staticmethod
    def _start(command: Sequence[str], env: Mapping[str, str] | None = None) -> subprocess.Popen | None:
        try:
            return subprocess.Popen(command, env=env)
        except OSError as error:
            # The notice only helps the teleoperator, so the service goes on without it.
            logger.error('Cannot show the notice with %s: %s', command[0], error)
            return None

    def show(self, text: NoticeText) -> None:
        if sys.platform == 'darwin':
            self._start(['afplay', str(self._sound)])
            title, message = self._applescript_string(text.title), self._applescript_string(text.message)
            self._dialog = self._start(['osascript', '-e', f'display alert {title} message {message}'])
        else:
            dialog = ['zenity', '--info', '--no-markup', '--width=700']
            dialog += [f'--title={text.title}', f'--text={text.message}']
            # GTK waits some seconds for the session bus and the accessibility bus of the screen. A user other than the
            # user at the screen cannot reach them, and the dialog needs neither.
            self._dialog = self._start(
                dialog, {**os.environ, 'DBUS_SESSION_BUS_ADDRESS': 'disabled:', 'NO_AT_BRIDGE': '1'}
            )
            self._start(['paplay', str(self._sound)])

    def hide(self) -> None:
        if self._dialog is not None:
            self._dialog.terminate()
            self._dialog = None


desktop_notice = cfn.Config(DesktopNotice)


# The teleop service on a host and a port, for a run on another machine.
network = cfn.Config(ServedHostPort, host='0.0.0.0', port=DEFAULT_PORT)


@cfn.config(
    webxr=positronic.cfg.webxr.oculus,
    served_address=socket_at.override(uds=str(default_socket_path())),
    ready_limit=120.0,
    window_port=5006,
    notice=desktop_notice,
)
def serve(
    webxr: WebXR,
    served_address: ServedAddress,
    ready_limit: float,
    window_port: int,
    notice: Notice | None,
    silence_limit: float = 0.5,
    hold_sec: float = 1.0,
    hold_threshold: float = 0.6,
    open_window: bool = True,
    cameras: Sequence[str] = (),
    controller_axes: geom.Rotation | None = CONTROLLER_AXES,
):
    """Run the teleop service until Ctrl-C. Open the WebXR page in the headset, and the window on the station screen.

    ``served_address`` is the Unix socket of this user by default.
    ``--served_address=@positronic.teleop.command.network`` serves on a host and a port. It checks the bearer token in
    ``AUTH_TOKEN_ENV``, and serves open when that is unset. A run sets the parameters of its session (``SessionParams``)
    in the query of its address.
    ``notice`` shows on this computer while a trial waits for Ready, and in the heads-up of a run that has a teleop
    episode. ``--notice=None`` turns it off.
    ``cameras`` names the cameras of the station, such as ``'["image.exterior", "image.wrist"]'``. A run whose address
    names no cameras sends these, and the headset page offers them.
    ``controller_axes`` turns the axes of the moves of the teleoperator by the heading of the controller, so the way the
    teleoperator faces does not count; it puts the axes of the controller in the frame of the headset.
    ``--controller_axes=None`` counts the moves in the axes of the headset.
    ``webxr.stale_limit`` must not exceed ``silence_limit``: a message that comes later than the silence limit finds
    control off, so its presses must do nothing.
    """
    if webxr.stale_limit > silence_limit:
        raise ValueError(
            f'the stale_limit of the headset must not exceed silence_limit, got {webxr.stale_limit:g} s and '
            f'{silence_limit:g} s'
        )
    # TODO: Set the ready limit and the silence limit for a rig.
    # The socket is in a directory of this user only, so it takes no token. A token in the environment then does not
    # lock out `cfg.policy.teleop`, which sends none.
    auth_token = None
    if isinstance(served_address, ServedHostPort):
        auth_token = os.environ.get(AUTH_TOKEN_ENV)
        if auth_token is None:
            logger.warning(
                '%s is unset: each peer that reaches %s:%d can open a teleop session',
                AUTH_TOKEN_ENV,
                served_address.host,
                served_address.port,
            )
    service = TeleopService(
        served_address,
        ready_limit,
        silence_limit,
        hold_sec,
        hold_threshold,
        auth_token=auth_token,
        cameras=cameras,
        controller_axes=controller_axes,
    )
    headset = _Headset(service, webxr.video_fps)
    ready_text = (
        f'A teleoperation trial waits for Ready. Within {ready_limit:g} s, press B on the Quest, '
        'or tap Ready on the phone page.'
    )
    texts = {
        Phase.HEADS_UP: NoticeText('Teleoperation run', HEADS_UP_TEXT),
        Phase.READY_WAIT: NoticeText('Teleoperation trial', ready_text),
    }
    phase_notice = None if notice is None else _PhaseNotice(notice, texts)
    with service, Window(window_app(service), window_port) as window, pimm.World() as world:
        world.connect(webxr.controller_positions, headset.controller_positions)
        world.connect(webxr.buttons, headset.buttons)
        world.connect(webxr.page_input, headset.page_input)
        world.connect(headset.view, webxr.frame)
        world.connect(headset.page_state, webxr.page_state)
        if phase_notice is not None:
            world.connect(headset.page_state, phase_notice.page_state)
        logger.info('The teleop service serves on %s. Its window is %s', service.served_address, window.url)
        if open_window:
            webbrowser.open(window.url)
        world.run([headset, phase_notice], [webxr])
