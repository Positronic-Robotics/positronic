"""The command ``positronic teleop``: the teleop service, the WebXR page of the headset, and the window."""

import logging
import math
import os
import subprocess
import sys
import webbrowser
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import configuronic as cfn
from platform_client.policy_container import AUTH_TOKEN_ENV
from positronic_model_server.server_wire import ServedAddress, ServedHostPort

import pimm
import positronic.cfg.webxr
from positronic.drivers.webxr import RIGHT, WebXR
from positronic.offboard.server import socket_at

from .protocol import DEFAULT_PORT, default_socket_path
from .service import Phase, TeleopService
from .window import Window, window_app

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageState:
    """The state of the trial that the phone page shows. Its field names are the fields of the JSON object."""

    phase: Phase
    control: bool


class _Headset(pimm.ControlSystem):
    """Pass the samples of the headset to the service, and the view of the service to the headset."""

    def __init__(self, service: TeleopService, view_fps: float = 30.0):
        if not 0 < view_fps < math.inf:
            raise ValueError(f'view_fps must be finite and positive, got {view_fps}')
        self._service = service
        self._view_period_sec = 1 / view_fps
        self.controller_positions = pimm.ControlSystemReceiver(self)
        self.buttons = pimm.ControlSystemReceiver(self)
        self.view = pimm.ControlSystemEmitter(self)
        # The state of the trial, as a `PageState` mapping.
        self.page_state = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        while not should_stop.value:
            positions = pimm.read_updated(self.controller_positions)
            if positions is not None:
                buttons = self.buttons.read()
                right = None if buttons is None else buttons.data[RIGHT]
                self._service.take_sample(positions.data[RIGHT], right)
            self.view.emit(self._service.view())
            state = self._service.state()
            self.page_state.emit(asdict(PageState(state.phase, state.control)))
            yield pimm.Sleep(self._view_period_sec)


class Notice(Protocol):
    """A notice to the teleoperator on the computer of the service."""

    def show(self, text: str) -> None: ...

    def hide(self) -> None: ...


class _ReadyNotice(pimm.ControlSystem):
    """Show ``text`` with ``notice`` while a trial waits for Ready."""

    def __init__(self, notice: Notice, text: str):
        self._notice = notice
        self._text = text
        # The state of the trial, as a `PageState` mapping.
        self.page_state = pimm.ControlSystemReceiver(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        waiting = False
        try:
            while not should_stop.value:
                state = pimm.read_updated(self.page_state)
                if state is not None:
                    was_waiting, waiting = waiting, PageState(**state.data).phase is Phase.READY_WAIT
                    if waiting and not was_waiting:
                        self._notice.show(self._text)
                    elif was_waiting and not waiting:
                        self._notice.hide()
                yield pimm.Sleep(0.1)
        finally:
            if waiting:
                self._notice.hide()


_MACOS_SOUND, _LINUX_SOUND = '/System/Library/Sounds/Glass.aiff', '/usr/share/sounds/freedesktop/stereo/complete.oga'


class DesktopNotice:
    """A dialog with ``sound`` in the centre of the screen of this computer: ``osascript`` and ``afplay`` on macOS,
    else ``zenity`` on ``DISPLAY`` and ``paplay``."""

    def __init__(
        self, title: str = 'positronic teleop', sound: str = _MACOS_SOUND if sys.platform == 'darwin' else _LINUX_SOUND
    ):
        self._title = title
        self._sound = Path(sound)
        self._dialog: subprocess.Popen | None = None

    @staticmethod
    def _applescript_string(text: str) -> str:
        return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'

    @staticmethod
    def _start(command: Sequence[str]) -> subprocess.Popen | None:
        try:
            return subprocess.Popen(command)
        except OSError as error:
            # The notice only helps the teleoperator, so the service goes on without it.
            logger.error('Cannot show the notice with %s: %s', command[0], error)
            return None

    def show(self, text: str) -> None:
        if sys.platform == 'darwin':
            self._start(['afplay', str(self._sound)])
            script = f'display alert {self._applescript_string(self._title)} message {self._applescript_string(text)}'
            self._dialog = self._start(['osascript', '-e', script])
        else:
            self._dialog = self._start(['zenity', '--info', '--no-markup', f'--title={self._title}', f'--text={text}'])
            self._start(['paplay', str(self._sound)])

    def hide(self) -> None:
        if self._dialog is not None:
            self._dialog.terminate()
            self._dialog = None


desktop_notice = cfn.Config(DesktopNotice)


# The teleop service on a host and a port, for a run on another machine.
network = cfn.Config(ServedHostPort, host='0.0.0.0', port=DEFAULT_PORT)


@cfn.config(
    # At the WebXR default quality of 50, the headset text and small objects are not clear
    webxr=positronic.cfg.webxr.oculus.override(jpeg_quality=90),
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
    press_threshold: float = 0.6,
    view_fps: float = 30.0,
    open_window: bool = True,
):
    """Run the teleop service until Ctrl-C. Open the WebXR page in the headset, and the window on the station screen.

    ``served_address`` is the Unix socket of this user by default.
    ``--served_address=@positronic.teleop.command.network`` serves on a host and a port, and needs the bearer token in
    ``AUTH_TOKEN_ENV``. A run sets the parameters of its session (``SessionParams``) in the query of its address.
    ``notice`` shows on this computer while a trial waits for Ready. ``--notice=None`` turns it off.
    """
    # TODO: Set the ready limit and the silence limit for a rig.
    # The socket is in a directory of this user only, so it takes no token. A token in the environment then does not
    # lock out `cfg.policy.teleop`, which sends none.
    auth_token = os.environ.get(AUTH_TOKEN_ENV) if isinstance(served_address, ServedHostPort) else None
    service = TeleopService(
        served_address, ready_limit, silence_limit, hold_sec, press_threshold, auth_token=auth_token
    )
    headset = _Headset(service, view_fps)
    text = f'A trial waits for Ready. Press Ready (B on a Quest) within {ready_limit:g} s.'
    ready_notice = None if notice is None else _ReadyNotice(notice, text)
    with service, Window(window_app(service), window_port) as window, pimm.World() as world:
        world.connect(webxr.controller_positions, headset.controller_positions)
        world.connect(webxr.buttons, headset.buttons)
        world.connect(headset.view, webxr.frame)
        world.connect(headset.page_state, webxr.page_state)
        if ready_notice is not None:
            world.connect(headset.page_state, ready_notice.page_state)
        logger.info('The teleop service serves on %s. Its window is %s', service.served_address, window.url)
        if open_window:
            webbrowser.open(window.url)
        world.run([headset, ready_notice], [webxr])
