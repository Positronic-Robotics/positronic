"""The command ``positronic teleop``: the teleop service, the WebXR page of the headset, and the window."""

import logging
import webbrowser
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import configuronic as cfn

import pimm
import positronic.cfg.webxr
from positronic.drivers.webxr import RIGHT, WebXR

from .clutch import OperatorPosition
from .protocol import SessionParams, default_socket_path
from .service import Phase, TeleopService
from .window import Window, window_app

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageState:
    """The state of the trial that the phone page shows. Its field names are the fields of the JSON object."""

    phase: Phase
    control: bool
    attempt_started: bool


class _Headset(pimm.ControlSystem):
    """Pass the samples of the headset to the service, and the view of the service to the headset."""

    def __init__(self, service: TeleopService, view_fps: float = 30.0):
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
            self.page_state.emit(asdict(PageState(state.phase, state.control, state.attempt_started)))
            yield pimm.Sleep(self._view_period_sec)


@cfn.config(webxr=positronic.cfg.webxr.oculus, socket=str(default_socket_path()), ready_limit=120.0, window_port=5006)
def serve(
    webxr: WebXR,
    socket: str,
    ready_limit: float,
    window_port: int,
    silence_limit: float = 0.5,
    hold_sec: float = 1.0,
    press_threshold: float = 0.6,
    view_fps: float = 30.0,
    headset_camera: str | None = None,
    open_window: bool = True,
    operator_position: OperatorPosition | None = SessionParams.operator_position,
    cameras: list[str] | None = SessionParams.cameras,
    image_fps: float = SessionParams.image_fps,
    still_clock_timeout: float = SessionParams.still_clock_timeout,
):
    """Run the teleop service until Ctrl-C. Open the WebXR page in the headset, and the window on the station screen.

    ``operator_position``, ``cameras``, ``image_fps`` and ``still_clock_timeout`` are the defaults of the session
    parameters (``SessionParams``). A run can set others in the query of its address.
    """
    # TODO: Set the ready limit and the silence limit for a rig.
    defaults = SessionParams(operator_position, cameras, image_fps, still_clock_timeout)
    service = TeleopService(
        Path(socket), ready_limit, silence_limit, hold_sec, press_threshold, headset_camera, session_defaults=defaults
    )
    headset = _Headset(service, view_fps)
    with service, Window(window_app(service), window_port) as window, pimm.World() as world:
        world.connect(webxr.controller_positions, headset.controller_positions)
        world.connect(webxr.buttons, headset.buttons)
        world.connect(headset.view, webxr.frame)
        world.connect(headset.page_state, webxr.page_state)
        logger.info('The teleop service serves on %s. Its window is %s', socket, window.url)
        if open_window:
            webbrowser.open(window.url)
        world.run(headset, [webxr])
