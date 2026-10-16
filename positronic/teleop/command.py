"""The command ``positronic teleop``: the teleop service, the WebXR page of the headset, and the window."""

import logging
import webbrowser
from collections.abc import Iterator
from pathlib import Path

import configuronic as cfn

import pimm
import positronic.cfg.webxr
from positronic.drivers.webxr import RIGHT, WebXR

from .service import TeleopService
from .session import default_socket_path
from .window import Window, window_app

logger = logging.getLogger(__name__)


class _Headset(pimm.ControlSystem):
    """Pass the samples of the headset to the service, and the view of the service to the headset."""

    VIEW_PERIOD_SEC = 1 / 30

    def __init__(self, service: TeleopService):
        self._service = service
        self.controller_positions = pimm.ControlSystemReceiver(self)
        self.buttons = pimm.ControlSystemReceiver(self)
        self.view = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        while not should_stop.value:
            positions = pimm.read_updated(self.controller_positions)
            if positions is not None:
                buttons = self.buttons.read()
                right = None if buttons is None else buttons.data[RIGHT]
                self._service.take_sample(positions.data[RIGHT], right)
            self.view.emit(self._service.view())
            yield pimm.Sleep(self.VIEW_PERIOD_SEC)


@cfn.config(webxr=positronic.cfg.webxr.oculus, socket=str(default_socket_path()), ready_limit=120.0, window_port=5006)
def serve(
    webxr: WebXR,
    socket: str,
    ready_limit: float,
    window_port: int,
    silence_limit: float = 0.5,
    hold_sec: float = 1.0,
    open_window: bool = True,
):
    """Run the teleop service until Ctrl-C. Open the WebXR page in the headset, and the window on the station screen."""
    # TODO: Set the ready limit and the silence limit for a rig.
    service = TeleopService(Path(socket), ready_limit, silence_limit, hold_sec)
    headset = _Headset(service)
    with service, Window(window_app(service), window_port) as window, pimm.World() as world:
        world.connect(webxr.controller_positions, headset.controller_positions)
        world.connect(webxr.buttons, headset.buttons)
        world.connect(headset.view, webxr.frame)
        logger.info('The teleop service serves on %s. Its window is %s', socket, window.url)
        if open_window:
            webbrowser.open(window.url)
        world.run(headset, [webxr])
