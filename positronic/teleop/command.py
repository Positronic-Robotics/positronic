"""The command ``positronic teleop``: the teleop service, the WebXR page of the headset, and the window."""

import logging
import math
import os
import webbrowser
from collections.abc import Iterator
from dataclasses import asdict, dataclass

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
    attempt_started: bool


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
            self.page_state.emit(asdict(PageState(state.phase, state.control, state.attempt_started)))
            yield pimm.Sleep(self._view_period_sec)


# The teleop service on a host and a port, for a run on another machine.
network = cfn.Config(ServedHostPort, host='0.0.0.0', port=DEFAULT_PORT)


@cfn.config(
    webxr=positronic.cfg.webxr.oculus,
    served_address=socket_at.override(uds=str(default_socket_path())),
    ready_limit=120.0,
    window_port=5006,
)
def serve(
    webxr: WebXR,
    served_address: ServedAddress,
    ready_limit: float,
    window_port: int,
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
    """
    # TODO: Set the ready limit and the silence limit for a rig.
    # The socket is in a directory of this user only, so it takes no token. A token in the environment then does not
    # lock out `cfg.policy.teleop`, which sends none.
    auth_token = os.environ.get(AUTH_TOKEN_ENV) if isinstance(served_address, ServedHostPort) else None
    service = TeleopService(
        served_address, ready_limit, silence_limit, hold_sec, press_threshold, auth_token=auth_token
    )
    headset = _Headset(service, view_fps)
    with service, Window(window_app(service), window_port) as window, pimm.World() as world:
        world.connect(webxr.controller_positions, headset.controller_positions)
        world.connect(webxr.buttons, headset.buttons)
        world.connect(headset.view, webxr.frame)
        world.connect(headset.page_state, webxr.page_state)
        logger.info('The teleop service serves on %s. Its window is %s', service.served_address, window.url)
        if open_window:
            webbrowser.open(window.url)
        world.run(headset, [webxr])
