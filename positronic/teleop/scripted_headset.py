"""A scripted headset: it does the trials of ``positronic teleop`` with no person, for a timing run.

It connects to the page of the service as the headset does. It sends a sample of the right controller at the rate of a
headset, and reads the phase of the trial from the page state. In each trial it presses Ready, turns control on, moves
the hand on a circle, and holds B for Complete. Start it beside the service and a run with ``--timing=True``:

    uv run --locked python -m positronic.teleop.scripted_headset --page=https://127.0.0.1:5005 --trials=3
"""

import json
import logging
import math
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from urllib.parse import urlsplit

import configuronic as cfn
import httpx
import numpy as np
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from pimm.logging import init_logging
from positronic import geom
from positronic.drivers.webxr import CONTROLLER_PATH, RIGHT, STATE_PATH, controller_message
from positronic.utils import check_finite_nonnegative, check_finite_positive, check_positive_int

from .command import PageState
from .service import Button, Phase

logger = logging.getLogger(__name__)

_STATE_POLL_SEC = 0.05
# A press of Ready that the page does not show within this time is released and pressed again.
_READY_PRESS_AGAIN_AFTER_SEC = 1.0


def _buttons(*down: Button) -> tuple[float, ...]:
    return tuple(1.0 if button in down else 0.0 for button in range(max(Button) + 1))


class _Hand:
    """The right controller. A thread sends its pose and its buttons to the page at ``rate_hz``."""

    # The pose before control comes on, in the reference space of the page.
    _REST = geom.Transform3D(np.array([0.0, 1.0, -0.3]), geom.Rotation.identity)

    def __init__(
        self,
        websocket: Connection,
        rate_hz: float,
        view_fps: float,
        min_hold_sec: float,
        radius: float,
        circle_sec: float,
    ):
        self._websocket = websocket
        self._period_sec = 1 / rate_hz
        # The service reads the latest sample once in each period of its view rate. A hold of two periods of the slower
        # rate puts at least one sample of each state of the buttons in a read of the service.
        self._hold_sec = max(min_hold_sec, 2 * self._period_sec, 2 / view_fps)
        self._radius = radius
        self._circle_sec = circle_sec
        self.buttons = _buttons()
        # The `time.monotonic()` instant at which the hand started its circle, or `None` while it rests.
        self.moving_since: float | None = None
        # The instant at which the hand stopped on its circle, or `None` while it moves or rests.
        self.stopped_at: float | None = None
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._send, name='scripted-headset', daemon=True)

    def __enter__(self) -> '_Hand':
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _pose(self, now: float) -> geom.Transform3D:
        if self.moving_since is None:
            return self._REST
        angle = 2 * math.pi * ((self.stopped_at or now) - self.moving_since) / self._circle_sec
        offset = self._radius * np.array([math.cos(angle) - 1, 0.0, math.sin(angle)])
        return geom.Transform3D(self._REST.translation + offset, self._REST.rotation)

    def _send(self) -> None:
        next_send = time.monotonic()
        try:
            while not self._stop.is_set():
                message = controller_message({RIGHT: self._pose(time.monotonic())}, {RIGHT: self.buttons})
                self._websocket.send(json.dumps(message))
                next_send += self._period_sec
                now = time.monotonic()
                if next_send < now:
                    # A late send skips the slots that it missed, so the thread sends no burst to catch up.
                    next_send += math.ceil((now - next_send) / self._period_sec) * self._period_sec
                self._stop.wait(next_send - now)
        except Exception as error:
            self._error = error

    def check(self) -> None:
        """Raise when the thread stopped sending."""
        if self._error is not None:
            raise RuntimeError('The scripted headset stopped sending samples') from self._error

    @property
    def hold_sec(self) -> float:
        """The shortest release of a button after a press."""
        return self._hold_sec


class _Page:
    """The page state of the service, which the scripted headset polls."""

    def __init__(self, client: httpx.Client, hand: _Hand):
        self._client = client
        self._hand = hand

    def state(self) -> PageState:
        self._hand.check()
        fields = self._client.get(STATE_PATH).raise_for_status().json()
        # The page state is empty until the service sends its first one.
        if not fields:
            return PageState(Phase.IDLE, control=False)
        state = PageState(**fields)
        return replace(state, phase=Phase(state.phase))

    def phase(self) -> Phase:
        return self.state().phase

    def until(self, condition: Callable[[PageState], bool], within: float = math.inf) -> PageState | None:
        """The first page state that passes ``condition``, or ``None`` when none does within ``within`` seconds."""
        deadline = time.monotonic() + within
        while not condition(state := self.state()):
            if time.monotonic() >= deadline:
                return None
            time.sleep(_STATE_POLL_SEC)
        return state


def _press_until(
    page: _Page, hand: _Hand, button: Button, seen: Callable[[PageState], bool], again_after: float = math.inf
) -> PageState:
    """Hold ``button`` until the page state passes ``seen``, then release it for ``hand.hold_sec``. When the page does
    not pass ``seen`` within ``again_after``, release the button and press it again.

    The press lasts until the page shows its effect: a loaded service can read no sample in a hold of fixed length. The
    release makes the next press of the button a new press, and a press again recovers a release the service missed.
    """
    while True:
        logger.info('The scripted headset presses %s', button.name)
        hand.buttons = _buttons(button)
        state = page.until(seen, within=again_after)
        hand.buttons = _buttons()
        if state is not None:
            time.sleep(hand.hold_sec)
            return state
        # The page polls while the button is up, so a late effect of the press still ends the wait.
        if (state := page.until(seen, within=hand.hold_sec)) is not None:
            return state


def _trial(page: _Page, hand: _Hand, trial_sec: float) -> None:
    page.until(lambda state: state.phase is Phase.READY_WAIT)
    # A trial that ends at once after Ready, for example by a short timeout, can end before this poll sees it run.
    ready = _press_until(
        page, hand, Button.B, lambda state: state.phase is not Phase.READY_WAIT, _READY_PRESS_AGAIN_AFTER_SEC
    )
    if ready.phase is not Phase.RUNNING:
        logger.warning('The trial ended before the scripted headset saw it run')
        return
    if not _press_until(page, hand, Button.A, lambda state: state.control or state.phase is not Phase.RUNNING).control:
        logger.warning('The trial ended before the scripted headset turned control on')
    hand.moving_since = time.monotonic()
    while page.phase() is Phase.RUNNING and time.monotonic() - hand.moving_since < trial_sec:
        time.sleep(_STATE_POLL_SEC)
    # A trial that ended already, for example at its timeout, must not take this hold as the Ready of the next trial.
    if page.phase() is Phase.RUNNING:
        hand.stopped_at = time.monotonic()
        _press_until(page, hand, Button.B, lambda state: state.phase is not Phase.RUNNING)
    hand.moving_since = hand.stopped_at = None


_WEBSOCKET_SCHEMES = {'http': 'ws', 'https': 'wss'}


@cfn.config(rate_hz=72.0, view_fps=30.0, min_hold_sec=0.2, trial_sec=20.0, radius=0.05, circle_sec=4.0)
def drive(
    page: str,
    trials: int,
    rate_hz: float,
    view_fps: float,
    min_hold_sec: float,
    trial_sec: float,
    radius: float,
    circle_sec: float,
) -> None:
    """Do ``trials`` trials on the page of ``positronic teleop`` at the URL ``page``. Each trial moves the hand for
    ``trial_sec`` on a circle of ``radius``, once each ``circle_sec``. ``view_fps`` is the ``--view_fps`` of the
    service. ``min_hold_sec`` is the shortest release of a button after a press; a loaded machine needs a longer one.

    The page of the service has a self-signed certificate, so the scripted headset does not check it.
    """
    check_positive_int(trials, 'trials')
    for name, value in (
        ('rate_hz', rate_hz),
        ('view_fps', view_fps),
        ('min_hold_sec', min_hold_sec),
        ('trial_sec', trial_sec),
        ('circle_sec', circle_sec),
    ):
        check_finite_positive(value, name)
    check_finite_nonnegative(radius, 'radius')
    url = urlsplit(page)
    tls = ssl.create_default_context()
    tls.check_hostname = False
    tls.verify_mode = ssl.CERT_NONE
    controller_url = url._replace(scheme=_WEBSOCKET_SCHEMES[url.scheme], path=CONTROLLER_PATH).geturl()
    with (
        # `connect` refuses a TLS context for a ws URL.
        connect(controller_url, ssl=tls if url.scheme == 'https' else None) as websocket,
        httpx.Client(base_url=page, verify=tls) as client,
        _Hand(websocket, rate_hz, view_fps, min_hold_sec, radius, circle_sec) as hand,
    ):
        for trial in range(trials):
            _trial(_Page(client, hand), hand, trial_sec)
            logger.info('The scripted headset did trial %d of %d', trial + 1, trials)


if __name__ == '__main__':
    init_logging()
    cfn.cli(drive)
