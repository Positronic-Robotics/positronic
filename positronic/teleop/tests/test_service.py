import json
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import types
from collections.abc import Callable, Generator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx
import numpy as np
import pos3
import pytest
from positronic_model_server import keys as server_keys
from positronic_model_server.protocol import (
    AUTH_HEADER,
    META,
    OBSERVATION,
    PROTOCOL_VERSION,
    RESULT,
    SESSION_ID,
    STATUS,
    ProtocolVersion,
    ServerStatus,
    bearer,
)
from positronic_model_server.serialization import deserialise, encode_jpeg, serialise
from positronic_model_server.server import ModelServer
from positronic_model_server.server_wire import ServedHostPort
from positronic_model_server.websocket_wire import ServedUnixSocket
from positronic_wire import registry, wire
from positronic_wire.websocket import WebsocketUnixClientWire
from positronic_wire.wire import SESSION_PATH, HostPortAddress, UnixSocketAddress
from websockets.http11 import Request, Response
from websockets.sync.client import unix_connect
from websockets.sync.server import ServerConnection, serve

import pimm
import positronic.cfg.policy
import positronic.cfg.simulator
from pimm.time import EMITTED_WORLD
from pimm.world import SystemClock
from positronic import geom, keys, telemetry, telemetry_keys
from positronic.cfg.eval.sim.positronic import stack_cubes
from positronic.cli.eval.run import main
from positronic.dataset.local_dataset import LocalDataset
from positronic.drivers import webxr
from positronic.drivers.roboarm.ik import ee_frames
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame
from positronic.drivers.webxr import INPUT_PATH, STATE_PATH, ControllerButtons, WebXR
from positronic.eval import InferenceCharge
from positronic.eval import keys as eval_keys
from positronic.offboard.client import InferenceClient
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd, Step
from positronic.policy.codecs import EncodeImages, EncodeVideo, EndCommand
from positronic.policy.codecs.metadata import ACTION
from positronic.policy.executor import Executor
from positronic.policy.keys import END
from positronic.policy.processors import ChunkedSchedule, EndOnField, InferenceSchedule, RequestTiming, TrajectoryPlayer
from positronic.policy.remote import RemotePolicy, declared_stack
from positronic.policy.sequential import Sequential
from positronic.teleop import command, protocol, scripted_headset
from positronic.teleop import service as service_module
from positronic.teleop.clutch import OperatorPosition
from positronic.teleop.protocol import VIEW_SIZE, session_query
from positronic.teleop.service import (
    OTHERS_WIDTH,
    Badge,
    Button,
    ButtonMap,
    CameraState,
    Event,
    Phase,
    ReadyAnswer,
    TeleopModel,
    TeleopService,
    TrialState,
)
from positronic.tests.testing_coutils import ManualCommandReceiver, MutableShouldStop, RecordingEmitter
from positronic.tests.testing_inference import free_port
from positronic.utils import package_assets_path

HAND = geom.Transform3D(np.array([0.1, 0.2, 1.0]), geom.Rotation.identity)
LEFT_HAND = geom.Transform3D(np.array([-0.3, 0.2, 1.0]), geom.Rotation.identity)
ROBOT_POSE = geom.Transform3D(np.array([0.4, 0.0, 0.3]), geom.Rotation.from_quat([0.0, 1.0, 0.0, 0.0]))
OBS = {
    keys.EE_POSE: ROBOT_POSE.as_vector(geom.Rotation.Representation.QUAT),
    keys.GRIP: 0.0,
    keys.TASK: 'pick up the cube',
    keys.EXTERIOR_IMAGE: np.full((48, 64, 3), 128, dtype=np.uint8),
}
FRAME_REQUEST = {protocol.TASK: OBS[keys.TASK], protocol.EE_POSE: OBS[keys.EE_POSE], protocol.GRIP: [OBS[keys.GRIP]]}
ARM_POSES = {
    keys.LEFT_ARM: geom.Transform3D(np.array([0.3, 0.3, 0.4]), geom.Rotation.from_quat([0.0, 1.0, 0.0, 0.0])),
    keys.RIGHT_ARM: geom.Transform3D(np.array([0.3, -0.3, 0.4]), geom.Rotation.from_quat([0.0, 1.0, 0.0, 0.0])),
}
ARM_GRIPS = {keys.LEFT_ARM: 0.0, keys.RIGHT_ARM: 0.5}
BIMANUAL_OBS = {
    keys.TASK: OBS[keys.TASK],
    **{
        f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.EE_POSE_SUFFIX}': pose.as_vector(
            geom.Rotation.Representation.QUAT
        )
        for arm, pose in ARM_POSES.items()
    },
    **{keys.arm_channel(keys.GRIP, arm): grip for arm, grip in ARM_GRIPS.items()},
}
ARM_COMMANDS = {arm: keys.arm_channel(keys.ROBOT_COMMAND, arm) for arm in keys.BIMANUAL_ARMS}
ARM_TARGET_GRIPS = {arm: keys.arm_channel(keys.TARGET_GRIP, arm) for arm in keys.BIMANUAL_ARMS}
HOLD_SEC = 0.2


def _buttons(trigger=0.0, stick=0.0, a=0.0, b=0.0, presses: dict[Button, int] | None = None) -> ControllerButtons:
    """The buttons that the headset page sends for a controller. With no ``presses``, each button that is down has a
    press count of 1 and the others of 0. So a sample with a button down after a sample with it up is one press."""
    values = [0.0] * (max(Button) + 1)
    for button, value in ((Button.TRIGGER, trigger), (Button.STICK, stick), (Button.A, a), (Button.B, b)):
        values[button] = value
    counts = [int(value >= 0.5) for value in values] if presses is None else [0] * len(values)
    for button, count in (presses or {}).items():
        counts[button] = count
    return ControllerButtons(tuple(values), tuple(counts))


def _map(*args, **kwargs) -> ButtonMap:
    """A ``ButtonMap`` that took one sample with no button down, as a page sends before the first press."""
    button_map = ButtonMap(*args, **kwargs)
    button_map.observe(_buttons(), 0)
    return button_map


class TestButtonMap:
    def test_b_and_the_stick_answer_the_wait_for_ready(self):
        assert _map(HOLD_SEC).ready_answer(_buttons(b=1.0), 0) is ReadyAnswer.READY
        assert _map(HOLD_SEC).ready_answer(_buttons(stick=1.0), 0) is ReadyAnswer.CANCELLED
        assert _map(HOLD_SEC).ready_answer(_buttons(), 0) is None

    @pytest.mark.parametrize('threshold', [0.0, -0.1, float('nan'), 1.5])
    def test_a_hold_threshold_outside_zero_to_one_is_refused(self, threshold):
        with pytest.raises(ValueError, match='hold_threshold must be in'):
            ButtonMap(HOLD_SEC, threshold)

    @pytest.mark.parametrize('hold_sec', [0.0, float('nan'), float('inf')])
    def test_a_hold_time_that_is_not_finite_and_positive_is_refused(self, hold_sec):
        with pytest.raises(ValueError, match='hold_sec must be finite and positive'):
            ButtonMap(hold_sec)

    def test_a_turns_control_on_or_off_at_each_press(self):
        buttons = _map(HOLD_SEC)
        assert buttons.events(_buttons(a=1.0), 0) == [Event.CLUTCH]
        assert buttons.events(_buttons(a=1.0), 10) == []
        assert buttons.events(_buttons(), 20) == []
        assert buttons.events(_buttons(a=1.0), 30) == [Event.CLUTCH]

    def test_a_hold_of_b_completes_once(self):
        buttons = _map(HOLD_SEC)
        assert buttons.events(_buttons(b=1.0), 0) == []
        assert buttons.events(_buttons(b=1.0), 300_000_000) == [Event.COMPLETE]
        assert buttons.events(_buttons(b=1.0), 600_000_000) == []

    def test_the_b_press_of_ready_does_not_complete(self):
        buttons = _map(HOLD_SEC)
        buttons.ready_answer(_buttons(b=1.0), 0)
        assert buttons.events(_buttons(b=1.0), 1_000_000_000) == []

    def test_a_hold_of_the_stick_gives_up_once(self):
        buttons = _map(HOLD_SEC)
        assert buttons.events(_buttons(stick=1.0), 0) == []
        assert buttons.events(_buttons(stick=1.0), 300_000_000) == [Event.GIVE_UP]
        assert buttons.events(_buttons(stick=1.0), 600_000_000) == []

    def test_a_value_below_the_hold_threshold_holds_nothing(self):
        buttons = _map(HOLD_SEC, hold_threshold=0.9)
        assert buttons.events(_buttons(b=0.8), 0) == []
        assert buttons.events(_buttons(b=0.8), 300_000_000) == []
        assert buttons.events(_buttons(b=0.95), 310_000_000) == []
        assert buttons.events(_buttons(b=0.95), 610_000_000) == [Event.COMPLETE]

    def test_a_press_that_went_down_and_up_between_two_samples_turns_control_on(self):
        buttons = _map(HOLD_SEC)
        assert buttons.events(_buttons(presses={Button.A: 1}), 0) == [Event.CLUTCH]

    def test_an_odd_number_of_presses_between_two_samples_is_one_clutch_and_an_even_number_two(self):
        buttons = _map(HOLD_SEC)
        assert buttons.events(_buttons(presses={Button.A: 2}), 0) == [Event.CLUTCH] * 2
        assert buttons.events(_buttons(presses={Button.A: 5}), 10) == [Event.CLUTCH]
        assert buttons.events(_buttons(presses={Button.A: 105}), 20) == [Event.CLUTCH] * 2

    def test_the_first_sample_sets_the_press_counts_and_presses_nothing(self):
        buttons = ButtonMap(HOLD_SEC)
        assert buttons.ready_answer(_buttons(presses={Button.B: 7}), 0) is None
        assert buttons.ready_answer(_buttons(presses={Button.B: 8}), 10) is ReadyAnswer.READY

    def test_a_press_count_that_goes_down_presses_nothing(self):
        """A page that loads again counts from 0."""
        buttons = _map(HOLD_SEC)
        assert buttons.events(_buttons(presses={Button.A: 1}), 0) == [Event.CLUTCH]
        assert buttons.events(_buttons(presses={Button.A: 0}), 10) == []
        assert buttons.events(_buttons(presses={Button.A: 1}), 20) == [Event.CLUTCH]

    def test_a_sample_with_any_count_that_went_down_sets_all_the_counts_to_count_from(self):
        """A page that loads again counts each button from 0, and can count a press of A before its first sample."""
        buttons = _map(HOLD_SEC)
        assert buttons.events(_buttons(presses={Button.TRIGGER: 4}), 0) == []
        assert buttons.events(_buttons(presses={Button.A: 1}), 10) == []
        assert buttons.events(_buttons(presses={Button.A: 2}), 20) == [Event.CLUTCH]

    def test_the_indices_match_the_buttons_that_the_phone_page_names(self):
        page = Path(package_assets_path('assets/webxr_phone/index.html')).read_text()
        [layout] = re.findall(r'const BUTTON = \{ trigger: (\d+), stick: (\d+), a: (\d+), b: (\d+) \};', page)
        assert tuple(map(int, layout)) == (Button.TRIGGER, Button.STICK, Button.A, Button.B)

    def test_the_trigger_is_the_grip(self):
        assert ButtonMap.grip(_buttons(trigger=0.7).values) == 0.7

    def test_a_hold_does_not_count_the_time_the_controller_is_not_tracked(self):
        buttons = _map(HOLD_SEC)
        assert buttons.events(_buttons(b=1.0), 0) == []
        buttons.cancel_holds()
        assert buttons.events(_buttons(b=1.0), 300_000_000) == []
        assert buttons.events(_buttons(), 310_000_000) == []
        assert buttons.events(_buttons(b=1.0), 320_000_000) == []
        assert buttons.events(_buttons(b=1.0), 620_000_000) == [Event.COMPLETE]

    def test_the_map_of_the_left_controller_reads_only_x(self):
        buttons = _map(HOLD_SEC, buttons=(Button.X,))
        assert buttons.ready_answer(_buttons(b=1.0), 0) is None
        assert buttons.ready_answer(_buttons(stick=1.0), 10) is None
        assert buttons.events(_buttons(a=1.0, b=1.0, stick=1.0), 20) == [Event.CLUTCH]
        assert buttons.events(_buttons(a=1.0, b=1.0, stick=1.0), 1_000_000_000) == []


@pytest.fixture
def service(socket_path):
    with TeleopService(ServedUnixSocket(socket_path), ready_limit=2.0, silence_limit=0.1, hold_sec=HOLD_SEC) as served:
        yield served


@pytest.fixture
def runtime():
    executor = Executor(time.monotonic_ns, simulated=False, charge=InferenceCharge.NONE)
    yield executor
    executor.close()


def _policy(socket_path: Path, **params) -> RemotePolicy:
    """An ordinary remote policy that dials the service, with ``params`` in the query of its address."""
    address = UnixSocketAddress(socket_path, SESSION_PATH, session_query(**params))
    return RemotePolicy('websocket_unix', address, infer_timeout=10.0)


def _client(socket_path: Path, query: str = '') -> InferenceClient:
    address = UnixSocketAddress(socket_path, SESSION_PATH, query)
    return InferenceClient(registry.client_wire('websocket_unix'), address, infer_timeout=5, connect_deadline=5)


def _take(service: TeleopService, buttons: ControllerButtons) -> None:
    """Pass one sample of the right controller at ``HAND`` with ``buttons`` to ``service``."""
    service.take_sample({webxr.RIGHT: HAND}, {webxr.RIGHT: buttons})


def _pressed_after(previous: ControllerButtons, buttons: ControllerButtons) -> ControllerButtons:
    """The values of ``buttons`` with the press counts that a page sends after ``previous``: each button that goes down
    counts one press more."""
    if len(buttons.values) != len(previous.values):
        return buttons
    counts = tuple(
        count + int(value >= 0.5 > before)
        for count, value, before in zip(previous.presses, buttons.values, previous.values, strict=True)
    )
    return ControllerButtons(buttons.values, counts)


class _Headset:
    """A headset that sends one sample of both controllers every 10 ms, with the poses and the button values that a
    test sets. It counts the presses as the page does. ``pose`` and ``buttons`` are of the right controller."""

    def __init__(self, service: TeleopService):
        self._service = service
        self.pose: geom.Transform3D | None = HAND
        self._buttons = _buttons()
        self.left_pose: geom.Transform3D | None = LEFT_HAND
        self._left_buttons = _buttons()
        self.sent = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._send, daemon=True)

    @property
    def buttons(self) -> ControllerButtons:
        return self._buttons

    @buttons.setter
    def buttons(self, buttons: ControllerButtons) -> None:
        self._buttons = _pressed_after(self._buttons, buttons)

    @property
    def left_buttons(self) -> ControllerButtons:
        return self._left_buttons

    @left_buttons.setter
    def left_buttons(self, buttons: ControllerButtons) -> None:
        self._left_buttons = _pressed_after(self._left_buttons, buttons)

    def _send(self) -> None:
        while not self._stop.is_set():
            poses = {webxr.LEFT: self.left_pose, webxr.RIGHT: self.pose}
            self._service.take_sample(poses, {webxr.LEFT: self.left_buttons, webxr.RIGHT: self.buttons})
            self.sent += 1
            time.sleep(0.01)

    def __enter__(self) -> '_Headset':
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _samples(self, samples: int) -> None:
        """Wait for at least ``samples`` samples, so a loaded machine does not miss the buttons that they send."""
        until = self.sent + samples
        deadline = time.monotonic() + 5
        while self.sent < until:
            assert time.monotonic() < deadline, 'the headset sent no sample'
            time.sleep(0.005)

    def press(self, trigger=0.0, stick=0.0, a=0.0, b=0.0) -> None:
        """Press the buttons of the right controller for a few samples, and release them."""
        self.buttons = _buttons(trigger, stick, a, b)
        self._samples(3)
        self.buttons = _buttons()
        self._samples(3)

    def press_left(self, trigger=0.0, stick=0.0, a=0.0, b=0.0) -> None:
        """Press the buttons of the left controller for a few samples, and release them. X is at the index of A."""
        self.left_buttons = _buttons(trigger, stick, a, b)
        self._samples(3)
        self.left_buttons = _buttons()
        self._samples(3)


def _until_phase(service: TeleopService, phase: Phase, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while service.phase is not phase:
        assert time.monotonic() < deadline, f'the service never reached {phase}'
        time.sleep(0.005)


def _until_cleared(service: TeleopService) -> None:
    """Wait until the session ends, and check that the service shows no trial."""
    _until_phase(service, Phase.IDLE)
    state = service.state()
    assert (state.task, state.control, state.cameras) == ('', False, ())


def _press_ready_when_asked(service: TeleopService, headset: _Headset, **buttons: float) -> threading.Thread:
    def press() -> None:
        _until_phase(service, Phase.READY_WAIT)
        headset.press(**buttons)

    thread = threading.Thread(target=press, daemon=True)
    thread.start()
    return thread


def _drive(run, seconds: float, obs: dict = OBS) -> tuple[list[Step], PolicyEnd | None]:
    """Send ``obs`` to the run for ``seconds``. Return its steps, and its end if it ended."""
    steps = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            step = run.send(obs)
        except StopIteration as end:
            return steps, end.value
        assert step is not None
        steps.append(step)
        time.sleep(0.005)
    return steps, None


def _moves(steps: list[Step]) -> list[Step]:
    return [step for step in steps if keys.ROBOT_COMMAND in step.commands]


def _is_move(step: Step) -> bool:
    return keys.ROBOT_COMMAND in step.commands


def _drive_until(run, condition: Callable[[Step], bool], seconds: float = 5.0, obs: dict = OBS) -> Step:
    """Send ``obs`` to the run until it makes a step for which ``condition`` holds, and return that step."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        step = run.send(obs)
        assert step is not None
        if condition(step):
            return step
        time.sleep(0.005)
    raise AssertionError(f'the run made no such step within {seconds} s')


def _target(step: Step) -> geom.Transform3D:
    return step.commands[keys.ROBOT_COMMAND].pose


def _readied_run(service: TeleopService, headset: '_Headset', runtime: Executor, socket_path: Path, **params):
    """A run of a remote policy that the teleoperator readied. Its session prepares until Ready."""
    pressing = _press_ready_when_asked(service, headset, b=1.0)
    run = runtime.start(_policy(socket_path, **params))
    # The press releases B after Ready. A test press before that release would be lost in it.
    pressing.join()
    return run


def test_the_meta_of_a_remote_policy_before_a_run_opens_no_session(service, socket_path):
    assert _policy(socket_path, operator_position='BACK').meta() == {policy_keys.TYPE: 'remote'}
    assert service.phase is Phase.IDLE


def test_a_teleoperator_readies_takes_control_and_completes_a_trial(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        assert service.phase is Phase.RUNNING

        first = _drive_until(run, _is_move)
        np.testing.assert_allclose(_target(first).translation, ROBOT_POSE.translation, atol=1e-6)
        assert first.commands[keys.TARGET_GRIP] == OBS[keys.GRIP]
        assert not service.state().control

        headset.press(a=1.0)
        _drive_until(run, lambda _: service.state().control)

        headset.buttons = _buttons(b=1.0)
        _, end = _drive(run, 2.0)
        assert end is PolicyEnd.COMPLETE
        run.close()
        _until_cleared(service)


def test_control_that_went_off_in_a_silence_stays_off_until_a_and_the_view_and_the_log_name_the_cause(
    socket_path, connect, caplog
):
    caplog.set_level(logging.INFO, logger=service_module.__name__)
    with TeleopService(ServedUnixSocket(socket_path), ready_limit=5.0, silence_limit=0.3, hold_sec=HOLD_SEC) as served:
        connection = _readied_connection(served, socket_path, connect)
        _take(served, _buttons(a=1.0))
        _take(served, _buttons())
        connection.frame(FRAME_REQUEST)
        assert served.state().control
        time.sleep(served.silence_limit + 0.05)
        connection.frame(FRAME_REQUEST)
        _take(served, _buttons())
        connection.frame(FRAME_REQUEST)
        assert not served.state().control, 'samples that come again do not turn control on'
        assert (
            Badge('Control off: connection lost (A: on)', TeleopService.CONTROL_OFF_COLOUR) in served.view_text().lines
        )
        silences = [record for record in caplog.records if 'silence limit' in record.getMessage()]
        assert len(silences) == 1
        assert silences[0].levelno == logging.INFO
        assert 'Control off' in silences[0].getMessage()
        _take(served, _buttons(a=1.0))
        _take(served, _buttons())
        connection.frame(FRAME_REQUEST)
        assert served.state().control
        assert Badge('Control on (A: off)', TeleopService.CONTROL_ON_COLOUR) in served.view_text().lines
        connection.close()


class _LockProbe(logging.Handler):
    """Each line that the process logs, and whether another thread could read the service while the line was
    written."""

    def __init__(self, service: TeleopService):
        super().__init__(logging.INFO)
        self._service = service
        self.lines: list[tuple[str, bool]] = []

    def emit(self, record: logging.LogRecord) -> None:
        reader = threading.Thread(target=self._service.shown_image_count, daemon=True)
        reader.start()
        reader.join(timeout=2)
        self.lines.append((record.getMessage(), not reader.is_alive()))


@pytest.fixture
def lock_probe(service, caplog) -> Generator[_LockProbe]:
    caplog.set_level(logging.INFO)
    probe = _LockProbe(service)
    logging.getLogger().addHandler(probe)
    yield probe
    logging.getLogger().removeHandler(probe)


def test_the_service_logs_each_change_of_the_phase_and_of_control_with_its_cause_after_the_lock(
    service, socket_path, connect, lock_probe
):
    connection = _readied_connection(service, socket_path, connect)
    _take(service, _buttons(a=1.0))
    _take(service, _buttons())
    connection.frame(FRAME_REQUEST)
    _take(service, _buttons(a=1.0))
    _take(service, _buttons())
    connection.frame(FRAME_REQUEST)
    _take(service, _buttons(b=1.0))
    time.sleep(HOLD_SEC + 0.05)
    _take(service, _buttons(b=1.0))
    assert END in connection.frame(FRAME_REQUEST)
    connection.close()
    _until_phase(service, Phase.IDLE)
    messages = [message for message, _ in lock_probe.lines]
    changes = ['-> ready_wait', '-> running (B: Ready)', 'Control on (A)', 'Control off (A)']
    changes += ['-> ended (hold B: Complete)', '-> idle (the session ended)']
    unread = iter(messages)
    for change in changes:
        assert any(change in message for message in unread), f'no log line with {change!r} in order in {messages}'
    assert [message for message, free in lock_probe.lines if not free] == [], 'each line is written after the lock'


def _press_a(service: TeleopService, connection) -> None:
    _take(service, _buttons(a=1.0))
    _take(service, _buttons())
    connection.frame(FRAME_REQUEST)


def _ready_in_umi_mode(service: TeleopService, connection) -> None:
    connection.close()


def _control_on_with_a(service: TeleopService, connection) -> None:
    _press_a(service, connection)
    connection.close()


def _control_off_with_a(service: TeleopService, connection) -> None:
    _press_a(service, connection)
    _press_a(service, connection)
    connection.close()


def _control_off_in_a_silence(service: TeleopService, connection) -> None:
    _press_a(service, connection)
    time.sleep(service.silence_limit + 0.05)
    connection.frame(FRAME_REQUEST)
    connection.close()


def _control_off_with_an_untracked_controller(service: TeleopService, connection) -> None:
    _press_a(service, connection)
    service.take_sample({webxr.RIGHT: None}, {webxr.RIGHT: _buttons()})
    connection.frame(FRAME_REQUEST)
    connection.close()


def _control_off_at_the_end_of_the_session(service: TeleopService, connection) -> None:
    _press_a(service, connection)
    connection.frame(FRAME_REQUEST)
    connection.close()
    _until_phase(service, Phase.IDLE)


def _control_off_at_the_end_of_a_dropped_trial(service: TeleopService, connection) -> None:
    connection.frame(FRAME_REQUEST)
    connection.close()
    time.sleep(TeleopService.RESUME_LIMIT_SEC + 0.05)
    service.view()


# Each site that turns control of an arm on or off, by its cause: the trackers that Ready makes, a press of A, a
# silence, a controller that the headset does not track, and the end of a trial
@pytest.mark.parametrize(
    ('umi', 'drive', 'line'),
    [
        (True, _ready_in_umi_mode, 'Control on (B: Ready, in UMI mode)'),
        (False, _control_on_with_a, 'Control on (A)'),
        (False, _control_off_with_a, 'Control off (A)'),
        (False, _control_off_in_a_silence, 'Control off: connection lost, no controller data within the silence limit'),
        (False, _control_off_with_an_untracked_controller, 'Control off: the headset does not track the controller'),
        (False, _control_off_at_the_end_of_the_session, 'Control off (the session ended)'),
        (True, _control_off_at_the_end_of_a_dropped_trial, 'Control off (no session resumed the dropped trial)'),
    ],
)
def test_each_change_of_control_logs_one_line_with_its_cause_after_the_lock(
    service, socket_path, connect, lock_probe, monkeypatch, umi, drive, line
):
    monkeypatch.setattr(TeleopService, 'RESUME_LIMIT_SEC', 0.2)
    query = session_query(operator_position=None) if umi else ''
    drive(service, _readied_connection(service, socket_path, connect, query))
    changes = [free for message, free in lock_probe.lines if message.startswith(line)]
    assert changes == [True], f'one log line {line!r}, written after the lock, in {lock_probe.lines}'


def test_the_view_logs_the_end_of_a_dropped_trial_after_the_lock(
    service, socket_path, connect, lock_probe, monkeypatch
):
    monkeypatch.setattr(TeleopService, 'RESUME_LIMIT_SEC', 0.2)
    connection = _readied_connection(service, socket_path, connect)
    connection.frame(FRAME_REQUEST)
    connection.close()
    time.sleep(TeleopService.RESUME_LIMIT_SEC + 0.05)
    service.view()
    ended = [(message, free) for message, free in lock_probe.lines if 'no session resumed the dropped trial' in message]
    assert ended == [('Teleop trial: running -> idle (no session resumed the dropped trial)', True)]


def test_a_silence_in_umi_mode_leaves_control_on_and_logs_no_change_of_control(
    service, socket_path, connect, lock_probe
):
    connection = _readied_connection(service, socket_path, connect, session_query(operator_position=None))
    _take(service, _buttons())
    connection.frame(FRAME_REQUEST)
    time.sleep(service.silence_limit + 0.05)
    connection.frame(FRAME_REQUEST)
    assert service.state().control
    assert Badge('Control on (A: off)', TeleopService.CONTROL_ON_COLOUR) in service.view_text().lines
    assert [message for message, _ in lock_probe.lines if message.startswith('Control off')] == []
    connection.close()


def test_the_control_line_of_the_view_is_yellow_while_control_is_off_and_green_while_it_is_on(
    service, socket_path, runtime
):
    def colours() -> set[str]:
        view = service.view()
        shown = {'on': TeleopService.CONTROL_ON_COLOUR, 'off': TeleopService.CONTROL_OFF_COLOUR}
        return {name for name, colour in shown.items() if (view == colour).all(axis=-1).any()}

    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        _drive_until(run, _is_move)
        assert colours() == {'off'}
        headset.press(a=1.0)
        _drive_until(run, lambda _: service.state().control)
        assert colours() == {'on'}
        run.close()


def test_the_view_shows_that_the_trial_starts_until_its_first_frame(service, socket_path, connect):
    connection = _readied_connection(service, socket_path, connect)
    starting = service.view_text().lines
    assert starting == ('Trial starting: waiting for the robot',), 'no task and no control line before the first frame'
    connection.frame(FRAME_REQUEST)
    running = service.view_text().lines
    assert OBS[keys.TASK] in running
    assert Badge('Control off (A: on)', TeleopService.CONTROL_OFF_COLOUR) in running
    assert 'Trial starting: waiting for the robot' not in running
    connection.close()


def test_a_dropped_trial_with_no_frame_ends_at_once_and_the_view_waits_for_a_trial(service, socket_path, connect):
    connection = _readied_connection(service, socket_path, connect)
    connection.close()
    _until_cleared(service)
    assert service.view_text().lines == ('Waiting for a trial',)


def test_control_follows_the_hand_from_where_it_came_on_and_holds_the_last_target(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        headset.press(a=1.0)
        _drive_until(run, lambda _: service.state().control)
        headset.pose = geom.Transform3D(HAND.translation + np.array([0.0, 0.0, 0.05]), HAND.rotation)
        moved = _drive_until(
            run,
            lambda step: (
                _is_move(step) and bool(np.linalg.norm(_target(step).translation - ROBOT_POSE.translation) > 0.04)
            ),
        )
        shift = _target(moved).translation - ROBOT_POSE.translation
        np.testing.assert_allclose(np.linalg.norm(shift), 0.05, atol=1e-6)
        headset.press(a=1.0)
        _drive_until(run, lambda _: not service.state().control)
        headset.pose = HAND
        steps, _ = _drive(run, 0.3)
        assert _moves(steps)
        for step in _moves(steps):
            np.testing.assert_allclose(_target(step).translation, _target(moved).translation, atol=1e-6)
        run.close()


def test_a_controller_with_too_few_buttons_turns_control_off_and_the_robot_holds(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        headset.press(a=1.0)
        headset.buttons = _buttons(trigger=0.8)
        held = _drive_until(run, lambda step: _is_move(step) and step.commands[keys.TARGET_GRIP] == pytest.approx(0.8))
        headset.buttons = ControllerButtons((), ())
        headset.pose = _raised(HAND)
        _drive_until(run, lambda _: not service.state().control)
        headset._samples(3)
        steps, end = _drive(run, 0.3)
        assert end is None and _moves(steps)
        for step in _moves(steps):
            np.testing.assert_allclose(_target(step).translation, _target(held).translation, atol=1e-6)
            assert step.commands[keys.TARGET_GRIP] == pytest.approx(0.8)
        run.close()


def test_with_no_operator_position_the_hand_pose_is_the_robot_pose_and_control_is_on(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, operator_position=None)
        moved = _drive_until(run, _is_move)
        np.testing.assert_allclose(_target(moved).translation, HAND.translation, atol=1e-6)
        run.close()


def _raised(pose: geom.Transform3D) -> geom.Transform3D:
    return geom.Transform3D(pose.translation + np.array([0.0, 0.0, 0.05]), pose.rotation)


def _arm_target(step: Step, arm: str) -> geom.Transform3D:
    return step.commands[ARM_COMMANDS[arm]].pose


def _arm_moved(step: Step, arm: str, start: geom.Transform3D) -> bool:
    return ARM_COMMANDS[arm] in step.commands and bool(
        np.linalg.norm(_arm_target(step, arm).translation - start.translation) > 0.04
    )


def test_in_a_bimanual_session_each_controller_drives_its_arm_from_where_its_control_came_on(
    service, socket_path, runtime
):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, arms='BIMANUAL')
        first = _drive_until(run, lambda step: ARM_COMMANDS[keys.LEFT_ARM] in step.commands, obs=BIMANUAL_OBS)
        for arm, pose in ARM_POSES.items():
            np.testing.assert_allclose(_arm_target(first, arm).translation, pose.translation, atol=1e-6)
            assert first.commands[ARM_TARGET_GRIPS[arm]] == ARM_GRIPS[arm]

        headset.press_left(a=1.0)
        _drive_until(run, lambda _: service.state().control, obs=BIMANUAL_OBS)
        headset.left_pose, headset.left_buttons = _raised(LEFT_HAND), _buttons(trigger=0.8)
        headset.pose = _raised(HAND)
        left = keys.LEFT_ARM
        moved = _drive_until(run, lambda step: _arm_moved(step, left, ARM_POSES[left]), obs=BIMANUAL_OBS)
        shift = _arm_target(moved, left).translation - ARM_POSES[left].translation
        np.testing.assert_allclose(np.linalg.norm(shift), 0.05, atol=1e-6)
        assert moved.commands[ARM_TARGET_GRIPS[left]] == pytest.approx(0.8)
        # Control of the right arm is off, so the right arm holds while the right hand moves.
        np.testing.assert_allclose(
            _arm_target(moved, keys.RIGHT_ARM).translation, ARM_POSES[keys.RIGHT_ARM].translation
        )

        headset.press(a=1.0)
        # A frame applies the press, so frames must run before the hand moves.
        _drive(run, 0.5, obs=BIMANUAL_OBS)
        headset.pose = _raised(_raised(HAND))
        right = keys.RIGHT_ARM
        moved = _drive_until(run, lambda step: _arm_moved(step, right, ARM_POSES[right]), obs=BIMANUAL_OBS)
        shift = _arm_target(moved, right).translation - ARM_POSES[right].translation
        np.testing.assert_allclose(np.linalg.norm(shift), 0.05, atol=1e-6)
        np.testing.assert_allclose(_arm_target(moved, left).translation, ARM_POSES[left].translation + shift, atol=1e-6)
        run.close()


def test_in_a_bimanual_session_in_umi_mode_each_arm_is_at_the_pose_of_its_hand(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, operator_position=None, arms='BIMANUAL')
        moved = _drive_until(run, lambda step: ARM_COMMANDS[keys.LEFT_ARM] in step.commands, obs=BIMANUAL_OBS)
        np.testing.assert_allclose(_arm_target(moved, keys.LEFT_ARM).translation, LEFT_HAND.translation, atol=1e-6)
        np.testing.assert_allclose(_arm_target(moved, keys.RIGHT_ARM).translation, HAND.translation, atol=1e-6)
        run.close()


def test_in_a_one_arm_session_the_left_controller_drives_nothing(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        headset.press_left(a=1.0)
        headset.left_pose = _raised(LEFT_HAND)
        steps, _ = _drive(run, 0.3)
        assert _moves(steps) and not service.state().control
        for step in _moves(steps):
            np.testing.assert_allclose(_target(step).translation, ROBOT_POSE.translation, atol=1e-6)
        run.close()


@pytest.mark.parametrize('params', [{}, {'video_bitrate': 2_000_000}], ids=['jpeg', 'h264'])
def test_the_service_shows_the_camera_image_of_the_run(service, socket_path, runtime, params):
    width, height = VIEW_SIZE
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, cameras=[keys.EXTERIOR_IMAGE], **params)
        # The view is black until the first frame request brings the image.
        _drive_until(run, lambda _: service.view()[height // 2, width // 2].any())
        view = service.view()
        assert view.shape == (height, width, 3)
        assert abs(int(view[height // 2, width // 2, 0]) - 128) < 10
        run.close()


def test_the_client_sends_the_cameras_that_the_session_names(service, socket_path, runtime):
    obs = {**OBS, keys.WRIST_IMAGE: np.zeros((24, 32, 3), dtype=np.uint8)}
    with _Headset(service) as headset:
        _press_ready_when_asked(service, headset, b=1.0)
        run = runtime.start(_policy(socket_path, cameras=[keys.WRIST_IMAGE]))
        deadline = time.monotonic() + 5
        while service.state().cameras != (keys.WRIST_IMAGE,):
            assert time.monotonic() < deadline, 'the service got no camera image'
            run.send(obs)
            time.sleep(0.01)
        run.close()


def test_a_cancel_fails_the_run_before_its_first_step(service, socket_path, runtime):
    with _Headset(service) as headset:
        _press_ready_when_asked(service, headset, stick=1.0)
        with pytest.raises(RuntimeError, match='cancelled the trial'):
            runtime.start(_policy(socket_path))
    _until_cleared(service)


def test_no_ready_within_the_ready_limit_fails_the_run(socket_path, runtime):
    with TeleopService(ServedUnixSocket(socket_path), ready_limit=0.3, silence_limit=0.1, hold_sec=HOLD_SEC) as service:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match='ready limit'):
            runtime.start(_policy(socket_path))
        assert time.monotonic() - started < 0.3 + 2.0
        _until_cleared(service)


def test_the_end_of_the_ready_limit_says_when_no_controller_data_came_in_the_wait(socket_path):
    service = TeleopService(ServedUnixSocket(socket_path), ready_limit=0.2, silence_limit=0.1, hold_sec=HOLD_SEC)
    with pytest.raises(service_module.TrialNotStarted, match='ready limit.*no controller data came in the wait'):
        service.wait_for_ready(protocol.SessionParams(), threading.Event())
    with _Headset(service), pytest.raises(service_module.TrialNotStarted, match='ready limit') as raised:
        service.wait_for_ready(protocol.SessionParams(), threading.Event())
    assert 'controller data' not in str(raised.value)


def test_the_state_says_whether_a_sample_came_within_the_silence_limit(socket_path):
    silence_limit = 0.5
    service = TeleopService(ServedUnixSocket(socket_path), 1.0, silence_limit, HOLD_SEC)
    assert not service.state().tracked
    before_sample = time.monotonic()
    _take(service, _buttons())
    assert service.state().tracked
    deadline = before_sample + silence_limit + 5.0
    while service.state().tracked:
        assert time.monotonic() < deadline, 'the state stays tracked long after the silence limit'
        time.sleep(0.01)
    assert time.monotonic() - before_sample >= silence_limit


def test_a_sample_that_tracks_no_controller_is_no_controller_data(socket_path):
    service = TeleopService(ServedUnixSocket(socket_path), ready_limit=0.2, silence_limit=60.0, hold_sec=HOLD_SEC)
    no_controller = {webxr.LEFT: None, webxr.RIGHT: None}
    service.take_sample(no_controller, no_controller)
    service.take_sample({webxr.RIGHT: HAND}, {webxr.RIGHT: ControllerButtons((1.0,), (0,))})
    assert not service.state().tracked
    headset = _Headset(service)
    headset.pose = headset.left_pose = None
    headset.buttons = headset.left_buttons = ControllerButtons((), ())
    with headset, pytest.raises(service_module.TrialNotStarted) as raised:
        service.wait_for_ready(protocol.SessionParams(), threading.Event())
    assert 'no controller data came in the wait' in str(raised.value)


def test_a_ready_wait_longer_than_the_infer_timeout_still_ends_with_ready(socket_path, runtime):
    """The wait for Ready is the preparation of the session, so the infer timeout does not bound it."""
    with TeleopService(
        ServedUnixSocket(socket_path), ready_limit=10.0, silence_limit=0.1, hold_sec=HOLD_SEC
    ) as service:
        policy = RemotePolicy('websocket_unix', UnixSocketAddress(socket_path, SESSION_PATH, ''), infer_timeout=1.5)
        with _Headset(service) as headset:

            def press_late() -> None:
                _until_phase(service, Phase.READY_WAIT)
                time.sleep(2.5)
                headset.press(b=1.0)

            pressing = threading.Thread(target=press_late, daemon=True)
            pressing.start()
            run = runtime.start(policy)
            assert service.phase is Phase.RUNNING
            pressing.join()
            run.close()


# A teleoperator on a real rig can take about this long to get ready, and the opening of the session waits for it.
TELEOPERATOR_GETS_READY_SEC = 300.0


class _FakeTime:
    """The ``time`` module as one module sees it, on a clock that moves only when that module sleeps or waits.

    It replaces the module's ``time`` name, not the attributes of ``time``: a thread that an earlier test left
    running and that sleeps would otherwise move this clock too."""

    time_ns = staticmethod(time.time_ns)

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


# A client stack the fake teleop server declares on Ready.
_TELEOP_STACK = Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=1.0))


class _ReadyWaitOnAFakeClock(wire.ClientConnection):
    """A teleop session on the fake clock ``clock``: a waiting status at each interval of the model server, and Ready
    at ``ready_at``. A wait that ends before the next message advances the clock and times out. After Ready, the
    server echoes the end of the session."""

    def __init__(self, clock: _FakeTime, ready_at: float):
        self._clock = clock
        self._ready_at = ready_at
        self._next_waiting = clock.now
        self._sent: list[bytes] = []

    def send(self, message: bytes, timeout: float | None = None) -> None:
        assert self._clock.now >= self._ready_at, 'the opening sends nothing before Ready'
        self._sent.append(message)

    def recv(self, timeout: float | None = None) -> bytes:
        if self._sent:
            return self._sent.pop(0)
        arrives = min(self._next_waiting, self._ready_at)
        if timeout is not None and arrives - self._clock.now > timeout:
            self._clock.now += timeout
            raise TimeoutError('no message within the timeout')
        self._clock.now = max(self._clock.now, arrives)
        if arrives == self._ready_at:
            return serialise({
                STATUS: ServerStatus.READY,
                PROTOCOL_VERSION: ProtocolVersion.V3.value,
                META: {server_keys.LOCAL_STACK: _TELEOP_STACK.to_spec()},
                SESSION_ID: 'teleop',
            })
        self._next_waiting += ModelServer.WAITING_INTERVAL_SEC
        return serialise({STATUS: ServerStatus.WAITING})

    def close(self, timeout: float | None = None) -> str:
        return 'closed'


def test_the_default_deadlines_of_a_teleop_policy_outlast_a_five_minute_wait_for_ready(runtime):
    """A run of ``cfg.policy.teleop`` with its default deadlines starts when Ready comes after five minutes."""
    clock = _FakeTime()
    ready_at = TELEOPERATOR_GETS_READY_SEC + ModelServer.WAITING_INTERVAL_SEC
    policy = positronic.cfg.policy.teleop.instantiate()
    with (
        patch.object(WebsocketUnixClientWire, 'dial', return_value=_ReadyWaitOnAFakeClock(clock, ready_at)),
        patch('positronic.offboard.client.time', clock),
    ):
        run = runtime.start(policy)
    try:
        assert runtime.metadata_snapshot()[policy_keys.SESSION_ID] == 'teleop'
        assert clock.now == ready_at
    finally:
        run.close()


def test_a_silent_headset_reads_as_not_tracked_and_turns_control_off(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        headset.press(a=1.0)
        _drive_until(run, lambda _: service.state().control)
        headset.pose = None
        _drive_until(run, lambda _: not service.state().control)
        held = _target(_drive_until(run, _is_move))
        headset.pose = geom.Transform3D(HAND.translation + np.array([0.0, 0.0, 0.05]), HAND.rotation)
        steps, _ = _drive(run, 0.3)
        for step in _moves(steps):
            np.testing.assert_allclose(_target(step).translation, held.translation, atol=1e-6)
        run.close()


def test_the_service_refuses_a_second_session_while_a_trial_runs(service, socket_path, runtime):
    with _Headset(service) as headset:
        first = _readied_run(service, headset, runtime, socket_path)
        with pytest.raises(RuntimeError, match='one session at a time'):
            _client(socket_path).new_session()
        first.close()


def test_a_session_with_a_wrong_operator_position_is_refused_at_connect(service, socket_path):
    with pytest.raises(RuntimeError, match='operator_position must be one of'):
        _client(socket_path, 'operator_position="SIDE"').new_session()


def test_a_session_with_a_headset_camera_that_it_does_not_send_is_refused_at_connect(service, socket_path):
    with pytest.raises(RuntimeError, match='headset_camera must be one of the cameras'):
        _client(socket_path, 'headset_camera="image.wrist"').new_session()


def test_a_session_with_unknown_arms_is_refused_at_connect(service, socket_path):
    with pytest.raises(RuntimeError, match='arms must be one of ONE, BIMANUAL'):
        _client(socket_path, 'arms="BOTH"').new_session()


def test_a_session_with_a_jpeg_quality_out_of_range_is_refused_at_connect(service, socket_path):
    with pytest.raises(RuntimeError, match='jpeg_quality must be an integer between 0 and 100'):
        _client(socket_path, 'jpeg_quality=101').new_session()
    assert service.phase is Phase.IDLE


def test_a_session_query_refuses_a_wrong_value_or_an_unknown_parameter():
    with pytest.raises(ValueError, match='operator_position must be one of'):
        session_query(operator_position='SIDE')
    with pytest.raises(TypeError, match='fps'):
        session_query(fps=10)
    with pytest.raises(ValueError, match='cameras must be a list of camera names'):
        session_query(cameras=keys.WRIST_IMAGE)
    with pytest.raises(ValueError, match='headset_camera must be one of the cameras'):
        session_query(cameras=[keys.WRIST_IMAGE], headset_camera=keys.EXTERIOR_IMAGE)
    with pytest.raises(ValueError, match='must not be a field of a frame request'):
        session_query(cameras=[protocol.TASK])
    with pytest.raises(ValueError, match='must not be a field of a frame request'):
        session_query(arms='BIMANUAL', cameras=[keys.arm_channel(protocol.GRIP, keys.LEFT_ARM)])
    with pytest.raises(ValueError, match='jpeg_quality must be an integer between 0 and 100'):
        session_query(jpeg_quality=90.5)
    with pytest.raises(ValueError, match='video_bitrate must be a positive integer'):
        session_query(video_bitrate=0)
    with pytest.raises(ValueError, match='video_fps must be a positive integer'):
        session_query(video_fps=2.5)
    with pytest.raises(ValueError, match='warm_up must be true or false'):
        session_query(warm_up='yes')
    assert session_query(operator_position=None, cameras=[keys.WRIST_IMAGE]) == (
        'operator_position=null&cameras=%5B%22image.wrist%22%5D'
    )


def test_session_params_refuse_an_operator_position_that_is_no_operator_position():
    with pytest.raises(ValueError, match="operator_position must be one of .*, got '.BACK'"):
        protocol.SessionParams('.BACK')  # pyright: ignore[reportArgumentType]
    assert protocol.SessionParams(None).operator() is None


def test_the_client_stack_ends_the_run_outside_a_schedule_that_asks_for_one_frame_at_a_time():
    stack = TeleopModel.client_stack(protocol.SessionParams(cameras=(keys.WRIST_IMAGE,)))
    end, player, schedule = stack['seq'][:3]
    assert end['name'] == EndOnField.WIRE_NAME
    assert player['name'] == TrajectoryPlayer.WIRE_NAME
    assert schedule['name'] == InferenceSchedule.WIRE_NAME
    assert schedule['args']['latency_estimate'] >= 1 / schedule['args']['fps']
    assert EndCommand.WIRE_NAME in json.dumps(stack['seq'][3])


@pytest.mark.parametrize('arms', list(protocol.Arms))
def test_the_client_stack_states_the_grip_point_as_the_frame_of_the_pose_and_the_target_of_each_arm(arms):
    client_stack = TeleopModel.client_stack(protocol.SessionParams(cameras=(keys.WRIST_IMAGE,), arms=arms))
    stack = declared_stack({server_keys.LOCAL_STACK: client_stack}, ProtocolVersion.V3)
    frames = ee_frames(stack.meta())
    for arm in arms.value:
        pose = f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.EE_POSE_SUFFIX}'
        target = f'{keys.arm_channel(keys.ROBOT_COMMAND, arm)}{keys.POSE_SUFFIX}'
        assert frames[pose] == frames[target] == EEFrame(site=GRIPPER_SITE)


@pytest.mark.parametrize(
    'params, timing',
    [({}, RequestTiming.before_end()), ({'video_bitrate': 1_000_000, 'video_fps': 15}, RequestTiming.every(1 / 15))],
    ids=['jpeg', 'h264'],
)
def test_an_h264_session_asks_for_frames_at_most_at_the_video_frame_rate(params, timing):
    stack = TeleopModel.client_stack(protocol.SessionParams(cameras=(keys.WRIST_IMAGE,), **params))
    schedule = stack['seq'][2]
    assert schedule['args']['request_timing'] == timing.to_spec()


@pytest.mark.parametrize(
    'params, encoder',
    [
        ({'jpeg_quality': 70}, {'name': EncodeImages.WIRE_NAME, 'args': {'quality': 70}}),
        (
            {'video_bitrate': 1_000_000, 'video_fps': 15},
            {
                'name': EncodeVideo.WIRE_NAME,
                'args': {'paths': [[keys.WRIST_IMAGE], [keys.EXTERIOR_IMAGE]], 'bitrate': 1_000_000, 'fps': 15},
            },
        ),
    ],
    ids=['jpeg', 'h264'],
)
def test_the_client_stack_sends_the_camera_images_as_jpeg_or_with_a_video_bit_rate_as_h264(params, encoder):
    cameras = (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)
    stack = TeleopModel.client_stack(protocol.SessionParams(cameras=cameras, **params))
    last = stack['seq'][-1]['seq'][-1]
    assert {'name': last['name'], 'args': last['args']} == encoder


class _BlankRenderer:
    """A MuJoCo renderer that draws black images, so the test needs no GL."""

    def __init__(self, _model, *, height, width, max_geom=10000, font_scale=None):
        self.height, self.width = height, width

    def update_scene(self, _data, camera=None):
        pass

    def render(self, out=None):
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if out is None:
            return image
        out[:] = image
        return None

    def close(self):
        pass


@pytest.mark.timeout(120)
def test_a_person_completes_a_trial_on_mujoco_franka_through_a_remote_policy(socket_path, tmp_path, monkeypatch):
    monkeypatch.setattr('positronic.simulator.mujoco.sim.mj.Renderer', _BlankRenderer)
    # The spans of each processor record only at the step level.
    monkeypatch.setenv(telemetry.ENV_TELEMETRY_LEVEL, telemetry.Level.STEP.name.lower())
    ev = stack_cubes(
        mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
        loaders=positronic.cfg.simulator.stack_cubes_loaders(),
        camera_fps=10,
        camera_dict={
            keys.EXTERIOR_IMAGE: 'back_view_ph',
            keys.WRIST_IMAGE: 'handcam_left_ph',
            keys.HANDCAM_RIGHT_IMAGE: 'handcam_right_ph',
        },
        instruction='Pick up the green cube.',
        timeout=60,
    )

    def teleoperate(service: TeleopService, headset: _Headset) -> None:
        # The sim loads before the first trial waits for Ready, and a loaded machine loads it slowly.
        _until_phase(service, Phase.READY_WAIT, seconds=60)
        headset.press(b=1.0)
        _until_phase(service, Phase.RUNNING)
        time.sleep(0.3)
        headset.press(a=1.0)
        headset.pose = geom.Transform3D(HAND.translation + np.array([0.0, 0.0, 0.03]), HAND.rotation)
        time.sleep(0.3)
        headset.buttons = _buttons(b=1.0)

    service = TeleopService(ServedUnixSocket(socket_path), ready_limit=60.0, silence_limit=0.1, hold_sec=HOLD_SEC)
    policy = _policy(socket_path, operator_position='BACK', cameras=[keys.EXTERIOR_IMAGE])
    with service, _Headset(service) as headset, pos3.mirror():
        threading.Thread(target=teleoperate, args=(service, headset), daemon=True).start()
        main(policy=policy, evals=[ev], output_dir=str(tmp_path))

    [episode] = LocalDataset(tmp_path)
    policy_meta = policy_keys.POLICY_META
    assert episode.static[eval_keys.ENDED_BY] == eval_keys.EndedBy.POLICY
    assert episode.static[eval_keys.POLICY_END] == PolicyEnd.COMPLETE
    assert episode.static[f'{policy_meta}.{policy_keys.TYPE}'] == 'remote'
    assert episode.static[f'{policy_meta}.{policy_keys.SERVER}.{policy_keys.TYPE}'] == 'teleop'
    effective = f'{policy_meta}.{policy_keys.SERVER}.{server_keys.EFFECTIVE_PARAMS}'
    assert episode.static[f'{effective}.operator_position'] == 'BACK'
    # The first frame reply carries a target, so each command signal starts one round trip after the first state.
    first_state = episode.signals[keys.EE_POSE].timestamps(EMITTED_WORLD)[0]
    for name in (keys.TARGET_EE_POSE, keys.TARGET_GRIP):
        assert episode.signals[name].timestamps(EMITTED_WORLD)[0] - first_state < 500_000_000
    names = {span.name for span in telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS))}
    assert {
        telemetry.component_name(Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=1.0))),
        telemetry_keys.SPAN_POLICY_INFER,
    } <= names


class TestButtonsBetweenTrials:
    def test_a_button_released_after_the_trial_presses_again_in_the_next(self):
        buttons = ButtonMap(HOLD_SEC)
        buttons.events(_buttons(b=1.0), 0)
        assert buttons.events(_buttons(b=1.0), 500_000_000) == [Event.COMPLETE]
        buttons.observe(_buttons(), 600_000_000)
        assert buttons.ready_answer(_buttons(b=1.0), 700_000_000) is ReadyAnswer.READY

    def test_a_button_held_from_the_last_trial_is_no_ready(self):
        buttons = ButtonMap(HOLD_SEC)
        buttons.observe(_buttons(b=1.0), 0)
        assert buttons.ready_answer(_buttons(b=1.0), 100_000_000) is None

    def test_a_stick_held_from_the_last_trial_gives_no_give_up_in_the_next(self):
        buttons = ButtonMap(HOLD_SEC)
        assert buttons.events(_buttons(stick=1.0), 0) == []
        buttons.observe(_buttons(stick=1.0), 100_000_000)
        assert buttons.ready_answer(_buttons(stick=1.0, b=1.0), 500_000_000) is ReadyAnswer.READY
        assert buttons.events(_buttons(stick=1.0), 900_000_000) == []

    def test_a_stick_held_into_the_ready_wait_gives_no_give_up_in_the_trial(self):
        buttons = ButtonMap(HOLD_SEC)
        assert buttons.events(_buttons(stick=1.0), 0) == []
        assert buttons.ready_answer(_buttons(stick=1.0, b=1.0), 500_000_000) is ReadyAnswer.READY
        assert buttons.events(_buttons(stick=1.0), 900_000_000) == []


def _raw_connection(socket_path: Path):
    """A websocket on the service, before its session is ready."""
    return unix_connect(str(socket_path), f'ws://localhost{SESSION_PATH}')


def _raw_session(service: TeleopService, socket_path: Path):
    """A websocket on the service whose session the teleoperator readied, and the ID of its session."""
    websocket = _raw_connection(socket_path)
    _until_phase(service, Phase.READY_WAIT)
    _take(service, _buttons())
    _take(service, _buttons(b=1.0))
    _take(service, _buttons())
    while (message := deserialise(websocket.recv(5)))[STATUS] != ServerStatus.READY:
        pass
    return websocket, message[SESSION_ID]


def _drop(websocket) -> None:
    """End the connection as a client process that exits does: with no close of the session."""
    websocket.socket.shutdown(socket.SHUT_RDWR)
    websocket.socket.close()


def _frames(websocket, session_id: str, count: int) -> None:
    for _ in range(count):
        websocket.send(serialise({SESSION_ID: session_id, OBSERVATION: FRAME_REQUEST}))
        deserialise(websocket.recv(5))[RESULT]


def test_a_stop_of_the_service_ends_its_waits(socket_path):
    service = TeleopService(ServedUnixSocket(socket_path), ready_limit=60.0, silence_limit=0.1, hold_sec=HOLD_SEC)
    with service:
        websocket = _raw_connection(socket_path)
        _until_phase(service, Phase.READY_WAIT)
        stopped_at = time.monotonic()
    assert time.monotonic() - stopped_at < 5
    websocket.close()


def test_a_service_that_cannot_bind_raises_the_cause_at_once(tmp_path):
    started_at = time.monotonic()
    with pytest.raises(OSError, match='too long'):
        with TeleopService(ServedUnixSocket(tmp_path / ('x' * 120) / 'teleop.sock'), 1.0, 0.1, HOLD_SEC):
            pass
    assert time.monotonic() - started_at < 5


def test_a_model_server_that_fails_after_its_start_stops_the_process(socket_path, monkeypatch):
    def serve_then_fail(self, wires, on_ready):
        on_ready()
        raise RuntimeError('the event loop failed')

    interrupted = threading.Event()
    monkeypatch.setattr(service_module.ModelServer, 'serve', serve_then_fail)
    monkeypatch.setattr(service_module._thread, 'interrupt_main', interrupted.set)
    monkeypatch.setattr(service_module.ModelServer, 'shutdown', lambda self: None)
    with TeleopService(ServedUnixSocket(socket_path), 1.0, 0.1, HOLD_SEC):
        assert interrupted.wait(timeout=5)


@pytest.mark.parametrize('name', ['ready_limit', 'silence_limit', 'heads_up_limit'])
@pytest.mark.parametrize('value', [0.0, -1.0, float('nan'), float('inf')])
def test_a_time_limit_that_is_not_finite_and_positive_is_refused(socket_path, name, value):
    limits = {'ready_limit': 1.0, 'silence_limit': 0.05, name: value}
    with pytest.raises(ValueError, match=f'{name} must be finite and positive'):
        TeleopService(ServedUnixSocket(socket_path), hold_sec=HOLD_SEC, **limits)


def test_a_socket_directory_that_others_can_open_is_refused(tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(PermissionError, match='mode 700'):
        with TeleopService(ServedUnixSocket(shared / 'teleop.sock'), 1.0, 0.1, HOLD_SEC):
            pass


def test_a_service_on_a_host_and_a_port_with_no_token_opens_a_session_with_no_token():
    with TeleopService(ServedHostPort('127.0.0.1', 0), 5.0, 0.1, HOLD_SEC) as service:
        served = service.served_address
        assert isinstance(served, ServedHostPort)
        address = HostPortAddress(served.host, served.port, SESSION_PATH, '')

        def cancel_when_asked() -> None:
            _until_phase(service, Phase.READY_WAIT)
            service.cancel_ready_wait()

        opening = threading.Thread(target=cancel_when_asked)
        opening.start()
        with pytest.raises(RuntimeError, match='cancelled the trial'):
            InferenceClient(registry.client_wire('websocket'), address, connect_deadline=5.0).new_session()
        opening.join()


def test_a_service_on_a_host_and_a_port_opens_a_session_only_with_its_token():
    with TeleopService(ServedHostPort('127.0.0.1', 0), 5.0, 0.1, HOLD_SEC, auth_token='secret') as service:
        served = service.served_address
        assert isinstance(served, ServedHostPort)
        address = HostPortAddress(served.host, served.port, SESSION_PATH, '')
        wire = registry.client_wire('websocket')
        with pytest.raises(OSError, match='HTTP 403'):
            InferenceClient(wire, address, connect_deadline=1.0).new_session()
        token = {AUTH_HEADER: bearer('secret')}

        def cancel_when_asked() -> None:
            _until_phase(service, Phase.READY_WAIT)
            service.cancel_ready_wait()

        opening = threading.Thread(target=cancel_when_asked)
        opening.start()
        with pytest.raises(RuntimeError, match='cancelled the trial'):
            InferenceClient(wire, address, headers=token, connect_deadline=5.0).new_session()
        opening.join()


def test_the_default_socket_with_no_runtime_directory_is_in_a_directory_of_the_user(monkeypatch):
    monkeypatch.delenv('XDG_RUNTIME_DIR', raising=False)
    assert protocol.default_socket_path().parent.name == f'positronic-{os.getuid()}'


def _readied_connection(service: TeleopService, socket_path: Path, connect, query: str = ''):
    """A test session with ``query`` that the teleoperator readied from the headset."""
    connection = connect(socket_path, query)
    _until_phase(service, Phase.READY_WAIT)
    _take(service, _buttons())
    _take(service, _buttons(b=1.0))
    _take(service, _buttons())
    connection.wait_open()
    return connection


def test_a_hold_of_b_completes_in_one_frame(service, socket_path, connect):
    connection = _readied_connection(service, socket_path, connect)
    _take(service, _buttons(b=1.0))
    time.sleep(HOLD_SEC + 0.05)
    _take(service, _buttons(b=1.0))
    ended = connection.frame(FRAME_REQUEST)
    assert ended[END] == PolicyEnd.COMPLETE.value
    np.testing.assert_allclose(ended[ACTION], [*OBS[keys.EE_POSE], OBS[keys.GRIP]])
    assert service.phase is Phase.ENDED


def test_each_frame_request_after_the_end_gets_the_same_end(service, socket_path, connect):
    connection = _readied_connection(service, socket_path, connect)
    _take(service, _buttons(stick=1.0))
    time.sleep(HOLD_SEC + 0.05)
    _take(service, _buttons(stick=1.0))
    ended = connection.frame(FRAME_REQUEST)
    again = connection.frame(FRAME_REQUEST)
    assert ended[END] == again[END] == PolicyEnd.GIVE_UP.value
    np.testing.assert_allclose(again[ACTION], ended[ACTION])


def test_each_frame_reply_has_a_target_from_the_first(service, socket_path, connect):
    connection = _readied_connection(service, socket_path, connect)
    first = connection.frame(FRAME_REQUEST)
    np.testing.assert_allclose(first[ACTION], [*OBS[keys.EE_POSE], OBS[keys.GRIP]])


def test_a_frame_answers_at_once_with_the_newest_sample(socket_path, connect):
    with TeleopService(ServedUnixSocket(socket_path), ready_limit=5.0, silence_limit=2.0, hold_sec=HOLD_SEC) as served:
        connection = _readied_connection(served, socket_path, connect)
        _take(served, _buttons(a=1.0))
        _take(served, _buttons())
        started = time.monotonic()
        connection.frame(FRAME_REQUEST)
        assert time.monotonic() - started < 1.0
        assert served.state().control
        connection.close()


def test_a_press_of_a_that_went_down_and_up_between_two_samples_turns_control_on(socket_path, connect):
    with TeleopService(ServedUnixSocket(socket_path), ready_limit=5.0, silence_limit=2.0, hold_sec=HOLD_SEC) as served:
        connection = _readied_connection(served, socket_path, connect)
        # No sample shows A down: only the press count of A steps
        _take(served, _buttons(presses={Button.A: 1}))
        connection.frame(FRAME_REQUEST)
        assert served.state().control
        connection.close()


def test_a_press_of_a_while_the_controller_is_not_tracked_turns_control_on_when_it_is_tracked_again(
    socket_path, connect
):
    with TeleopService(ServedUnixSocket(socket_path), ready_limit=5.0, silence_limit=2.0, hold_sec=HOLD_SEC) as served:
        connection = _readied_connection(served, socket_path, connect)
        served.take_sample({webxr.RIGHT: None}, {webxr.RIGHT: None})
        _take(served, _buttons(presses={Button.A: 1}))
        connection.frame(FRAME_REQUEST)
        assert served.state().control
        connection.close()


def test_two_presses_of_a_between_two_samples_take_control_again_from_the_new_place_of_the_hand(socket_path, connect):
    with TeleopService(ServedUnixSocket(socket_path), ready_limit=5.0, silence_limit=2.0, hold_sec=HOLD_SEC) as served:
        connection = _readied_connection(served, socket_path, connect)
        _take(served, _buttons(presses={Button.A: 1}))
        connection.frame(FRAME_REQUEST)
        assert served.state().control
        # A goes off and on again while the hand moves 5 cm
        moved = geom.Transform3D(HAND.translation + np.array([0.0, 0.0, 0.05]), HAND.rotation)
        served.take_sample({webxr.RIGHT: moved}, {webxr.RIGHT: _buttons(presses={Button.A: 3})})
        reply = connection.frame(FRAME_REQUEST)
        assert served.state().control
        np.testing.assert_allclose(reply[ACTION][:3], OBS[keys.EE_POSE][:3], atol=1e-6)
        connection.close()


def test_a_frame_with_no_sample_within_the_silence_limit_has_no_pose_and_turns_control_off(socket_path, connect):
    with TeleopService(ServedUnixSocket(socket_path), ready_limit=5.0, silence_limit=1.0, hold_sec=HOLD_SEC) as served:
        connection = _readied_connection(served, socket_path, connect)
        _take(served, _buttons(a=1.0))
        _take(served, _buttons())
        connection.frame(FRAME_REQUEST)
        assert served.state().control
        time.sleep(served.silence_limit + 0.05)
        connection.frame(FRAME_REQUEST)
        assert not served.state().control
        connection.close()


def test_the_view_leaves_the_camera_image_of_the_display_unchanged(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, cameras=[keys.EXTERIOR_IMAGE])
        _drive_until(run, lambda _: service.state().cameras == (keys.EXTERIOR_IMAGE,))
        stored = service.camera_image(keys.EXTERIOR_IMAGE)
        before = stored.copy()
        service.view()
        np.testing.assert_array_equal(stored, before)
        run.close()


def test_the_service_takes_no_camera_image_outside_a_running_trial(service):
    service.take_images({keys.EXTERIOR_IMAGE: OBS[keys.EXTERIOR_IMAGE]})
    assert service.state().cameras == () and service.camera_feeds() == {}


def test_the_camera_feeds_count_the_images_of_the_last_feed_span(service, socket_path, connect, monkeypatch):
    _readied_connection(service, socket_path, connect, session_query(cameras=[keys.EXTERIOR_IMAGE]))
    for _ in range(4):
        service.take_images({keys.EXTERIOR_IMAGE: OBS[keys.EXTERIOR_IMAGE]})
    [(name, feed)] = service.camera_feeds().items()
    assert name == keys.EXTERIOR_IMAGE and service.state().cameras == (keys.EXTERIOR_IMAGE,)
    assert feed.rate_hz == 4 / TeleopService.FEED_SEC and 0 <= feed.age_sec < 1
    monkeypatch.setattr(TeleopService, 'FEED_SEC', 0.05)
    time.sleep(0.1)
    assert service.camera_feeds() == {}


@pytest.mark.parametrize('headset_camera', [None, keys.EXTERIOR_IMAGE])
def test_the_headset_shows_the_first_camera_or_the_one_that_the_session_names(socket_path, connect, headset_camera):
    images = {
        keys.WRIST_IMAGE: np.full((24, 32, 3), 50, np.uint8),
        keys.EXTERIOR_IMAGE: np.full((48, 64, 3), 200, np.uint8),
    }
    query = session_query(cameras=list(images), headset_camera=headset_camera)
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC) as service:
        connection = _readied_connection(service, socket_path, connect, query)
        encoded = {name: encode_jpeg(image) for name, image in images.items()}
        connection.frame({**FRAME_REQUEST, **encoded})
        shown = keys.WRIST_IMAGE if headset_camera is None else headset_camera
        assert _main_level(service.view()) == images[shown][0, 0, 0]
        assert service.state().cameras == (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)
        connection.close()


def _main_level(view: np.ndarray) -> int:
    """The value at the centre of the part of ``view`` that shows the main camera."""
    width, height = VIEW_SIZE
    return int(view[height // 2, width // 2, 0])


def _column_levels(view: np.ndarray, count: int) -> list[int]:
    """The value at the centre of each of the ``count`` cells of the column of the other cameras."""
    width, height = VIEW_SIZE
    cell = height // count
    return [int(view[row * cell + cell // 2, width + OTHERS_WIDTH // 2, 0]) for row in range(count)]


def test_the_headset_shows_the_other_cameras_in_a_column_beside_the_main_camera(socket_path, connect):
    levels = {keys.EXTERIOR_IMAGE: 200, keys.WRIST_IMAGE: 50, keys.EXTERIOR_IMAGE_2: 120}
    query = session_query(cameras=list(levels))
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC) as service:
        connection = _readied_connection(service, socket_path, connect, query)
        images = {name: encode_jpeg(np.full((48, 64, 3), level, np.uint8)) for name, level in levels.items()}
        frame = {**FRAME_REQUEST, **images}
        connection.frame(frame)
        connection.frame(frame)
        width, height = VIEW_SIZE
        view = service.view()
        assert view.shape == (height, width + OTHERS_WIDTH, 3)
        assert _main_level(view) == 200
        assert _column_levels(view, 2) == [50, 120]
        service.choose_camera(keys.WRIST_IMAGE)
        assert _main_level(service.view()) == 50
        assert _column_levels(service.view(), 2) == [200, 120]
        assert _column_levels(service.view(camera=False), 2) == [0, 0], 'a clear panel shows the text alone'
        connection.close()


def test_the_headset_view_keeps_its_size_from_the_wait_for_a_trial_to_the_trial(socket_path, connect):
    station = (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC, cameras=station) as service:
        idle = service.view()
        connection = _readied_connection(service, socket_path, connect)
        image = encode_jpeg(np.full((24, 32, 3), 50, np.uint8))
        connection.frame({**FRAME_REQUEST, **dict.fromkeys(station, image)})
        assert service.view().shape == idle.shape
        connection.close()


def test_the_wait_for_ready_shows_the_cameras_of_its_session_and_not_those_of_the_last_run(socket_path, connect):
    image = encode_jpeg(np.full((24, 32, 3), 50, np.uint8))
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC) as service:
        last = _readied_connection(service, socket_path, connect, session_query(cameras=[keys.WRIST_IMAGE]))
        for _ in range(2):
            last.frame({**FRAME_REQUEST, keys.WRIST_IMAGE: image})
        last.close()
        _until_cleared(service)
        cameras = [keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2]
        connection = connect(socket_path, session_query(cameras=cameras))
        _until_phase(service, Phase.READY_WAIT)
        assert service.cameras().names == tuple(cameras)
        waiting = service.view()
        _take(service, _buttons(b=1.0))
        _take(service, _buttons())
        connection.wait_open()
        connection.frame({**FRAME_REQUEST, **dict.fromkeys(cameras, image)})
        assert service.view().shape == waiting.shape
        connection.close()


def test_a_wait_for_ready_that_fails_leaves_the_cameras_of_the_station(socket_path):
    session = protocol.SessionParams(cameras=(keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2))
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC, cameras=[keys.WRIST_IMAGE]) as service:

        def cancel_when_asked() -> None:
            _until_phase(service, Phase.READY_WAIT)
            service.cancel_ready_wait()

        cancelling = threading.Thread(target=cancel_when_asked)
        cancelling.start()
        with pytest.raises(service_module.TrialNotStarted, match='cancelled the trial'):
            service.wait_for_ready(session, threading.Event())
        cancelling.join()
        assert service.cameras().names == (keys.WRIST_IMAGE,)


@pytest.mark.parametrize('headset_camera', [None, keys.EXTERIOR_IMAGE])
def test_the_camera_that_the_teleoperator_chose_is_the_main_camera_of_the_wait_for_ready(
    socket_path, connect, headset_camera
):
    station = [keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE]
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC, cameras=station) as service:
        service.choose_camera(keys.WRIST_IMAGE)
        connection = connect(socket_path, session_query(cameras=station, headset_camera=headset_camera))
        _until_phase(service, Phase.READY_WAIT)
        assert service.cameras().shown == keys.WRIST_IMAGE
        connection.close()


def test_a_session_of_one_camera_shows_that_camera_with_no_column(socket_path, connect):
    query = session_query(cameras=[keys.WRIST_IMAGE])
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC) as service:
        connection = _readied_connection(service, socket_path, connect, query)
        connection.frame({**FRAME_REQUEST, keys.WRIST_IMAGE: encode_jpeg(np.zeros((48, 64, 3), np.uint8))})
        assert service.view().shape == (48, 64, 3)
        connection.close()


def test_a_session_that_names_no_cameras_gets_the_cameras_of_the_station(socket_path, connect):
    """So a plan need not name the cameras of the rig that the station runs."""
    station = (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC, cameras=station) as service:
        assert service.cameras() == CameraState(station, (), None, None)
        connection = _readied_connection(service, socket_path, connect)
        image = encode_jpeg(np.full((24, 32, 3), 50, np.uint8))
        connection.frame({**FRAME_REQUEST, **dict.fromkeys(station, image)})
        assert service.state().cameras == station
        assert service.cameras() == CameraState(station, station, keys.WRIST_IMAGE, None)
        connection.close()


def test_a_station_camera_list_that_is_one_string_is_refused(socket_path):
    with pytest.raises(ValueError, match='cameras must be a list of camera names'):
        TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC, cameras=keys.WRIST_IMAGE)


def test_the_cameras_that_a_session_names_go_before_the_cameras_of_the_station(socket_path, connect):
    query = session_query(cameras=[keys.EXTERIOR_IMAGE])
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC, cameras=[keys.WRIST_IMAGE]) as service:
        connection = _readied_connection(service, socket_path, connect, query)
        connection.frame({**FRAME_REQUEST, keys.EXTERIOR_IMAGE: encode_jpeg(np.zeros((24, 32, 3), np.uint8))})
        assert service.state().cameras == (keys.EXTERIOR_IMAGE,)
        connection.close()


def test_the_camera_that_the_teleoperator_chose_goes_before_the_one_of_the_session_and_holds(socket_path, connect):
    images = {
        keys.WRIST_IMAGE: np.full((24, 32, 3), 50, np.uint8),
        keys.EXTERIOR_IMAGE: np.full((48, 64, 3), 200, np.uint8),
    }
    query = session_query(cameras=list(images), headset_camera=keys.WRIST_IMAGE)
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC) as service:
        connection = _readied_connection(service, socket_path, connect, query)
        frame = {**FRAME_REQUEST, **{name: encode_jpeg(image) for name, image in images.items()}}
        # Two frames: a trial that answered one frame stays for a session that resumes it
        connection.frame(frame)
        connection.frame(frame)
        service.choose_camera(keys.EXTERIOR_IMAGE)
        assert _main_level(service.view()) == 200
        assert service.cameras().shown == keys.EXTERIOR_IMAGE
        service.choose_camera(None)
        assert _main_level(service.view()) == 50
        service.choose_camera(keys.EXTERIOR_IMAGE)
        connection.close()
        _until_cleared(service)
        assert service.cameras().chosen == keys.EXTERIOR_IMAGE, 'the choice went with the trial'


def test_a_view_with_no_camera_shows_the_text_on_black(socket_path, connect):
    """A clear panel shows black as clear: with no camera image, only the text floats over the room."""
    query = session_query(cameras=[keys.WRIST_IMAGE])
    width, height = VIEW_SIZE
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC) as service:
        connection = _readied_connection(service, socket_path, connect, query)
        bright = np.full((height, width, 3), 200, np.uint8)
        connection.frame({**FRAME_REQUEST, keys.WRIST_IMAGE: encode_jpeg(bright)})
        assert service.view()[height // 2, width // 2].min() > 150
        assert not service.view(camera=False)[height // 2, width // 2].any()
        connection.close()


def test_a_view_with_no_task_shows_the_control_line_and_not_the_task(service, socket_path, connect):
    connection = _readied_connection(service, socket_path, connect)
    connection.frame({**FRAME_REQUEST, protocol.TASK: 'Pick the cube.'})
    with_task, no_task = service.view(), service.view(task=False)
    other_task = {**FRAME_REQUEST, protocol.TASK: 'Open the drawer and put the cup in it.'}
    connection.frame(other_task)
    assert not np.array_equal(service.view(), with_task), 'the view with the task shows the new task'
    np.testing.assert_array_equal(service.view(task=False), no_task, 'the view with no task shows no task')
    _take(service, _buttons(a=1.0))
    _take(service, _buttons())
    connection.frame(other_task)
    assert service.state().control
    assert not np.array_equal(service.view(task=False), no_task), 'the view with no task shows the control line'


def test_the_bottom_of_the_view_names_the_camera_and_the_use_of_y(socket_path, connect):
    query = session_query(cameras=[keys.WRIST_IMAGE])
    width, height = VIEW_SIZE
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC) as service:
        idle = service.view()
        connection = _readied_connection(service, socket_path, connect, query)
        connection.frame({**FRAME_REQUEST, keys.WRIST_IMAGE: encode_jpeg(np.zeros((height, width, 3), np.uint8))})
        running = service.view(camera=False)
        connection.close()
    # One line of the hint while no camera shows, and the camera above it while one does
    assert idle[height - 26 :].any() and not idle[height - 48 : height - 30].any()
    assert running[height - 48 : height - 30].any()


def test_a_long_task_wraps_to_the_width_of_the_main_camera(service, socket_path, connect):
    def text_bottom(view: np.ndarray) -> int:
        """The lowest row of the text at the top of a view on black."""
        return int(np.flatnonzero(view[: view.shape[0] // 2].any(axis=(1, 2))).max())

    task = 'Pick each cube from the left bin and place it in the right bin, then close the lid of the box.'
    connection = _readied_connection(service, socket_path, connect)
    connection.frame({**FRAME_REQUEST, protocol.TASK: 'Pick the cube.'})
    short = service.view()
    connection.frame({**FRAME_REQUEST, protocol.TASK: task})
    long = service.view()
    assert not long[:, -8:].any(), 'no text reaches the right margin of the view'
    assert text_bottom(long) > text_bottom(short), 'the long task takes more rows'


# A word of 43 letters just fits a row, so a row of it with '...' after it does not
@pytest.mark.parametrize(
    'task', ['Pick the ' + 'a' * 120 + ' cube.', ' '.join(['a' * 43] * 5)], ids=['a long word', 'a long line']
)
def test_each_row_of_a_long_task_fits_the_width_of_the_main_camera(service, socket_path, connect, task):
    connection = _readied_connection(service, socket_path, connect)
    connection.frame({**FRAME_REQUEST, protocol.TASK: task})
    assert not service.view()[:, -8:].any(), 'no text reaches the right margin of the view'


def test_a_task_of_many_rows_leaves_the_state_of_the_trial_at_the_top_of_the_view(service, socket_path, connect):
    task = ' '.join(['Pick each cube from the left bin and place it in the right bin.'] * 6)
    connection = _readied_connection(service, socket_path, connect)
    connection.frame({**FRAME_REQUEST, protocol.TASK: task})
    view = service.view()
    middle = view.shape[0] // 2
    assert not view[middle : middle + 30].any(), 'the task, the control line and the line of the ends stay at the top'


def test_a_camera_of_the_run_that_the_station_does_not_name_stays_listed_and_live_between_trials(socket_path, connect):
    query = session_query(cameras=[keys.EXTERIOR_IMAGE])
    image = encode_jpeg(np.full((24, 32, 3), 50, np.uint8))
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC, cameras=[keys.WRIST_IMAGE]) as service:
        connection = _readied_connection(service, socket_path, connect, query)
        connection.frame({**FRAME_REQUEST, keys.EXTERIOR_IMAGE: image})
        connection.frame({**FRAME_REQUEST, keys.EXTERIOR_IMAGE: image})
        connection.close()
        _until_cleared(service)
        cameras = service.cameras()
    assert cameras.names == (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)
    assert cameras.live == (keys.EXTERIOR_IMAGE,)


def test_a_camera_stays_live_between_trials_and_not_longer_than_the_live_window(socket_path, connect, monkeypatch):
    """Images come only while a trial runs, when the teleoperator is in the headset and does not see the page."""
    query = session_query(cameras=[keys.WRIST_IMAGE])
    image = encode_jpeg(np.full((24, 32, 3), 50, np.uint8))
    with TeleopService(ServedUnixSocket(socket_path), 5.0, 0.05, HOLD_SEC) as service:
        connection = _readied_connection(service, socket_path, connect, query)
        connection.frame({**FRAME_REQUEST, keys.WRIST_IMAGE: image})
        connection.frame({**FRAME_REQUEST, keys.WRIST_IMAGE: image})
        connection.close()
        _until_cleared(service)
        assert service.cameras().live == (keys.WRIST_IMAGE,)
        monkeypatch.setattr(TeleopService, 'LIVE_SEC', 0.0)
        assert service.cameras().live == ()


def test_the_headset_bridge_takes_the_view_that_the_page_asks_for(socket_path, monkeypatch):
    service = TeleopService(ServedUnixSocket(socket_path), 1.0, 0.05, HOLD_SEC)
    asked = []

    def view(camera: bool = True, task: bool = True) -> np.ndarray:
        asked.append((camera, task))
        return np.zeros((2, 2, 3), np.uint8)

    monkeypatch.setattr(service, 'view', view)
    bridge = command._Headset(service)
    bridge.page_state._bind(RecordingEmitter[dict](), clock=SystemClock())
    bridge.view._bind(RecordingEmitter[np.ndarray](), clock=SystemClock())
    bridge.controller_positions._bind(ManualCommandReceiver())
    bridge.buttons._bind(ManualCommandReceiver())
    page_input = ManualCommandReceiver()
    bridge.page_input._bind(page_input)
    now = [0.0]
    clock = cast(pimm.Clock, types.SimpleNamespace(now=lambda: now[0]))
    loop = bridge.run(cast(pimm.SignalReceiver, types.SimpleNamespace(value=False)), clock)
    next(loop)
    for text_only, task_hidden in ((False, True), (True, False), (False, False)):
        page_input.push({command.TEXT_ONLY: text_only, command.TASK_HIDDEN: task_hidden})
        now[0] += 1 / 16
        next(loop)
    assert asked == [(True, True), (True, False), (False, True), (True, True)]


def test_the_headset_bridge_logs_the_camera_feeds_once_each_log_span(socket_path, monkeypatch, caplog):
    service = TeleopService(ServedUnixSocket(socket_path), 1.0, 0.05, HOLD_SEC)
    monkeypatch.setattr(service, 'view', lambda camera=True, task=True: np.zeros((2, 2, 3), np.uint8))
    monkeypatch.setattr(service, 'camera_feeds', lambda: {keys.EXTERIOR_IMAGE: service_module.CameraFeed(9.5, 0.04)})
    bridge = command._Headset(service)
    bridge.page_state._bind(RecordingEmitter[dict](), clock=SystemClock())
    bridge.view._bind(RecordingEmitter[np.ndarray](), clock=SystemClock())
    for receiver in (bridge.controller_positions, bridge.buttons, bridge.page_input):
        receiver._bind(ManualCommandReceiver())
    now = [0.0]
    clock = cast(pimm.Clock, types.SimpleNamespace(now=lambda: now[0]))
    loop = bridge.run(cast(pimm.SignalReceiver, types.SimpleNamespace(value=False)), clock)
    with caplog.at_level(logging.INFO, logger=command.__name__):
        next(loop)
        now[0] = command._Headset.FEED_LOG_SEC
        next(loop)
        next(loop)
    assert [record.getMessage() for record in caplog.records] == [
        f'Camera images: {keys.EXTERIOR_IMAGE} at 9.5 Hz, the last 0.04 s ago'
    ]


def _bridge_on_a_fake_clock(socket_path, monkeypatch, view_fps):
    """A headset bridge at ``view_fps`` on a fake clock, the time of that clock, the count of the images of the shown
    camera, and the views that the bridge sends."""
    service = TeleopService(ServedUnixSocket(socket_path), 1.0, 0.05, HOLD_SEC)
    image_count = [0]
    monkeypatch.setattr(service, 'shown_image_count', lambda: image_count[0])
    bridge = command._Headset(service, view_fps=view_fps)
    views = RecordingEmitter[np.ndarray]()
    bridge.view._bind(views, clock=SystemClock())
    bridge.page_state._bind(RecordingEmitter[dict](), clock=SystemClock())
    for receiver in (bridge.controller_positions, bridge.buttons, bridge.page_input):
        receiver._bind(ManualCommandReceiver())
    now = [0.0]
    clock = cast(pimm.Clock, types.SimpleNamespace(now=lambda: now[0]))
    loop = bridge.run(cast(pimm.SignalReceiver, types.SimpleNamespace(value=False)), clock)
    return loop, now, image_count, views


def test_the_headset_bridge_sends_a_new_image_of_the_shown_camera_at_the_next_view_slot(socket_path, monkeypatch):
    loop, now, image_count, views = _bridge_on_a_fake_clock(socket_path, monkeypatch, view_fps=8.0)
    sent = []
    # One view period is 1/8 s, and the bridge sends the view again after two periods with no new image.
    for at, new_image in [(0, 0), (10, 1), (11, 1), (14, 0), (18, 0), (25, 0), (33, 0), (34, 0)]:
        now[0] = at / 64
        image_count[0] += new_image
        next(loop)
        sent.append(len(views.emitted))
    assert sent == [1, 2, 2, 2, 3, 3, 3, 4]


def test_the_headset_bridge_sends_at_most_one_view_each_view_period_under_a_burst_of_images(socket_path, monkeypatch):
    loop, now, image_count, views = _bridge_on_a_fake_clock(socket_path, monkeypatch, view_fps=8.0)
    for poll in range(256):
        now[0] = poll / 256
        image_count[0] += 1
        next(loop)
    assert len(views.emitted) == 8


def test_the_headset_bridge_sends_the_view_at_the_view_rate_under_a_steady_stream_of_images(socket_path, monkeypatch):
    loop, now, image_count, views = _bridge_on_a_fake_clock(socket_path, monkeypatch, view_fps=30.0)
    sent_at = []
    sleep = next(loop)
    while now[0] < 1.0:
        now[0] += sleep.seconds
        image_count[0] += 1
        sleep = next(loop)
        if len(views.emitted) > len(sent_at) + 1:
            sent_at.append(now[0])
    assert len(sent_at) >= 29
    np.testing.assert_allclose(np.diff(sent_at), 1 / 30)


def test_the_service_counts_only_the_images_of_the_shown_camera(service, socket_path, connect):
    query = session_query(cameras=[keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE], headset_camera=keys.WRIST_IMAGE)
    _readied_connection(service, socket_path, connect, query)
    image = OBS[keys.EXTERIOR_IMAGE]
    before = service.shown_image_count()
    service.take_images({keys.EXTERIOR_IMAGE: image})
    assert service.shown_image_count() == before
    service.take_images({keys.EXTERIOR_IMAGE: image, keys.WRIST_IMAGE: image})
    assert service.shown_image_count() == before + 1


def test_a_camera_that_neither_the_run_nor_the_station_has_cannot_be_chosen(socket_path):
    service = TeleopService(ServedUnixSocket(socket_path), 1.0, 0.05, HOLD_SEC, cameras=[keys.WRIST_IMAGE])
    with pytest.raises(ValueError, match='no camera'):
        service.choose_camera(keys.EXTERIOR_IMAGE)
    service.choose_camera(keys.WRIST_IMAGE)
    assert service.cameras().chosen == keys.WRIST_IMAGE


@pytest.mark.parametrize(
    ('camera', 'chosen'), [(keys.WRIST_IMAGE, keys.WRIST_IMAGE), ([], None), ({'name': keys.WRIST_IMAGE}, None)]
)
def test_the_headset_bridge_takes_the_camera_that_the_page_chose_and_logs_a_camera_that_is_no_name(
    socket_path, camera, chosen
):
    service = TeleopService(ServedUnixSocket(socket_path), 1.0, 0.05, HOLD_SEC, cameras=[keys.WRIST_IMAGE])
    bridge = command._Headset(service)
    sent = RecordingEmitter[dict]()
    bridge.page_state._bind(sent, clock=SystemClock())
    bridge.view._bind(RecordingEmitter[np.ndarray](), clock=SystemClock())
    bridge.controller_positions._bind(ManualCommandReceiver())
    bridge.buttons._bind(ManualCommandReceiver())
    page_input = ManualCommandReceiver()
    page_input.push({command.HEADSET_CAMERA: camera})
    bridge.page_input._bind(page_input)
    stop = types.SimpleNamespace(value=False)
    next(bridge.run(cast(pimm.SignalReceiver, stop), SystemClock()))
    [(_, state)] = sent.emitted
    assert state['chosen'] == chosen and state['cameras'] == (keys.WRIST_IMAGE,)


@pytest.mark.parametrize('view_fps', [0.0, -1.0, float('nan'), float('inf')])
def test_a_view_rate_that_is_not_finite_and_positive_is_refused(socket_path, view_fps):
    with pytest.raises(ValueError, match='view_fps must be finite and positive'):
        command._Headset(TeleopService(ServedUnixSocket(socket_path), 1.0, 0.05, HOLD_SEC), view_fps)


@pytest.mark.parametrize('name', ['rate_hz', 'view_fps', 'stale_limit', 'min_hold_sec', 'trial_sec', 'circle_sec'])
@pytest.mark.parametrize('value', [0.0, -1.0, float('nan'), float('inf')])
def test_a_scripted_headset_rate_or_time_that_is_not_finite_and_positive_is_refused(name, value):
    with pytest.raises(ValueError, match=f'{name} must be finite and positive'):
        scripted_headset.drive(page='http://127.0.0.1:1', trials=1, **{name: value})


@pytest.mark.parametrize('trials', [0, -1, 1.5])
def test_a_scripted_headset_trial_count_that_is_not_a_positive_integer_is_refused(trials):
    with pytest.raises(ValueError, match='trials must be a positive integer'):
        scripted_headset.drive(page='http://127.0.0.1:1', trials=trials)


@pytest.mark.parametrize('radius', [-0.01, float('nan'), float('inf')])
def test_a_scripted_headset_radius_that_is_negative_or_not_finite_is_refused(radius):
    with pytest.raises(ValueError, match='radius must be finite and nonnegative'):
        scripted_headset.drive(page='http://127.0.0.1:1', trials=1, radius=radius)


class _PagePassingASampleAtEachPoll:
    """A page of ``service`` that passes the latest sample of the headset to the service only when the page state is
    polled. So the service reads no sample of a press that starts and ends between two polls."""

    def __init__(self, service: TeleopService):
        self._service = service
        self._latest: dict | None = None
        self._server = serve(self._take_samples, '127.0.0.1', 0, process_request=self._answer_state)
        self.url = f'http://127.0.0.1:{self._server.socket.getsockname()[1]}'

    def __enter__(self) -> '_PagePassingASampleAtEachPoll':
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()

    def _take_samples(self, connection: ServerConnection) -> None:
        for message in connection:
            self._latest = json.loads(message)

    def _answer_state(self, connection: ServerConnection, request: Request) -> Response | None:
        if request.path != STATE_PATH:
            return None
        if self._latest is not None:
            positions, buttons = webxr.parse_controller_message(self._latest)
            right = buttons[webxr.RIGHT]
            if self._passes(right):
                self._service.take_sample(positions, buttons)
        state = self._service.state()
        self._answered(state.phase)
        return connection.respond(200, json.dumps(asdict(command.PageState(state.phase, state.control))))

    def _passes(self, buttons: ControllerButtons | None) -> bool:
        return True

    def _answered(self, phase: Phase) -> None:
        pass


class _PageMissingTheReleaseAfterComplete(_PagePassingASampleAtEachPoll):
    """A page that passes no sample with B up from the end of a trial until it showed the next Ready wait twice. So the
    service reads B down from Complete into the first press of Ready of the next trial."""

    def __init__(self, service: TeleopService):
        super().__init__(service)
        self._phase = Phase.IDLE
        # The answers of the next Ready wait still to show before a release of B passes, or 0 when it passes.
        self._ready_waits_to_miss = 0

    def _passes(self, buttons: ControllerButtons | None) -> bool:
        return not (self._ready_waits_to_miss and buttons is not None and buttons.values[Button.B] == 0.0)

    def _answered(self, phase: Phase) -> None:
        if self._phase is Phase.RUNNING and phase is not Phase.RUNNING:
            self._ready_waits_to_miss = 2
        elif self._ready_waits_to_miss and phase is Phase.READY_WAIT:
            self._ready_waits_to_miss -= 1
        self._phase = phase


class _PageLosingTheFirstPressOfA(_PagePassingASampleAtEachPoll):
    """A page that loses the first press of A, as the WebXR driver loses the presses of a stale message: the press
    counts of that sample become the base of the counts that the page passes."""

    def _take_samples(self, connection: ServerConnection) -> None:
        base, lost = None, False
        for message in connection:
            positions, buttons = webxr.parse_controller_message(json.loads(message))
            pose, right = positions[webxr.RIGHT], buttons[webxr.RIGHT]
            assert pose is not None and right is not None, 'the scripted headset tracks its right controller'
            if base is None:
                base = right.presses
            elif not lost and right.presses[Button.A] > base[Button.A]:
                base, lost = right.presses, True
            counted = replace(right, presses=tuple(now - then for now, then in zip(right.presses, base, strict=True)))
            self._latest = webxr.controller_message({webxr.RIGHT: pose}, {webxr.RIGHT: counted})


# A station asks for a frame about every 33 ms. A frame request answers at once, so a loop with no pause only
# competes with the headset for the service.
STATION_FRAME_SEC = 1 / 30


def _frames_until_end(service: TeleopService) -> list[dict]:
    """The elements of the frames of one trial, asked for at the pace of a station, the end element last."""
    elements = [service.next_frame(FRAME_REQUEST)]
    while END not in elements[-1]:
        time.sleep(STATION_FRAME_SEC)
        elements.append(service.next_frame(FRAME_REQUEST))
    return elements


@pytest.mark.timeout(30)
def test_the_scripted_headset_holds_each_press_until_a_slow_service_reads_it(socket_path):
    service = TeleopService(ServedUnixSocket(socket_path), ready_limit=2.0, silence_limit=1.0, hold_sec=HOLD_SEC)
    with _PagePassingASampleAtEachPoll(service) as page, ThreadPoolExecutor(max_workers=1) as pool:
        trials = pool.submit(
            scripted_headset.drive, page=page.url, trials=1, min_hold_sec=0.05, trial_sec=1.0, circle_sec=1.0
        )
        service.wait_for_ready(protocol.SessionParams(OperatorPosition.BACK), threading.Event())
        *frames, element = _frames_until_end(service)
        trials.result(timeout=10)
    moves = [frame[ACTION][:3] for frame in frames]
    assert element[END] == PolicyEnd.COMPLETE.value
    assert np.ptp(moves, axis=0).max() > 0.01, f'{len(moves)} moves with no motion'


@pytest.mark.timeout(30)
def test_the_scripted_headset_presses_ready_again_when_the_service_missed_the_release_of_complete(socket_path):
    service = TeleopService(ServedUnixSocket(socket_path), ready_limit=5.0, silence_limit=1.0, hold_sec=HOLD_SEC)
    with _PageMissingTheReleaseAfterComplete(service) as page, ThreadPoolExecutor(max_workers=1) as pool:
        trials = pool.submit(
            scripted_headset.drive, page=page.url, trials=2, min_hold_sec=0.05, trial_sec=0.5, circle_sec=1.0
        )
        ends = []
        for _ in range(2):
            service.wait_for_ready(protocol.SessionParams(OperatorPosition.BACK), threading.Event())
            ends.append(_frames_until_end(service)[-1][END])
        trials.result(timeout=10)
    assert ends == [PolicyEnd.COMPLETE.value, PolicyEnd.COMPLETE.value]


@pytest.mark.timeout(30)
def test_the_scripted_headset_presses_a_again_when_the_service_lost_its_press_of_a(socket_path):
    service = TeleopService(ServedUnixSocket(socket_path), ready_limit=2.0, silence_limit=1.0, hold_sec=HOLD_SEC)
    with _PageLosingTheFirstPressOfA(service) as page, ThreadPoolExecutor(max_workers=1) as pool:
        trials = pool.submit(
            scripted_headset.drive, page=page.url, trials=1, min_hold_sec=0.05, trial_sec=1.0, circle_sec=1.0
        )
        service.wait_for_ready(protocol.SessionParams(OperatorPosition.BACK), threading.Event())
        frames = [service.next_frame(FRAME_REQUEST)]
        deadline = time.monotonic() + 10.0
        while END not in frames[-1] and time.monotonic() < deadline:
            time.sleep(STATION_FRAME_SEC)
            frames.append(service.next_frame(FRAME_REQUEST))
        # A trial that does not end lets the scripted headset stop.
        service.end_session()
        trials.result(timeout=10)
    moves = [frame[ACTION][:3] for frame in frames[:-1]]
    assert frames[-1].get(END) == PolicyEnd.COMPLETE.value
    assert np.ptp(moves, axis=0).max() > 0.01, f'{len(moves)} moves with no motion'


def test_the_phone_page_reads_fields_of_the_page_state():
    page = Path(package_assets_path('assets/webxr_phone/index.html')).read_text()
    [fields] = re.findall(r'const STATE_FIELD = \{([^}]*)\};', page)
    assert set(re.findall(r"'(\w+)'", fields)) <= set(command.PageState.__dataclass_fields__)


def test_the_headset_page_names_the_routes_the_page_state_fields_and_the_input_fields_of_the_service():
    page = Path(package_assets_path('assets/webxr/index.html')).read_text()
    routes_and_inputs = [STATE_PATH, INPUT_PATH, command.HEADSET_CAMERA, command.TEXT_ONLY, command.TASK_HIDDEN]
    contract = [*routes_and_inputs, *command.PageState.__dataclass_fields__]
    assert [name for name in contract if not re.search(f"""['"]{re.escape(name)}['"]""", page)] == []


# The part of the travel of a Quest 3 trigger that reads 0
DEAD_ZONE = 0.17
QUEST_3 = ['meta-quest-touch-plus', 'oculus-touch-v3', 'oculus-touch', 'generic-trigger-squeeze-thumbstick']
needs_node = pytest.mark.skipif(shutil.which('node') is None, reason='node runs the modules of the headset page')


def _run_page_module(module: str, script: str, data: list) -> list:
    """The JSON that ``script`` prints under node for ``data`` on its input, with the URL of page ``module`` as its
    first argument."""
    url = Path(package_assets_path(f'assets/webxr/{module}')).as_uri()
    node = subprocess.run(
        ['node', '--input-type=module', '-e', script, url],
        input=json.dumps(data),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(node.stdout)


def _bend_quest_trigger(cases: list[dict]) -> list[list[float]]:
    script = """
    const { bendQuestTrigger } = await import(process.argv[1]);
    let input = '';
    for await (const chunk of process.stdin) input += chunk;
    console.log(JSON.stringify(JSON.parse(input).map(c => bendQuestTrigger(c.buttons, c.profiles))));
    """
    return _run_page_module('quest-trigger.js', script, cases)


def _press_counts(frames: list[list[bool]]) -> list[list[int]]:
    """The press counts that the headset page sends for a controller whose buttons are down as in each of ``frames``."""
    script = """
    const { PressCounter } = await import(process.argv[1]);
    let input = '';
    for await (const chunk of process.stdin) input += chunk;
    const counter = new PressCounter();
    console.log(JSON.stringify(JSON.parse(input).map(pressed => counter.count(pressed))));
    """
    return _run_page_module('press-count.js', script, frames)


@needs_node
def test_the_headset_page_counts_each_press_of_a_button_once():
    frames = [[False, False], [True, False], [True, False], [False, True], [True, True], [False, False]]
    assert _press_counts(frames) == [[0, 0], [1, 0], [1, 0], [1, 1], [2, 1], [2, 1]]


def _presses_in_frames(frames: list[tuple[list[bool], str]]) -> list[list[int] | None]:
    """The press counts that the headset page sends for a right controller in each of ``frames``, or ``None`` for a
    frame that sends no controller. Each frame gives the state of each button, and ``'pose'``, ``'no_grip_pose'`` (the
    controller has no pose) or ``'no_viewer_pose'`` (the headset has no pose, so the page draws no frame).

    The frame loop of ``startXRApp`` in ``core.js`` runs each frame, in a fake XR session with a fake WebSocket. The
    page hooks are those of the headset page."""
    script = """
    import { register } from 'node:module';
    const coreUrl = process.argv[1];
    // The server serves these two modules of core.js; the test has a stand-in for each
    const served = {
      '/session-mode.js': 'export const SESSION_MODE = "immersive-ar";',
      '/webxr-button.js':
        'export class WebXRButton { constructor(o) { globalThis.xrButton = o; this.domElement = {} } setSession() {} }',
    };
    register('data:text/javascript,' + encodeURIComponent(`
      const served = ${JSON.stringify(served)};
      export async function resolve(specifier, context, next) {
        if (specifier in served) {
          return { url: 'data:text/javascript,' + encodeURIComponent(served[specifier]), shortCircuit: true };
        }
        return next(specifier, context);
      }`));

    const sent = [];
    globalThis.WebSocket = class { static OPEN = 1; readyState = 1; send(text) { sent.push(JSON.parse(text)); } };
    const element = () => ({ appendChild() {}, classList: { add() {}, remove() {} } });
    globalThis.window = { location: { protocol: 'http:', host: 'page' }, innerWidth: 1, innerHeight: 1 };
    globalThis.document = { body: element(), createElement: element, querySelector: () => null };
    const matrix = () => ({ fromArray() {} });
    globalThis.THREE = {
      Scene: class { add() {} },
      AmbientLight: class {},
      PerspectiveCamera: class { matrix = matrix(); projectionMatrix = matrix(); updateMatrixWorld() {} },
      WebGLRenderer: class {
        xr = { setSession() {} };
        domElement = {};
        setSize() {}
        setPixelRatio() {}
        render() {}
        getContext() { return { bindFramebuffer() {}, viewport() {} }; }
      },
    };
    let nextFrame = null;
    const session = {
      inputSources: [],
      renderState: { baseLayer: { getViewport: () => ({ x: 0, y: 0, width: 1, height: 1 }) } },
      addEventListener() {},
      requestReferenceSpace: async () => ({}),
      requestAnimationFrame: callback => { nextFrame = callback; },
    };
    const xr = { isSessionSupported: async () => true, requestSession: async () => session };
    Object.defineProperty(globalThis, 'navigator', { value: { xr }, configurable: true });

    const { controllerSample, startXRApp } = await import(coreUrl);
    const { PressCounter, controllerInFrame, countPresses } = await import(new URL('press-count.js', coreUrl));
    const counters = { left: new PressCounter(), right: new PressCounter() };
    startXRApp({
      onEveryFrame: ({ session }) => countPresses(session, counters),
      onFrame: ({ session, frame, refSpace, everyFrame: presses }) => {
        const controllers = { left: null, right: null };
        for (const inputSource of session.inputSources) {
          const controller = controllerInFrame(inputSource, frame, refSpace);
          if (controller === null) continue;
          const hand = inputSource.handedness;
          controllers[hand] = controllerSample([0, 0, 0], [1, 0, 0, 0], controller.values, presses[hand]);
        }
        return controllers;
      },
    });
    await globalThis.xrButton.onRequestSession();
    await new Promise(resolve => setTimeout(resolve));

    let input = '';
    for await (const chunk of process.stdin) input += chunk;
    const identity = { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } };
    const view = { transform: { matrix: [] }, projectionMatrix: [] };
    const read = JSON.parse(input).map(([pressed, pose]) => {
      const gamepad = { buttons: pressed.map(down => ({ pressed: down, value: down ? 1 : 0 })) };
      session.inputSources = [{ gamepad, gripSpace: {}, handedness: 'right', profiles: [] }];
      const before = sent.length;
      nextFrame(0, {
        session,
        getViewerPose: () => (pose === 'no_viewer_pose' ? null : { views: [view] }),
        getPose: () => (pose === 'pose' ? { transform: identity } : null),
      });
      const right = sent.length > before ? sent.at(-1).controllers.right : null;
      return right === null ? null : right.presses;
    });
    console.log(JSON.stringify(read));
    """
    return _run_page_module('core.js', script, frames)


def test_the_teleop_command_refuses_a_stale_limit_of_the_headset_above_its_silence_limit():
    # A message that comes later than the silence limit finds control off, so its presses must do nothing
    command_with = command.serve.override(silence_limit=0.1, open_window=False, notice=None)
    with pytest.raises(ValueError, match='stale_limit of the headset must not exceed silence_limit'):
        command_with.instantiate()


@needs_node
def test_the_headset_page_sends_a_press_from_a_frame_with_no_pose_with_the_next_pose():
    # A press while the controller has no pose, and one while the headset has none, each released with a pose
    frames = [
        ([False], 'pose'),
        ([True], 'no_grip_pose'),
        ([False], 'pose'),
        ([True], 'no_viewer_pose'),
        ([False], 'pose'),
    ]
    assert _presses_in_frames(frames) == [[0], None, [1], None, [2]]


def _quest_grip(raw_values: list[float]) -> list[float]:
    return [buttons[0] for buttons in _bend_quest_trigger([{'buttons': [v], 'profiles': QUEST_3} for v in raw_values])]


@needs_node
@pytest.mark.parametrize(('raw_value', 'travel'), [(0.114, 0.35), (0.255, 0.5), (0.595, 0.75), (0.766, 0.85)])
def test_the_headset_page_sends_a_quest_trigger_press_linear_in_the_travel_past_the_dead_zone(raw_value, travel):
    assert _quest_grip([raw_value]) == [pytest.approx((travel - DEAD_ZONE) / (1 - DEAD_ZONE), abs=0.04)]


@needs_node
def test_the_headset_page_sends_a_released_and_a_full_quest_trigger_as_is_and_clamps_values_out_of_range():
    assert _quest_grip([0.0, 1.0, -0.1, 1.2]) == [0.0, 1.0, 0.0, 1.0]


@needs_node
def test_the_headset_page_sends_a_deeper_quest_trigger_as_a_closer_grip():
    grips = _quest_grip([i / 20 for i in range(21)])
    assert all(a < b for a, b in zip(grips, grips[1:], strict=False))


@needs_node
@pytest.mark.parametrize(
    ('case', 'bent'),
    [
        ({'profiles': QUEST_3}, True),
        ({'profiles': ['oculus-touch-v3', 'oculus-touch', 'generic-trigger-squeeze-thumbstick']}, True),
        ({'profiles': ['oculus-touch', 'generic-trigger-squeeze']}, False),
        ({'profiles': ['hp-mixed-reality', 'oculus-touch', 'generic-trigger-squeeze-thumbstick']}, False),
        ({'profiles': ['valve-index']}, False),
        ({'profiles': []}, False),
        ({}, False),
    ],
)
def test_the_headset_page_bends_only_the_trigger_of_a_quest_controller(case, bent):
    [buttons] = _bend_quest_trigger([{'buttons': [0.255, 0.255, 0.5], **case}])
    trigger = pytest.approx((0.5 - DEAD_ZONE) / (1 - DEAD_ZONE), abs=0.04) if bent else 0.255
    assert buttons == [trigger, 0.255, 0.5]


def test_the_headset_bridge_sends_the_phase_and_the_control_for_the_phone_page(socket_path):
    service = TeleopService(ServedUnixSocket(socket_path), 1.0, 0.05, HOLD_SEC)
    bridge = command._Headset(service)
    sent, views = RecordingEmitter[dict](), RecordingEmitter[np.ndarray]()
    bridge.page_state._bind(sent, clock=SystemClock())
    bridge.view._bind(views, clock=SystemClock())
    bridge.controller_positions._bind(ManualCommandReceiver())
    bridge.buttons._bind(ManualCommandReceiver())
    stop = types.SimpleNamespace(value=False)
    next(bridge.run(cast(pimm.SignalReceiver, stop), SystemClock()))
    [(_, state)] = sent.emitted
    assert state == {
        'phase': Phase.IDLE,
        'control': False,
        'cameras': (),
        'live': (),
        'shown': None,
        'chosen': None,
        'tracked': False,
    }


def test_the_headset_bridge_sends_whether_the_service_gets_controller_data(socket_path):
    service = TeleopService(ServedUnixSocket(socket_path), 1.0, 60.0, HOLD_SEC)
    bridge = command._Headset(service)
    sent = RecordingEmitter[dict]()
    bridge.page_state._bind(sent, clock=SystemClock())
    bridge.view._bind(RecordingEmitter[np.ndarray](), clock=SystemClock())
    positions, buttons = ManualCommandReceiver(), ManualCommandReceiver()
    bridge.controller_positions._bind(positions)
    bridge.buttons._bind(buttons)
    bridge.page_input._bind(ManualCommandReceiver())
    now = [0.0]
    clock = cast(pimm.Clock, types.SimpleNamespace(now=lambda: now[0]))
    loop = bridge.run(cast(pimm.SignalReceiver, types.SimpleNamespace(value=False)), clock)
    next(loop)
    positions.push({webxr.RIGHT: HAND})
    buttons.push({webxr.RIGHT: _buttons()})
    now[0] += 1.0
    next(loop)
    assert [state['tracked'] for _, state in sent.emitted] == [False, True]


def _until_served(url: str, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while True:
        try:
            httpx.get(url).raise_for_status()
            return
        except httpx.TransportError:
            assert time.monotonic() < deadline, f'nothing serves {url}'
            time.sleep(0.05)


@pytest.mark.timeout(60)
def test_the_scripted_headset_readies_moves_and_completes_each_trial_through_the_page(socket_path, runtime, caplog):
    caplog.set_level(logging.INFO, logger=scripted_headset.__name__)
    port = free_port()
    page = f'http://127.0.0.1:{port}'
    webxr = WebXR(port, use_https=False)
    # A loaded machine starts the page and the scripted headset slowly, so the ready limit is long. The samples pass
    # through the process of the page, so the silence limit is the one of `serve`.
    service = TeleopService(ServedUnixSocket(socket_path), ready_limit=30.0, silence_limit=0.5, hold_sec=HOLD_SEC)
    headset = command._Headset(service)
    with service, pimm.World() as world, ThreadPoolExecutor(max_workers=3) as pool:
        world.connect(webxr.controller_positions, headset.controller_positions)
        world.connect(webxr.buttons, headset.buttons)
        world.connect(headset.page_state, webxr.page_state)
        # As in `serve`, the page runs in a process of its own. Its start blocks for a while on some machines.
        loops = pool.submit(world.run, [headset], [webxr])
        try:
            _until_served(page + STATE_PATH)
            # The hand moves for 2 s, so a loaded machine still turns control on and plays a move in each trial.
            trials = pool.submit(scripted_headset.drive, page=page, trials=2, trial_sec=2.0, circle_sec=2.0)
            ends, moves = [], []
            for _ in range(2):
                started = pool.submit(runtime.start, _policy(socket_path, operator_position='BACK'))
                # A failure of the scripted headset is the cause of a run that never gets Ready, so it raises first.
                wait([trials, started], return_when=FIRST_COMPLETED)
                if trials.done():
                    trials.result()
                run = started.result()
                steps, end = _drive(run, 10.0)
                run.close()
                ends.append(end)
                moves.append(np.array([_target(step).translation for step in _moves(steps)]))
            trials.result(timeout=10)
        finally:
            world.request_stop()
            loops.result(timeout=10)
    assert ends == [PolicyEnd.COMPLETE, PolicyEnd.COMPLETE]
    for trial, targets in enumerate(moves):
        assert np.ptp(targets, axis=0).max() > 0.01, f'trial {trial} played {len(targets)} moves with no motion'


class _RecordingNotice:
    def __init__(self):
        self.calls: list[str] = []

    def show(self, text: command.NoticeText) -> None:
        self.calls.append(f'show {text.message}')

    def hide(self) -> None:
        self.calls.append('hide')


NOTICE_TEXTS = {
    Phase.HEADS_UP: command.NoticeText('Teleoperation run', 'A teleop episode comes'),
    Phase.READY_WAIT: command.NoticeText('Teleoperation trial', 'Press Ready'),
}


def _phase_notice_after(phases: list[Phase]) -> tuple[_RecordingNotice, MutableShouldStop, Generator]:
    """A phase notice that has read ``phases``, its stop flag and its loop."""
    notice, polled = _RecordingNotice(), iter(phases)
    phase_notice = command._PhaseNotice(notice, NOTICE_TEXTS, lambda: TrialState(next(polled), '', False, (), False, 0))
    stop = MutableShouldStop()
    loop = cast(Generator, phase_notice.run(cast(pimm.SignalReceiver, stop), SystemClock()))
    for _ in phases:
        next(loop)
    return notice, stop, loop


def test_the_notice_shows_while_a_trial_waits_for_ready():
    phases = [Phase.IDLE, Phase.READY_WAIT, Phase.READY_WAIT, Phase.RUNNING, Phase.ENDED, Phase.IDLE, Phase.READY_WAIT]
    notice, _, _ = _phase_notice_after([*phases, Phase.IDLE])
    assert notice.calls == ['show Press Ready', 'hide', 'show Press Ready', 'hide']


def test_the_notice_shows_the_heads_up_of_a_run_until_its_trial_waits_for_ready():
    notice, _, _ = _phase_notice_after([Phase.IDLE, Phase.HEADS_UP, Phase.HEADS_UP, Phase.READY_WAIT, Phase.RUNNING])
    assert notice.calls == ['show A teleop episode comes', 'hide', 'show Press Ready', 'hide']


def _service_notice(service: TeleopService) -> tuple[_RecordingNotice, Callable[[], None]]:
    """A phase notice of ``service``, and a function that runs one poll of its loop."""
    notice = _RecordingNotice()
    phase_notice = command._PhaseNotice(notice, NOTICE_TEXTS, service.state)
    loop = cast(Generator, phase_notice.run(cast(pimm.SignalReceiver, MutableShouldStop()), SystemClock()))
    return notice, lambda: next(loop)


def test_each_warm_up_shows_the_notice_again(service):
    notice, poll = _service_notice(service)
    service.start_heads_up()
    poll()
    service.start_heads_up()
    poll()
    assert notice.calls == ['show A teleop episode comes', 'hide', 'show A teleop episode comes']


def test_a_heads_up_that_stays_shows_the_notice_once(service):
    notice, poll = _service_notice(service)
    service.start_heads_up()
    for _ in range(3):
        poll()
    assert notice.calls == ['show A teleop episode comes']


def test_the_notice_hides_the_heads_up_when_it_expires():
    notice, _, _ = _phase_notice_after([Phase.HEADS_UP, Phase.IDLE])
    assert notice.calls == ['show A teleop episode comes', 'hide']


def test_a_service_that_stops_in_the_ready_wait_hides_the_notice():
    notice, stop, loop = _phase_notice_after([Phase.READY_WAIT])
    stop.set(True)
    next(loop, None)
    assert notice.calls == ['show Press Ready', 'hide']


def test_a_world_that_closes_the_loop_in_the_ready_wait_hides_the_notice():
    notice, _, loop = _phase_notice_after([Phase.READY_WAIT])
    loop.close()
    assert notice.calls == ['show Press Ready', 'hide']


MACOS_DIALOG = ['osascript', '-e', 'display alert "Teleoperation trial" message "Press \\"Ready\\""']
LINUX_DIALOG = ['zenity', '--info', '--no-markup', '--width=700', '--title=Teleoperation trial', '--text=Press "Ready"']
LINUX_SOUND = ['paplay', '/sounds/ready.wav']


class _Process:
    def __init__(self, command: list[str], started: list[list[str]]):
        started.append(command)
        self.command = command
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True


@pytest.mark.parametrize(
    'platform, commands',
    [('darwin', [['afplay', '/sounds/ready.wav'], MACOS_DIALOG]), ('linux', [LINUX_DIALOG, LINUX_SOUND])],
    ids=['macos', 'linux'],
)
def test_the_desktop_notice_runs_the_notice_commands_of_the_platform(monkeypatch, platform, commands):
    started: list[list[str]] = []
    monkeypatch.setattr(command.sys, 'platform', platform)
    monkeypatch.setattr(command.subprocess, 'Popen', lambda argv, env: _Process(argv, started))
    command.DesktopNotice(sound='/sounds/ready.wav').show(command.NoticeText('Teleoperation trial', 'Press "Ready"'))
    assert started == commands


@pytest.mark.parametrize('platform, dialog', [('darwin', 'osascript'), ('linux', 'zenity')], ids=['macos', 'linux'])
def test_hide_closes_the_dialog(monkeypatch, platform, dialog):
    processes: list[_Process] = []

    def start(argv: list[str], env: dict[str, str] | None) -> _Process:
        processes.append(_Process(argv, []))
        return processes[-1]

    monkeypatch.setattr(command.sys, 'platform', platform)
    monkeypatch.setattr(command.subprocess, 'Popen', start)
    notice = command.DesktopNotice()
    notice.show(NOTICE_TEXTS[Phase.READY_WAIT])
    notice.hide()
    assert [process.command[0] for process in processes if process.terminated] == [dialog]


def test_the_linux_dialog_starts_with_no_session_bus_and_no_accessibility_bus(monkeypatch):
    envs: dict[str, dict[str, str] | None] = {}

    def start(argv: list[str], env: dict[str, str] | None) -> _Process:
        envs[argv[0]] = env
        return _Process(argv, [])

    monkeypatch.setattr(command.sys, 'platform', 'linux')
    monkeypatch.setattr(command.subprocess, 'Popen', start)
    command.DesktopNotice().show(NOTICE_TEXTS[Phase.READY_WAIT])
    zenity_env = envs['zenity']
    assert zenity_env is not None
    assert (zenity_env['DBUS_SESSION_BUS_ADDRESS'], zenity_env['NO_AT_BRIDGE']) == ('disabled:', '1')
    assert envs['paplay'] is None


def test_a_notice_program_that_is_missing_is_logged_and_the_next_one_runs(monkeypatch, caplog):
    started: list[list[str]] = []

    def start(argv: list[str], env: dict[str, str] | None) -> _Process:
        if argv[0] == 'zenity':
            raise FileNotFoundError(argv[0])
        return _Process(argv, started)

    monkeypatch.setattr(command.sys, 'platform', 'linux')
    monkeypatch.setattr(command.subprocess, 'Popen', start)
    command.DesktopNotice().show(NOTICE_TEXTS[Phase.READY_WAIT])
    assert [argv[0] for argv in started] == ['paplay']
    assert 'Cannot show the notice with zenity' in caplog.text


@pytest.mark.parametrize('ending', ['give_up', 'time_limit'])
def test_every_end_of_a_session_leaves_no_trial_in_the_service(service, socket_path, runtime, ending):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, cameras=[keys.EXTERIOR_IMAGE])
        # The run makes two moves: the service keeps a trial whose session ends after one frame.
        _drive_until(run, _is_move)
        _drive_until(run, _is_move)
        assert service.state().task == OBS[keys.TASK] and service.state().cameras == (keys.EXTERIOR_IMAGE,)
        if ending == 'give_up':
            headset.buttons = _buttons(stick=1.0)
            assert _drive(run, 2.0)[1] is PolicyEnd.GIVE_UP
        run.close()
        _until_cleared(service)


def test_a_dropped_connection_in_the_ready_wait_frees_the_service_for_the_next_session(service, socket_path, connect):
    websocket = _raw_connection(socket_path)
    _until_phase(service, Phase.READY_WAIT)
    _drop(websocket)
    _until_cleared(service)
    _readied_connection(service, socket_path, connect)
    assert service.phase is Phase.RUNNING


def test_a_dropped_connection_after_frames_leaves_no_trial_in_the_service(service, socket_path):
    websocket, session_id = _raw_session(service, socket_path)
    _frames(websocket, session_id, count=2)
    assert service.state().task == OBS[keys.TASK]
    _drop(websocket)
    _until_cleared(service)


def test_a_session_that_drops_before_its_second_frame_resumes_with_no_second_ready(service, socket_path, connect):
    websocket, session_id = _raw_session(service, socket_path)
    _frames(websocket, session_id, count=1)
    _drop(websocket)
    connection = connect(socket_path)
    connection.wait_open()
    assert service.phase is Phase.RUNNING
    np.testing.assert_allclose(connection.frame(FRAME_REQUEST)[ACTION], [*OBS[keys.EE_POSE], OBS[keys.GRIP]])


def test_a_resumed_trial_keeps_its_clutch(service, socket_path, connect):
    with _Headset(service) as headset:
        websocket = _raw_connection(socket_path)
        _until_phase(service, Phase.READY_WAIT)
        headset.press(b=1.0)
        while (message := deserialise(websocket.recv(5)))[STATUS] != ServerStatus.READY:
            pass
        headset.press(a=1.0)
        _frames(websocket, message[SESSION_ID], count=1)
        assert service.state().control
        _drop(websocket)
        connection = connect(socket_path)
        connection.wait_open()
        assert service.state().control
        headset.press(a=1.0)
        connection.frame(FRAME_REQUEST)
        assert not service.state().control


def test_a_dropped_trial_that_no_session_resumes_in_time_is_forgotten(service, socket_path, monkeypatch):
    monkeypatch.setattr(TeleopService, 'RESUME_LIMIT_SEC', 0.2)
    websocket, session_id = _raw_session(service, socket_path)
    _frames(websocket, session_id, count=1)
    _drop(websocket)
    _until_cleared(service)


@pytest.mark.parametrize('ending', ['close', 'drop'])
def test_a_session_that_ends_before_its_first_frame_ends_its_trial_at_once(service, socket_path, connect, ending):
    if ending == 'close':
        _readied_connection(service, socket_path, connect).session.close()
    else:
        websocket, _ = _raw_session(service, socket_path)
        _drop(websocket)
    _until_cleared(service)
    connection = connect(socket_path)
    _until_phase(service, Phase.READY_WAIT)
    assert not connection.opening.done(), 'the next session waits for Ready'
    service.cancel_ready_wait()


def _warm_up(service: TeleopService, socket_path: Path, connect) -> None:
    """Open and close a warm-up session."""
    connection = connect(socket_path, session_query(warm_up=True))
    connection.wait_open()
    assert service.phase is Phase.HEADS_UP
    connection.session.close()


def test_a_warm_up_session_opens_with_no_ready_and_leaves_the_heads_up_of_the_run(service, socket_path, connect):
    _warm_up(service, socket_path, connect)
    assert service.phase is Phase.HEADS_UP
    assert service.view_text().lines == ('Waiting for a trial',), 'the headset view shows no heads-up'


def test_the_heads_up_stays_until_the_trial_of_the_run_waits_for_ready(service, socket_path, connect):
    _warm_up(service, socket_path, connect)
    _take(service, _buttons())
    _take(service, _buttons(b=1.0))
    _take(service, _buttons())
    assert service.phase is Phase.HEADS_UP, 'a press of B in the heads-up readies nothing'
    connection = connect(socket_path)
    _until_phase(service, Phase.READY_WAIT)
    assert not connection.opening.done(), 'the trial of the run waits for Ready'
    _take(service, _buttons(b=1.0))
    _take(service, _buttons())
    connection.wait_open()
    assert service.phase is Phase.RUNNING


def test_the_heads_up_goes_after_its_time_limit(socket_path, connect):
    with TeleopService(ServedUnixSocket(socket_path), 2.0, 0.1, HOLD_SEC, heads_up_limit=0.2) as service:
        _warm_up(service, socket_path, connect)
        _until_cleared(service)


def test_the_heads_up_logs_its_start_and_its_expiry_after_the_lock(service, socket_path, connect, lock_probe):
    service.heads_up_limit = 0.2
    _warm_up(service, socket_path, connect)
    _until_cleared(service)
    changes = [(message, free) for message, free in lock_probe.lines if Phase.HEADS_UP in message]
    assert changes == [
        ('Teleop trial: idle -> heads_up (a warm-up session starts)', True),
        ('Teleop trial: heads_up -> idle (the heads-up expired)', True),
    ]


def test_a_warm_up_session_does_not_resume_a_dropped_trial(service, socket_path, connect):
    websocket, session_id = _raw_session(service, socket_path)
    _frames(websocket, session_id, count=1)
    _drop(websocket)
    _warm_up(service, socket_path, connect)
    connection = connect(socket_path)
    _until_phase(service, Phase.READY_WAIT)
    assert not connection.opening.done(), 'the dropped trial went with the heads-up, so the session waits for Ready'
    service.cancel_ready_wait()


def test_the_next_session_of_a_run_starts_a_new_trial_after_the_last_one_ended(service, socket_path, runtime):
    with _Headset(service) as headset:
        first = _readied_run(service, headset, runtime, socket_path)
        _drive(first, 0.2)
        headset.buttons = _buttons(stick=1.0)
        assert _drive(first, 2.0)[1] is PolicyEnd.GIVE_UP
        first.close()
        headset.buttons = _buttons()
        _until_cleared(service)
        second = _readied_run(service, headset, runtime, socket_path)
        assert service.phase is Phase.RUNNING
        # Two moves: the service keeps a trial whose session ends after one frame.
        _drive_until(second, _is_move)
        _drive_until(second, _is_move)
        second.close()
        _until_cleared(service)


def test_a_symlink_in_place_of_the_socket_directory_is_refused(tmp_path):
    own = tmp_path / 'own'
    own.mkdir(mode=0o700)
    link = tmp_path / 'link'
    link.symlink_to(own)
    with pytest.raises(PermissionError, match='not a link'):
        with TeleopService(ServedUnixSocket(link / 'teleop.sock'), 1.0, 0.1, HOLD_SEC):
            pass
