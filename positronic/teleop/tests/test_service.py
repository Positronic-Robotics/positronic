import json
import os
import re
import socket
import threading
import time
import types
from collections.abc import Callable, Generator
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import pos3
import pytest
from positronic_model_server import keys as server_keys
from positronic_model_server.protocol import AUTH_HEADER, OBSERVATION, RESULT, SESSION_ID, STATUS, ServerStatus, bearer
from positronic_model_server.serialization import deserialise, encode_jpeg, serialise
from positronic_model_server.server_wire import ServedHostPort
from positronic_model_server.websocket_wire import ServedUnixSocket
from positronic_wire import registry
from positronic_wire.wire import SESSION_PATH, HostPortAddress, UnixSocketAddress
from websockets.sync.client import unix_connect

import pimm
import positronic.cfg.simulator
from pimm.time import EMITTED_WORLD
from pimm.world import SystemClock
from positronic import geom, keys, telemetry, telemetry_keys
from positronic.cfg.eval.sim.positronic import stack_cubes
from positronic.cli.eval.run import main
from positronic.dataset.local_dataset import LocalDataset
from positronic.eval import keys as eval_keys
from positronic.offboard.client import InferenceClient
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd, Step
from positronic.policy.codecs import EndCommand
from positronic.policy.codecs.metadata import ACTION
from positronic.policy.executor import Executor
from positronic.policy.keys import END
from positronic.policy.processors import ChunkedSchedule, EndOnField
from positronic.policy.remote import RemotePolicy
from positronic.teleop import command, protocol
from positronic.teleop import service as service_module
from positronic.teleop.protocol import VIEW_SIZE, session_query
from positronic.teleop.service import ButtonMap, Event, Phase, ReadyAnswer, TeleopModel, TeleopService
from positronic.tests.testing_coutils import ManualCommandReceiver, MutableShouldStop, RecordingEmitter
from positronic.utils import package_assets_path

HAND = geom.Transform3D(np.array([0.1, 0.2, 1.0]), geom.Rotation.identity)
ROBOT_POSE = geom.Transform3D(np.array([0.4, 0.0, 0.3]), geom.Rotation.from_quat([0.0, 1.0, 0.0, 0.0]))
OBS = {
    keys.EE_POSE: ROBOT_POSE.as_vector(geom.Rotation.Representation.QUAT),
    keys.GRIP: 0.0,
    keys.TASK: 'pick up the cube',
    keys.EXTERIOR_IMAGE: np.full((48, 64, 3), 128, dtype=np.uint8),
}
FRAME_REQUEST = {protocol.TASK: OBS[keys.TASK], protocol.EE_POSE: OBS[keys.EE_POSE], protocol.GRIP: [OBS[keys.GRIP]]}
HOLD_SEC = 0.2


def _buttons(trigger=0.0, stick=0.0, a=0.0, b=0.0) -> list[float]:
    """The values that the headset page sends for a controller: trigger, grip, touchpad, stick, A, B."""
    return [trigger, 0.0, 0.0, stick, a, b]


class TestButtonMap:
    def test_b_and_the_stick_answer_the_wait_for_ready(self):
        assert ButtonMap(HOLD_SEC).ready_answer(_buttons(b=1.0), 0) is ReadyAnswer.READY
        assert ButtonMap(HOLD_SEC).ready_answer(_buttons(stick=1.0), 0) is ReadyAnswer.CANCELLED
        assert ButtonMap(HOLD_SEC).ready_answer(_buttons(), 0) is None

    @pytest.mark.parametrize('threshold', [0.0, -0.1, float('nan'), 1.5])
    def test_a_press_threshold_outside_zero_to_one_is_refused(self, threshold):
        with pytest.raises(ValueError, match='press_threshold must be in'):
            ButtonMap(HOLD_SEC, threshold)

    @pytest.mark.parametrize('hold_sec', [0.0, float('nan'), float('inf')])
    def test_a_hold_time_that_is_not_finite_and_positive_is_refused(self, hold_sec):
        with pytest.raises(ValueError, match='hold_sec must be finite and positive'):
            ButtonMap(hold_sec)

    def test_a_turns_control_on_or_off_at_each_press(self):
        buttons = ButtonMap(HOLD_SEC)
        assert buttons.events(_buttons(a=1.0), 0) == [Event.CLUTCH]
        assert buttons.events(_buttons(a=1.0), 10) == []
        assert buttons.events(_buttons(), 20) == []
        assert buttons.events(_buttons(a=1.0), 30) == [Event.CLUTCH]

    def test_a_hold_of_b_completes_once(self):
        buttons = ButtonMap(HOLD_SEC)
        assert buttons.events(_buttons(b=1.0), 0) == []
        assert buttons.events(_buttons(b=1.0), 300_000_000) == [Event.COMPLETE]
        assert buttons.events(_buttons(b=1.0), 600_000_000) == []

    def test_the_b_press_of_ready_does_not_complete(self):
        buttons = ButtonMap(HOLD_SEC)
        buttons.ready_answer(_buttons(b=1.0), 0)
        assert buttons.events(_buttons(b=1.0), 1_000_000_000) == []

    def test_a_hold_of_the_stick_gives_up_once(self):
        buttons = ButtonMap(HOLD_SEC)
        assert buttons.events(_buttons(stick=1.0), 0) == []
        assert buttons.events(_buttons(stick=1.0), 300_000_000) == [Event.GIVE_UP]
        assert buttons.events(_buttons(stick=1.0), 600_000_000) == []

    def test_a_value_below_the_press_threshold_presses_nothing(self):
        buttons = ButtonMap(HOLD_SEC, press_threshold=0.9)
        assert buttons.events(_buttons(a=0.8), 0) == []
        assert buttons.events(_buttons(a=0.95), 10) == [Event.CLUTCH]

    def test_the_indices_match_the_buttons_that_the_phone_page_names(self):
        page = Path(package_assets_path('assets/webxr_iphone/index.html')).read_text()
        [layout] = re.findall(r'const BUTTON = \{ trigger: (\d+), stick: (\d+), a: (\d+), b: (\d+) \};', page)
        assert tuple(map(int, layout)) == (
            service_module._TRIGGER,
            service_module._STICK,
            service_module._A,
            service_module._B,
        )

    def test_the_trigger_is_the_grip(self):
        assert ButtonMap.grip(_buttons(trigger=0.7)) == 0.7


@pytest.fixture
def service(socket_path):
    with TeleopService(ServedUnixSocket(socket_path), ready_limit=2.0, silence_limit=0.1, hold_sec=HOLD_SEC) as served:
        yield served


@pytest.fixture
def runtime():
    executor = Executor(time.monotonic_ns, simulated=False, charge_inference_time=False)
    yield executor
    executor.close()


def _policy(socket_path: Path, **params) -> RemotePolicy:
    """An ordinary remote policy that dials the service, with ``params`` in the query of its address."""
    address = UnixSocketAddress(socket_path, SESSION_PATH, session_query(**params))
    return RemotePolicy('websocket_unix', address, infer_timeout=10.0)


def _client(socket_path: Path, query: str = '') -> InferenceClient:
    address = UnixSocketAddress(socket_path, SESSION_PATH, query)
    return InferenceClient(registry.client_wire('websocket_unix'), address, infer_timeout=5, connect_deadline=5)


class _Headset:
    """A headset that sends one sample of the right controller every 10 ms, with the buttons that a test sets."""

    def __init__(self, service: TeleopService):
        self._service = service
        self.pose: geom.Transform3D | None = HAND
        self.buttons = _buttons()
        self.sent = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._send, daemon=True)

    def _send(self) -> None:
        while not self._stop.is_set():
            self._service.take_sample(self.pose, self.buttons)
            self.sent += 1
            time.sleep(0.01)

    def __enter__(self) -> '_Headset':
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _hold(self, buttons: list[float], samples: int) -> None:
        """Send ``buttons`` in at least ``samples`` samples, so a loaded machine does not miss them."""
        self.buttons = buttons
        until = self.sent + samples
        deadline = time.monotonic() + 5
        while self.sent < until:
            assert time.monotonic() < deadline, 'the headset sent no sample'
            time.sleep(0.005)

    def press(self, **buttons: float) -> None:
        """Press ``buttons`` for a few samples, and release them."""
        self._hold(_buttons(**buttons), samples=3)
        self._hold(_buttons(), samples=3)


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


def _drive(run, seconds: float) -> tuple[list[Step], PolicyEnd | None]:
    """Send ``OBS`` to the run for ``seconds``. Return its steps, and its end if it ended."""
    steps = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            step = run.send(OBS)
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


def _drive_until(run, condition: Callable[[Step], bool], seconds: float = 5.0) -> Step:
    """Send ``OBS`` to the run until it makes a step for which ``condition`` holds, and return that step."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        step = run.send(OBS)
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


def test_with_no_operator_position_the_hand_pose_is_the_robot_pose_and_control_is_on(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, operator_position=None)
        moved = _drive_until(run, _is_move)
        np.testing.assert_allclose(_target(moved).translation, HAND.translation, atol=1e-6)
        run.close()


def test_the_service_shows_the_camera_image_of_the_run(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, cameras=[keys.EXTERIOR_IMAGE])
        _drive(run, 0.2)
        view = service.view()
        width, height = VIEW_SIZE
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
    with pytest.raises(ValueError, match='jpeg_quality must be an integer between 0 and 100'):
        session_query(jpeg_quality=90.5)
    assert session_query(operator_position=None, cameras=[keys.WRIST_IMAGE]) == (
        'operator_position=null&cameras=%5B%22image.wrist%22%5D'
    )


def test_session_params_refuse_an_operator_position_that_is_no_operator_position():
    with pytest.raises(ValueError, match="operator_position must be one of .*, got '.BACK'"):
        protocol.SessionParams('.BACK')  # pyright: ignore[reportArgumentType]
    assert protocol.SessionParams(None).operator() is None


def test_the_client_stack_ends_the_run_outside_a_schedule_that_asks_for_one_frame_at_a_time():
    stack = TeleopModel.client_stack(protocol.SessionParams(cameras=(keys.WRIST_IMAGE,)))
    end, schedule = stack['seq'][:2]
    assert end['name'] == EndOnField.WIRE_NAME
    assert schedule['name'] == ChunkedSchedule.WIRE_NAME
    assert schedule['args']['inference_lead'] >= 1 / schedule['args']['fps']
    assert EndCommand.WIRE_NAME in json.dumps(stack['seq'][2])


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
    assert episode.static[eval_keys.ENDED_BY] == eval_keys.ENDED_BY_POLICY
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
    assert {telemetry.component_name(ChunkedSchedule(fps=1.0)), telemetry_keys.SPAN_POLICY_INFER} <= names


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
    service.take_sample(HAND, _buttons(b=1.0))
    service.take_sample(HAND, _buttons())
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


@pytest.mark.parametrize('name', ['ready_limit', 'silence_limit'])
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


def test_a_service_on_a_host_and_a_port_needs_a_token():
    with pytest.raises(ValueError, match='needs an auth token'):
        TeleopService(ServedHostPort('127.0.0.1', 0), 1.0, 0.1, HOLD_SEC)


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
    service.take_sample(HAND, _buttons(b=1.0))
    service.take_sample(HAND, _buttons())
    connection.wait_open()
    return connection


def test_a_hold_of_b_completes_in_one_frame(service, socket_path, connect):
    connection = _readied_connection(service, socket_path, connect)
    service.take_sample(HAND, _buttons(b=1.0))
    time.sleep(HOLD_SEC + 0.05)
    service.take_sample(HAND, _buttons(b=1.0))
    ended = connection.frame(FRAME_REQUEST)
    assert ended[END] == PolicyEnd.COMPLETE.value
    np.testing.assert_allclose(ended[ACTION], [*OBS[keys.EE_POSE], OBS[keys.GRIP]])
    assert service.phase is Phase.ENDED


def test_each_frame_request_after_the_end_gets_the_same_end(service, socket_path, connect):
    connection = _readied_connection(service, socket_path, connect)
    service.take_sample(HAND, _buttons(stick=1.0))
    time.sleep(HOLD_SEC + 0.05)
    service.take_sample(HAND, _buttons(stick=1.0))
    ended = connection.frame(FRAME_REQUEST)
    again = connection.frame(FRAME_REQUEST)
    assert ended[END] == again[END] == PolicyEnd.GIVE_UP.value
    np.testing.assert_allclose(again[ACTION], ended[ACTION])


def test_each_frame_reply_has_a_target_from_the_first(service, socket_path, connect):
    connection = _readied_connection(service, socket_path, connect)
    first = connection.frame(FRAME_REQUEST)
    np.testing.assert_allclose(first[ACTION], [*OBS[keys.EE_POSE], OBS[keys.GRIP]])


def test_the_view_leaves_the_camera_image_of_the_display_unchanged(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, cameras=[keys.EXTERIOR_IMAGE])
        _drive(run, 0.2)
        stored = service.camera_image(keys.EXTERIOR_IMAGE)
        before = stored.copy()
        service.view()
        np.testing.assert_array_equal(stored, before)
        run.close()


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
        assert service.view().shape == images[shown].shape
        assert service.state().cameras == (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)
        connection.close()


@pytest.mark.parametrize('view_fps', [0.0, -1.0, float('nan'), float('inf')])
def test_a_view_rate_that_is_not_finite_and_positive_is_refused(socket_path, view_fps):
    with pytest.raises(ValueError, match='view_fps must be finite and positive'):
        command._Headset(TeleopService(ServedUnixSocket(socket_path), 1.0, 0.05, HOLD_SEC), view_fps)


def test_the_phone_page_reads_the_fields_of_the_page_state():
    page = Path(package_assets_path('assets/webxr_iphone/index.html')).read_text()
    [fields] = re.findall(r'const STATE_FIELD = \{([^}]*)\};', page)
    assert set(re.findall(r"'(\w+)'", fields)) == set(command.PageState.__dataclass_fields__)


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
    assert state == {'phase': Phase.IDLE, 'control': False}


class _RecordingNotice:
    def __init__(self):
        self.calls: list[str] = []

    def show(self, text: str) -> None:
        self.calls.append(f'show {text}')

    def hide(self) -> None:
        self.calls.append('hide')


def _ready_notice_after(phases: list[Phase]) -> tuple[_RecordingNotice, MutableShouldStop, Generator]:
    """A ready notice that has read ``phases``, its stop flag and its loop."""
    notice = _RecordingNotice()
    ready_notice = command._ReadyNotice(notice, 'Press Ready')
    states = ManualCommandReceiver[dict]()
    ready_notice.page_state._bind(states)
    stop = MutableShouldStop()
    loop = cast(Generator, ready_notice.run(cast(pimm.SignalReceiver, stop), SystemClock()))
    for phase in phases:
        states.push(asdict(command.PageState(phase, control=False)))
        next(loop)
    return notice, stop, loop


def test_the_notice_shows_while_a_trial_waits_for_ready():
    phases = [Phase.IDLE, Phase.READY_WAIT, Phase.READY_WAIT, Phase.RUNNING, Phase.ENDED, Phase.IDLE, Phase.READY_WAIT]
    notice, _, _ = _ready_notice_after([*phases, Phase.IDLE])
    assert notice.calls == ['show Press Ready', 'hide', 'show Press Ready', 'hide']


def test_a_service_that_stops_in_the_ready_wait_hides_the_notice():
    notice, stop, loop = _ready_notice_after([Phase.READY_WAIT])
    stop.set(True)
    next(loop, None)
    assert notice.calls == ['show Press Ready', 'hide']


def test_a_world_that_closes_the_loop_in_the_ready_wait_hides_the_notice():
    notice, _, loop = _ready_notice_after([Phase.READY_WAIT])
    loop.close()
    assert notice.calls == ['show Press Ready', 'hide']


MACOS_DIALOG = ['osascript', '-e', 'display alert "positronic teleop" message "Press \\"Ready\\""']
LINUX_DIALOG = ['zenity', '--info', '--no-markup', '--title=positronic teleop', '--text=Press "Ready"']
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
    monkeypatch.setattr(command.subprocess, 'Popen', lambda argv: _Process(argv, started))
    command.DesktopNotice(sound='/sounds/ready.wav').show('Press "Ready"')
    assert started == commands


@pytest.mark.parametrize('platform, dialog', [('darwin', 'osascript'), ('linux', 'zenity')], ids=['macos', 'linux'])
def test_hide_closes_the_dialog(monkeypatch, platform, dialog):
    processes: list[_Process] = []

    def start(argv: list[str]) -> _Process:
        processes.append(_Process(argv, []))
        return processes[-1]

    monkeypatch.setattr(command.sys, 'platform', platform)
    monkeypatch.setattr(command.subprocess, 'Popen', start)
    notice = command.DesktopNotice()
    notice.show('Press Ready')
    notice.hide()
    assert [process.command[0] for process in processes if process.terminated] == [dialog]


def test_a_notice_program_that_is_missing_is_logged_and_the_next_one_runs(monkeypatch, caplog):
    started: list[list[str]] = []

    def start(argv: list[str]) -> _Process:
        if argv[0] == 'zenity':
            raise FileNotFoundError(argv[0])
        return _Process(argv, started)

    monkeypatch.setattr(command.sys, 'platform', 'linux')
    monkeypatch.setattr(command.subprocess, 'Popen', start)
    command.DesktopNotice().show('Press Ready')
    assert [argv[0] for argv in started] == ['paplay']
    assert 'Cannot show the notice with zenity' in caplog.text


@pytest.mark.parametrize('ending', ['give_up', 'time_limit'])
def test_every_end_of_a_session_leaves_no_trial_in_the_service(service, socket_path, runtime, ending):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, cameras=[keys.EXTERIOR_IMAGE])
        _drive(run, 0.2)
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
    websocket, _ = _raw_session(service, socket_path)
    _drop(websocket)
    _until_cleared(service)


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
        _drive(second, 0.2)
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
