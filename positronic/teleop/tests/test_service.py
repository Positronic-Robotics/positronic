import os
import re
import socket
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pos3
import pytest
from positronic_model_server import keys as server_keys
from positronic_model_server.protocol import OBSERVATION, RESULT, SESSION_ID, STATUS, ServerStatus
from positronic_model_server.serialization import deserialise, encode_jpeg, serialise
from positronic_wire.wire import SESSION_PATH, UnixSocketAddress
from websockets.sync.client import unix_connect

import pimm
import positronic.cfg.simulator
from pimm.world import SystemClock
from positronic import geom, keys, telemetry, telemetry_keys
from positronic.cfg.eval.sim.positronic import stack_cubes
from positronic.cli.eval.run import main
from positronic.dataset.local_dataset import LocalDataset
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd, Step
from positronic.policy.executor import Executor
from positronic.policy.remote import RemotePolicy
from positronic.policy.teleop import TeleopClient, TrialNotStarted
from positronic.teleop import command, protocol
from positronic.teleop import service as service_module
from positronic.teleop.protocol import Reply, Request, session_query
from positronic.teleop.service import ButtonMap, Event, Phase, TeleopService
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter
from positronic.utils import package_assets_path

HAND = geom.Transform3D(np.array([0.1, 0.2, 1.0]), geom.Rotation.identity)
ROBOT_POSE = geom.Transform3D(np.array([0.4, 0.0, 0.3]), geom.Rotation.from_quat([0.0, 1.0, 0.0, 0.0]))
OBS = {
    keys.EE_POSE: ROBOT_POSE.as_vector(geom.Rotation.Representation.QUAT),
    keys.GRIP: 0.0,
    keys.TASK: 'pick up the cube',
    keys.EXTERIOR_IMAGE: np.full((48, 64, 3), 128, dtype=np.uint8),
}
FRAME_REQUEST = {protocol.KIND: Request.FRAME, protocol.TASK: OBS[keys.TASK], protocol.EE_POSE: OBS[keys.EE_POSE]}
HOLD_SEC = 0.2


def _buttons(trigger=0.0, stick=0.0, a=0.0, b=0.0) -> list[float]:
    """The values that the headset page sends for a controller: trigger, grip, touchpad, stick, A, B."""
    return [trigger, 0.0, 0.0, stick, a, b]


class TestButtonMap:
    def test_b_and_the_stick_answer_the_ready_request(self):
        assert ButtonMap(HOLD_SEC).ready_reply(_buttons(b=1.0), 0) is Reply.READY
        assert ButtonMap(HOLD_SEC).ready_reply(_buttons(stick=1.0), 0) is Reply.CANCELLED
        assert ButtonMap(HOLD_SEC).ready_reply(_buttons(), 0) is None

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
        assert buttons.events(_buttons(a=1.0), 0, False) == [Event.CLUTCH]
        assert buttons.events(_buttons(a=1.0), 10, False) == []
        assert buttons.events(_buttons(), 20, False) == []
        assert buttons.events(_buttons(a=1.0), 30, False) == [Event.CLUTCH]

    def test_b_starts_the_task_and_a_later_hold_of_b_completes_it(self):
        buttons = ButtonMap(HOLD_SEC)
        assert buttons.events(_buttons(b=1.0), 0, False) == [Event.START_TASK]
        assert buttons.events(_buttons(b=1.0), 1_000_000_000, True) == []
        buttons.events(_buttons(), 1_100_000_000, True)
        assert buttons.events(_buttons(b=1.0), 1_200_000_000, True) == []
        assert buttons.events(_buttons(b=1.0), 1_500_000_000, True) == [Event.COMPLETE]

    def test_no_hold_completes_before_the_task_starts(self):
        buttons = ButtonMap(HOLD_SEC)
        buttons.ready_reply(_buttons(b=1.0), 0)
        assert buttons.events(_buttons(b=1.0), 1_000_000_000, False) == []

    def test_a_hold_of_the_stick_gives_up_once(self):
        buttons = ButtonMap(HOLD_SEC)
        assert buttons.events(_buttons(stick=1.0), 0, False) == []
        assert buttons.events(_buttons(stick=1.0), 300_000_000, False) == [Event.GIVE_UP]
        assert buttons.events(_buttons(stick=1.0), 600_000_000, False) == []

    def test_a_value_below_the_press_threshold_presses_nothing(self):
        buttons = ButtonMap(HOLD_SEC, press_threshold=0.9)
        assert buttons.events(_buttons(a=0.8), 0, False) == []
        assert buttons.events(_buttons(a=0.95), 10, False) == [Event.CLUTCH]

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
    with TeleopService(socket_path, ready_limit=2.0, silence_limit=0.1, hold_sec=HOLD_SEC) as served:
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
    assert (state.task, state.control, state.attempt_started, state.cameras) == ('', False, False, ())


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
    """A run of a remote policy that the teleoperator readied, after its stay command."""
    pressing = _press_ready_when_asked(service, headset, b=1.0)
    run = runtime.start(_policy(socket_path, **params))
    stay = run.send(OBS)
    assert stay is not None and keys.ROBOT_COMMAND in stay.commands
    # The press releases B after Ready. A test press before that release would be lost in it.
    pressing.join()
    return run


def test_the_meta_of_a_remote_policy_needs_no_ready_and_names_the_teleop_service(service, socket_path):
    meta = _policy(socket_path, operator_position='BACK').meta()
    assert meta[policy_keys.TYPE] == 'remote'
    assert meta[f'{policy_keys.SERVER}.{policy_keys.TYPE}'] == 'teleop'
    assert meta[f'{policy_keys.SERVER}.{server_keys.EFFECTIVE_PARAMS}.operator_position'] == 'BACK'
    assert service.phase is Phase.IDLE


def test_a_teleoperator_readies_takes_control_starts_and_completes_a_trial(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        assert service.phase is Phase.RUNNING

        headset.press(a=1.0)
        first = _drive_until(run, _is_move)
        np.testing.assert_allclose(_target(first).translation, ROBOT_POSE.translation)
        assert service.state().control

        headset.press(b=1.0)
        _drive_until(run, lambda _: policy_keys.ATTEMPT_START_NS in runtime.metadata_snapshot())

        headset.buttons = _buttons(b=1.0)
        _, end = _drive(run, 2.0)
        assert end is PolicyEnd.COMPLETE
        run.close()
        _until_cleared(service)


def test_control_follows_the_hand_from_where_it_came_on(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        headset.press(a=1.0)
        _drive_until(run, _is_move)
        headset.pose = geom.Transform3D(HAND.translation + np.array([0.0, 0.0, 0.05]), HAND.rotation)
        moved = _drive_until(
            run,
            lambda step: (
                _is_move(step) and bool(np.linalg.norm(_target(step).translation - ROBOT_POSE.translation) > 0.04)
            ),
        )
        shift = _target(moved).translation - ROBOT_POSE.translation
        np.testing.assert_allclose(np.linalg.norm(shift), 0.05, atol=1e-9)
        headset.press(a=1.0)
        _drive_until(run, lambda _: not service.state().control)
        steps, _ = _drive(run, 0.3)
        assert not _moves(steps)
        run.close()


def test_with_no_operator_position_the_hand_pose_is_the_robot_pose_and_control_is_on(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, operator_position=None)
        moved = _drive_until(run, _is_move)
        np.testing.assert_allclose(_target(moved).translation, HAND.translation)
        run.close()


def test_a_hold_of_b_before_the_start_task_press_does_not_complete(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        _drive(run, 0.2)
        headset.buttons = _buttons(b=1.0)
        _, end = _drive(run, 1.0)
        assert end is None
        assert policy_keys.ATTEMPT_START_NS in runtime.metadata_snapshot()
        run.close()


def test_the_service_shows_the_camera_image_of_the_run(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path, cameras=[keys.EXTERIOR_IMAGE])
        _drive(run, 0.2)
        assert service.view().shape[:2] == OBS[keys.EXTERIOR_IMAGE].shape[:2]
        run.close()


def test_the_client_sends_the_cameras_that_the_session_names(service, socket_path, runtime):
    obs = {**OBS, keys.WRIST_IMAGE: np.zeros((24, 32, 3), dtype=np.uint8)}
    with _Headset(service) as headset:
        _press_ready_when_asked(service, headset, b=1.0)
        run = runtime.start(_policy(socket_path, cameras=[keys.WRIST_IMAGE]))
        run.send(obs)
        time.sleep(0.2)
        run.send(obs)
        assert service.state().cameras == (keys.WRIST_IMAGE,)
        run.close()


def test_a_cancel_fails_the_run_before_its_first_step(service, socket_path, runtime):
    with _Headset(service) as headset:
        _press_ready_when_asked(service, headset, stick=1.0)
        with pytest.raises(TrialNotStarted, match='cancelled'):
            runtime.start(_policy(socket_path))
    _until_cleared(service)


def test_no_ready_within_the_ready_limit_fails_the_run(socket_path, runtime):
    with TeleopService(socket_path, ready_limit=0.3, silence_limit=0.1, hold_sec=HOLD_SEC) as service:
        started = time.monotonic()
        with pytest.raises(TrialNotStarted, match='ready limit'):
            runtime.start(_policy(socket_path))
        assert time.monotonic() - started < 0.3 + TeleopService.READY_SLICE_SEC + 1.0
        _until_cleared(service)


def test_a_ready_wait_longer_than_the_infer_timeout_still_ends_with_ready(socket_path, runtime):
    """Each ready request ends within a slice, so the infer timeout bounds a slice, not the whole wait."""
    with TeleopService(socket_path, ready_limit=10.0, silence_limit=0.1, hold_sec=HOLD_SEC) as service:
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
        _drive_until(run, _is_move)
        headset.pose = None
        _drive_until(run, lambda _: not service.state().control)
        headset.pose = HAND
        steps, _ = _drive(run, 0.3)
        assert not _moves(steps)
        run.close()


def test_the_service_refuses_a_second_session_while_a_trial_runs(service, socket_path, runtime):
    with _Headset(service) as headset:
        first = _readied_run(service, headset, runtime, socket_path)
        with pytest.raises(RuntimeError, match='one session at a time'):
            _policy(socket_path).meta()
        first.close()


def test_the_service_refuses_a_second_session_in_the_ready_wait_long_before_the_ready_limit(socket_path, connect):
    """The worker thread takes the second preparation after the current ready slice, not after the whole wait."""
    with TeleopService(socket_path, ready_limit=60.0, silence_limit=0.1, hold_sec=HOLD_SEC) as service:
        waiting = connect(socket_path)
        try:
            answer = waiting.ready_in_background()
            _until_phase(service, Phase.READY_WAIT)
            started = time.monotonic()
            with pytest.raises(RuntimeError, match='one session at a time'):
                _policy(socket_path).meta()
            assert time.monotonic() - started < 30
            service.cancel_ready_wait()
            assert answer.result(5) is Reply.CANCELLED
        finally:
            waiting.close()


def test_a_session_with_a_wrong_operator_position_is_refused_at_connect(service, socket_path):
    address = UnixSocketAddress(socket_path, SESSION_PATH, 'operator_position="SIDE"')
    with pytest.raises(RuntimeError, match='operator_position must be one of'):
        RemotePolicy('websocket_unix', address).meta()


def test_a_session_query_refuses_a_wrong_value_or_an_unknown_parameter():
    with pytest.raises(ValueError, match='operator_position must be one of'):
        session_query(operator_position='SIDE')
    with pytest.raises(TypeError, match='fps'):
        session_query(fps=10)
    with pytest.raises(ValueError, match='image_fps must be finite and positive'):
        session_query(image_fps=0)
    with pytest.raises(ValueError, match='still_clock_timeout must be finite and positive'):
        session_query(still_clock_timeout=float('inf'))
    with pytest.raises(ValueError, match='cameras must be a list of camera names'):
        session_query(cameras=keys.WRIST_IMAGE)
    assert session_query(operator_position=None, cameras=[keys.WRIST_IMAGE]) == (
        'operator_position=null&cameras=%5B%22image.wrist%22%5D'
    )


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
        headset.press(b=1.0)
        time.sleep(0.3)
        headset.buttons = _buttons(b=1.0)

    service = TeleopService(socket_path, ready_limit=60.0, silence_limit=0.1, hold_sec=HOLD_SEC)
    with service, _Headset(service) as headset, pos3.mirror():
        threading.Thread(target=teleoperate, args=(service, headset), daemon=True).start()
        main(policy=_policy(socket_path), evals=[ev], output_dir=str(tmp_path))

    [episode] = LocalDataset(tmp_path)
    policy_meta = policy_keys.POLICY_META
    assert episode.static[eval_keys.ENDED_BY] == eval_keys.ENDED_BY_POLICY
    assert episode.static[eval_keys.POLICY_END] == PolicyEnd.COMPLETE
    assert f'{policy_meta}.{policy_keys.ATTEMPT_START_NS}' in episode.static
    assert episode.static[f'{policy_meta}.{policy_keys.TYPE}'] == 'remote'
    assert episode.static[f'{policy_meta}.{policy_keys.SERVER}.{policy_keys.TYPE}'] == 'teleop'
    assert keys.TARGET_EE_POSE in episode.signals
    names = {span.name for span in telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS))}
    assert {telemetry.component_name(TeleopClient(1.0, (), 15.0, 1.0)), telemetry_keys.SPAN_POLICY_INFER} <= names


class TestButtonsBetweenTrials:
    def test_a_button_released_after_the_trial_presses_again_in_the_next(self):
        buttons = ButtonMap(HOLD_SEC)
        buttons.events(_buttons(b=1.0), 0, True)
        assert buttons.events(_buttons(b=1.0), 500_000_000, True) == [Event.COMPLETE]
        buttons.observe(_buttons(), 600_000_000)
        assert buttons.ready_reply(_buttons(b=1.0), 700_000_000) is Reply.READY

    def test_a_button_held_from_the_last_trial_is_no_ready(self):
        buttons = ButtonMap(HOLD_SEC)
        buttons.observe(_buttons(b=1.0), 0)
        assert buttons.ready_reply(_buttons(b=1.0), 100_000_000) is None


def _raw_session(socket_path: Path):
    """A websocket on the service, after the handshake of protocol v3, and the ID of its session."""
    websocket = unix_connect(str(socket_path), f'ws://localhost{SESSION_PATH}')
    while (message := deserialise(websocket.recv(5)))[STATUS] != ServerStatus.READY:
        pass
    return websocket, message[SESSION_ID]


def _drop(websocket) -> None:
    """End the connection as a client process that exits does: with no close of the session."""
    websocket.socket.shutdown(socket.SHUT_RDWR)
    websocket.socket.close()


def test_a_stop_of_the_service_ends_its_waits(socket_path):
    service = TeleopService(socket_path, ready_limit=60.0, silence_limit=0.1, hold_sec=HOLD_SEC)
    with service:
        websocket, session_id = _raw_session(socket_path)
        websocket.send(serialise({SESSION_ID: session_id, OBSERVATION: {protocol.KIND: Request.READY}}))
        _until_phase(service, Phase.READY_WAIT)
        stopped_at = time.monotonic()
    assert time.monotonic() - stopped_at < 5
    websocket.close()


def test_a_service_that_cannot_bind_raises_the_cause_at_once(tmp_path):
    started_at = time.monotonic()
    with pytest.raises(OSError, match='too long'):
        with TeleopService(tmp_path / ('x' * 120) / 'teleop.sock', 1.0, 0.1, HOLD_SEC):
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
    with TeleopService(socket_path, 1.0, 0.1, HOLD_SEC):
        assert interrupted.wait(timeout=5)


def test_a_socket_directory_that_others_can_open_is_refused(tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(PermissionError, match='mode 700'):
        with TeleopService(shared / 'teleop.sock', 1.0, 0.1, HOLD_SEC):
            pass


def test_the_default_socket_with_no_runtime_directory_is_in_a_directory_of_the_user(monkeypatch):
    monkeypatch.delenv('XDG_RUNTIME_DIR', raising=False)
    assert protocol.default_socket_path().parent.name == f'positronic-{os.getuid()}'


def test_a_frame_request_before_ready_is_refused(service, socket_path, connect):
    connection = connect(socket_path)
    frame = {protocol.KIND: Request.FRAME, protocol.TASK: '', protocol.EE_POSE: OBS[keys.EE_POSE]}
    with pytest.raises(RuntimeError, match='a frame request came in the idle phase'):
        connection.ask(frame)


def _readied_connection(service: TeleopService, socket_path: Path, connect):
    """A test session that the teleoperator readied from the headset."""
    connection = connect(socket_path)
    answer = connection.ready_in_background()
    _until_phase(service, Phase.READY_WAIT)
    service.take_sample(HAND, _buttons(b=1.0))
    service.take_sample(HAND, _buttons())
    assert answer.result(5) is Reply.READY
    return connection


def test_a_hold_of_b_after_start_task_completes_in_one_frame_with_the_start(service, socket_path, connect):
    connection = _readied_connection(service, socket_path, connect)
    service.take_sample(HAND, _buttons(b=1.0))
    service.take_sample(HAND, _buttons())
    service.take_sample(HAND, _buttons(b=1.0))
    time.sleep(HOLD_SEC + 0.05)
    service.take_sample(HAND, _buttons(b=1.0))
    frame = connection.ask(FRAME_REQUEST)
    assert frame[protocol.START_TASK] is True
    assert frame[protocol.END] == PolicyEnd.COMPLETE.value
    assert service.phase is Phase.ENDED


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
def test_the_headset_shows_the_first_camera_or_the_one_it_names(socket_path, connect, headset_camera):
    images = {
        keys.WRIST_IMAGE: np.full((24, 32, 3), 50, np.uint8),
        keys.EXTERIOR_IMAGE: np.full((48, 64, 3), 200, np.uint8),
    }
    with TeleopService(socket_path, 5.0, 0.05, HOLD_SEC, headset_camera=headset_camera) as service:
        connection = _readied_connection(service, socket_path, connect)
        encoded = {name: encode_jpeg(image) for name, image in images.items()}
        connection.ask({**FRAME_REQUEST, protocol.IMAGES: encoded})
        shown = keys.WRIST_IMAGE if headset_camera is None else headset_camera
        assert service.view().shape == images[shown].shape
        assert service.state().cameras == (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE)
        connection.close()


def test_the_headset_bridge_sends_the_phase_and_the_control_for_the_phone_page(socket_path):
    service = TeleopService(socket_path, 1.0, 0.05, HOLD_SEC)
    bridge = command._Headset(service)
    sent, views = RecordingEmitter[dict](), RecordingEmitter[np.ndarray]()
    bridge.page_state._bind(sent, clock=SystemClock())
    bridge.view._bind(views, clock=SystemClock())
    bridge.controller_positions._bind(ManualCommandReceiver())
    bridge.buttons._bind(ManualCommandReceiver())
    stop = types.SimpleNamespace(value=False)
    next(bridge.run(cast(pimm.SignalReceiver, stop), SystemClock()))
    [(_, state)] = sent.emitted
    assert state == {'phase': Phase.IDLE, 'control': False, 'attempt_started': False}


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


def test_a_dropped_connection_in_the_ready_wait_frees_the_service_for_the_next_session(service, socket_path):
    websocket, session_id = _raw_session(socket_path)
    websocket.send(serialise({SESSION_ID: session_id, OBSERVATION: {protocol.KIND: Request.READY}}))
    _until_phase(service, Phase.READY_WAIT)
    _drop(websocket)
    _until_cleared(service)
    assert _policy(socket_path).meta()[f'{policy_keys.SERVER}.{policy_keys.TYPE}'] == 'teleop'


def test_a_dropped_connection_mid_trial_leaves_no_trial_in_the_service(service, socket_path):
    websocket, session_id = _raw_session(socket_path)

    def ask(request: dict) -> dict:
        websocket.send(serialise({SESSION_ID: session_id, OBSERVATION: request}))
        return deserialise(websocket.recv(5))[RESULT]

    answer = threading.Thread(target=ask, args=({protocol.KIND: Request.READY},), daemon=True)
    answer.start()
    _until_phase(service, Phase.READY_WAIT)
    service.take_sample(HAND, _buttons(b=1.0))
    answer.join(5)
    ask({**FRAME_REQUEST, protocol.IMAGES: {keys.EXTERIOR_IMAGE: encode_jpeg(OBS[keys.EXTERIOR_IMAGE])}})
    assert service.state().task == OBS[keys.TASK]
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
        second.close()
        _until_cleared(service)


def test_a_symlink_in_place_of_the_socket_directory_is_refused(tmp_path):
    own = tmp_path / 'own'
    own.mkdir(mode=0o700)
    link = tmp_path / 'link'
    link.symlink_to(own)
    with pytest.raises(PermissionError, match='not a link'):
        with TeleopService(link / 'teleop.sock', 1.0, 0.1, HOLD_SEC):
            pass
