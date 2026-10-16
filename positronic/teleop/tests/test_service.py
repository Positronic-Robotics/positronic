import asyncio
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import cast

import numpy as np
import pos3
import pytest
from positronic_model_server import server_wire
from positronic_model_server.serialization import deserialise, serialise
from positronic_wire import registry
from positronic_wire.wire import SESSION_PATH, PeerDisconnected, UnixSocketAddress

import positronic.cfg.simulator
from positronic import geom, keys
from positronic.cfg.eval.sim.positronic import stack_cubes
from positronic.cli.eval.run import main
from positronic.dataset.local_dataset import LocalDataset
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd, Step
from positronic.policy.executor import Executor
from positronic.teleop import TeleopPolicy, session
from positronic.teleop.service import ButtonMap, Phase, TeleopService
from positronic.utils import package_assets_path

HAND = geom.Transform3D(np.array([0.1, 0.2, 1.0]), geom.Rotation.identity)
ROBOT_POSE = geom.Transform3D(np.array([0.4, 0.0, 0.3]), geom.Rotation.from_quat([0.0, 1.0, 0.0, 0.0]))
OBS = {
    keys.EE_POSE: ROBOT_POSE.as_vector(geom.Rotation.Representation.QUAT),
    keys.GRIP: 0.0,
    keys.TASK: 'pick up the cube',
    keys.EXTERIOR_IMAGE: np.full((48, 64, 3), 128, dtype=np.uint8),
}
HOLD_SEC = 0.2


def _buttons(trigger=0.0, stick=0.0, a=0.0, b=0.0) -> list[float]:
    """The values that the headset page sends for a controller: trigger, grip, touchpad, stick, A, B."""
    return [trigger, 0.0, 0.0, stick, a, b]


class TestButtonMap:
    def test_b_and_the_stick_answer_the_ready_request(self):
        assert ButtonMap(HOLD_SEC).ready_reply(_buttons(b=1.0), 0) is session.Reply.READY
        assert ButtonMap(HOLD_SEC).ready_reply(_buttons(stick=1.0), 0) is session.Reply.CANCELLED
        assert ButtonMap(HOLD_SEC).ready_reply(_buttons(), 0) is None

    def test_a_turns_control_on_or_off_at_each_press(self):
        buttons = ButtonMap(HOLD_SEC)
        assert buttons.events(_buttons(a=1.0), 0, False) == [session.Event.CLUTCH]
        assert buttons.events(_buttons(a=1.0), 10, False) == []
        assert buttons.events(_buttons(), 20, False) == []
        assert buttons.events(_buttons(a=1.0), 30, False) == [session.Event.CLUTCH]

    def test_b_starts_the_task_and_a_later_hold_of_b_completes_it(self):
        buttons = ButtonMap(HOLD_SEC)
        assert buttons.events(_buttons(b=1.0), 0, False) == [session.Event.START_TASK]
        assert buttons.events(_buttons(b=1.0), 1_000_000_000, True) == []
        buttons.events(_buttons(), 1_100_000_000, True)
        assert buttons.events(_buttons(b=1.0), 1_200_000_000, True) == []
        assert buttons.events(_buttons(b=1.0), 1_500_000_000, True) == [session.Event.COMPLETE]

    def test_no_hold_completes_before_the_task_starts(self):
        buttons = ButtonMap(HOLD_SEC)
        buttons.ready_reply(_buttons(b=1.0), 0)
        assert buttons.events(_buttons(b=1.0), 1_000_000_000, False) == []

    def test_a_hold_of_the_stick_gives_up_once(self):
        buttons = ButtonMap(HOLD_SEC)
        assert buttons.events(_buttons(stick=1.0), 0, False) == []
        assert buttons.events(_buttons(stick=1.0), 300_000_000, False) == [session.Event.GIVE_UP]
        assert buttons.events(_buttons(stick=1.0), 600_000_000, False) == []

    def test_a_value_below_the_press_threshold_presses_nothing(self):
        buttons = ButtonMap(HOLD_SEC, press_threshold=0.9)
        assert buttons.events(_buttons(a=0.8), 0, False) == []
        assert buttons.events(_buttons(a=0.95), 10, False) == [session.Event.CLUTCH]

    def test_the_trigger_is_the_grip(self):
        assert ButtonMap.grip(_buttons(trigger=0.7)) == 0.7


@pytest.fixture
def socket_path():
    # A Unix socket path has a short length limit, so it does not go under the long pytest directory.
    with tempfile.TemporaryDirectory(prefix='teleop-') as directory:
        yield Path(directory) / 'teleop.sock'


@pytest.fixture
def service(socket_path):
    with TeleopService(socket_path, ready_limit=2.0, silence_limit=0.1, hold_sec=HOLD_SEC) as served:
        yield served


@pytest.fixture
def runtime():
    executor = Executor(time.monotonic_ns, simulated=False, charge_inference_time=False)
    yield executor
    executor.close()


def _policy(socket_path: Path, **kwargs) -> TeleopPolicy:
    return TeleopPolicy('websocket_unix', UnixSocketAddress(socket_path, SESSION_PATH, ''), **kwargs)


class _Headset:
    """A headset that sends one sample of the right controller every 10 ms, with the buttons that a test sets."""

    def __init__(self, service: TeleopService):
        self._service = service
        self.pose: geom.Transform3D | None = HAND
        self.buttons = _buttons()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._send, daemon=True)

    def _send(self) -> None:
        while not self._stop.is_set():
            self._service.take_sample(self.pose, self.buttons)
            time.sleep(0.01)

    def __enter__(self) -> '_Headset':
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def press(self, **buttons: float) -> None:
        """Press ``buttons`` for 50 ms."""
        self.buttons = _buttons(**buttons)
        time.sleep(0.05)
        self.buttons = _buttons()
        time.sleep(0.05)


def _until_phase(service: TeleopService, phase: Phase) -> None:
    deadline = time.monotonic() + 5
    while service.phase is not phase:
        assert time.monotonic() < deadline, f'the service never reached {phase}'
        time.sleep(0.005)


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


def _readied_run(service: TeleopService, headset: '_Headset', runtime: Executor, socket_path: Path):
    """A run that the teleoperator readied, after its stay command."""
    _press_ready_when_asked(service, headset, b=1.0)
    run = runtime.start(_policy(socket_path))
    stay = run.send(OBS)
    assert stay is not None and keys.ROBOT_COMMAND in stay.commands
    return run


def test_meta_probes_the_service(service, socket_path):
    assert _policy(socket_path).meta()[policy_keys.TYPE] == 'teleop'


def test_a_teleoperator_readies_takes_control_starts_and_completes_a_trial(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        assert service.phase is Phase.RUNNING

        headset.press(a=1.0)
        steps, _ = _drive(run, 0.3)
        [first, *_] = _moves(steps)
        np.testing.assert_allclose(first.commands[keys.ROBOT_COMMAND].pose.translation, ROBOT_POSE.translation)

        headset.press(b=1.0)
        _drive(run, 0.3)
        assert policy_keys.ATTEMPT_START_NS in runtime.metadata_snapshot()

        headset.buttons = _buttons(b=1.0)
        _, end = _drive(run, 2.0)
        assert end is PolicyEnd.COMPLETE
        assert service.phase is Phase.ENDED


def test_a_hold_of_b_before_the_start_task_press_does_not_complete(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        _drive(run, 0.2)
        headset.buttons = _buttons(b=1.0)
        _, end = _drive(run, 1.0)
        assert end is None
        assert policy_keys.ATTEMPT_START_NS in runtime.metadata_snapshot()


def test_the_service_shows_the_camera_image_of_the_run(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        _drive(run, 0.2)
        assert service.view().shape[:2] == OBS[keys.EXTERIOR_IMAGE].shape[:2]


def test_a_cancel_fails_the_run_before_its_first_step(service, socket_path, runtime):
    with _Headset(service) as headset:
        _press_ready_when_asked(service, headset, stick=1.0)
        with pytest.raises(session.Cancelled):
            runtime.start(_policy(socket_path))


def test_no_ready_within_the_ready_limit_fails_the_run(socket_path, runtime):
    with TeleopService(socket_path, ready_limit=0.1, silence_limit=0.1, hold_sec=HOLD_SEC):
        with pytest.raises(session.ServiceError, match='not ready within 0.1 s'):
            runtime.start(_policy(socket_path))


def test_a_silent_headset_reads_as_not_tracked_and_turns_control_off(service, socket_path, runtime):
    with _Headset(service) as headset:
        run = _readied_run(service, headset, runtime, socket_path)
        headset.press(a=1.0)
        steps, _ = _drive(run, 0.3)
        assert _moves(steps)
        headset.pose = None
        _drive(run, 0.3)
        headset.pose = HAND
        steps, _ = _drive(run, 0.3)
        assert not _moves(steps)


def test_the_service_refuses_a_second_session(service, socket_path, runtime):
    with _Headset(service) as headset:
        first = _readied_run(service, headset, runtime, socket_path)
        second = Executor(time.monotonic_ns, simulated=False, charge_inference_time=False)
        try:
            with pytest.raises(PeerDisconnected, match='one session at a time'):
                second.start(_policy(socket_path, ready_timeout=1.0))
        finally:
            second.close()
        first.close()


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
def test_a_person_completes_a_trial_on_mujoco_franka(service, socket_path, tmp_path, monkeypatch):
    monkeypatch.setattr('positronic.simulator.mujoco.sim.mj.Renderer', _BlankRenderer)
    ev = stack_cubes(
        mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
        loaders=positronic.cfg.simulator.stack_cubes_loaders(),
        camera_fps=10,
        camera_dict={keys.EXTERIOR_IMAGE: 'back_view_ph'},
        instruction='Pick up the green cube.',
        timeout=60,
    )

    def teleoperate(headset: _Headset) -> None:
        _until_phase(service, Phase.READY_WAIT)
        headset.press(b=1.0)
        _until_phase(service, Phase.RUNNING)
        time.sleep(0.3)
        headset.press(a=1.0)
        headset.pose = geom.Transform3D(HAND.translation + np.array([0.0, 0.0, 0.03]), HAND.rotation)
        time.sleep(0.3)
        headset.press(b=1.0)
        time.sleep(0.3)
        headset.buttons = _buttons(b=1.0)

    with _Headset(service) as headset, pos3.mirror():
        threading.Thread(target=teleoperate, args=(headset,), daemon=True).start()
        main(policy=_policy(socket_path), evals=[ev], output_dir=str(tmp_path))

    [episode] = LocalDataset(tmp_path)
    assert episode.static[eval_keys.ENDED_BY] == eval_keys.ENDED_BY_POLICY
    assert episode.static[eval_keys.POLICY_END] == PolicyEnd.COMPLETE
    assert f'{policy_keys.POLICY_META}.{policy_keys.ATTEMPT_START_NS}' in episode.static
    assert keys.TARGET_EE_POSE in episode.signals


class TestButtonsBetweenTrials:
    def test_a_button_released_after_the_trial_presses_again_in_the_next(self):
        buttons = ButtonMap(HOLD_SEC)
        buttons.events(_buttons(b=1.0), 0, True)
        assert buttons.events(_buttons(b=1.0), 500_000_000, True) == [session.Event.COMPLETE]
        buttons.observe(_buttons(), 600_000_000)
        assert buttons.ready_reply(_buttons(b=1.0), 700_000_000) is session.Reply.READY

    def test_a_button_held_from_the_last_trial_is_no_ready(self):
        buttons = ButtonMap(HOLD_SEC)
        buttons.observe(_buttons(b=1.0), 0)
        assert buttons.ready_reply(_buttons(b=1.0), 100_000_000) is None


def test_a_policy_that_stops_waiting_for_ready_releases_the_session(socket_path, runtime):
    with TeleopService(socket_path, ready_limit=60.0, silence_limit=0.1, hold_sec=HOLD_SEC) as service:
        with pytest.raises(TimeoutError):
            runtime.start(_policy(socket_path, ready_timeout=0.2))
        _until_phase(service, Phase.IDLE)
        with _Headset(service) as headset:
            run = _readied_run(service, headset, runtime, socket_path)
            run.close()


def test_a_stop_of_the_service_ends_its_waits(socket_path):
    service = TeleopService(socket_path, ready_limit=60.0, silence_limit=0.1, hold_sec=HOLD_SEC)
    with service:
        connection = registry.client_wire('websocket_unix').dial(
            UnixSocketAddress(socket_path, SESSION_PATH, ''), None, 5
        )
        connection.send(serialise({session.KIND: session.Request.READY}))
        _until_phase(service, Phase.READY_WAIT)
        stopped_at = time.monotonic()
    assert time.monotonic() - stopped_at < 5
    connection.close()


def test_a_service_that_cannot_bind_raises_the_cause_at_once(tmp_path):
    started_at = time.monotonic()
    with pytest.raises(OSError, match='too long'):
        with TeleopService(tmp_path / ('x' * 120) / 'teleop.sock', 1.0, 0.1, HOLD_SEC):
            pass
    assert time.monotonic() - started_at < 5


def test_a_socket_directory_that_others_can_open_is_refused(tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(PermissionError, match='mode 700'):
        with TeleopService(shared / 'teleop.sock', 1.0, 0.1, HOLD_SEC):
            pass


def test_the_default_socket_with_no_runtime_directory_is_in_a_directory_of_the_user(monkeypatch):
    monkeypatch.delenv('XDG_RUNTIME_DIR', raising=False)
    assert session.default_socket_path().parent.name == f'positronic-{os.getuid()}'


def test_a_hold_of_b_after_start_task_completes_before_the_policy_hears_of_the_start(service, socket_path):
    connection = registry.client_wire('websocket_unix').dial(UnixSocketAddress(socket_path, SESSION_PATH, ''), None, 5)
    try:
        connection.send(serialise({session.KIND: session.Request.READY}))
        _until_phase(service, Phase.READY_WAIT)
        service.take_sample(HAND, _buttons(b=1.0))
        service.take_sample(HAND, _buttons())
        assert deserialise(connection.recv(5)) == {session.KIND: session.Reply.READY}
        service.take_sample(HAND, _buttons(b=1.0))
        service.take_sample(HAND, _buttons())
        service.take_sample(HAND, _buttons(b=1.0))
        time.sleep(HOLD_SEC + 0.05)
        service.take_sample(HAND, _buttons(b=1.0))
        connection.send(serialise({session.KIND: session.Request.FRAME, session.DISPLAY: {}}))
        frame = deserialise(connection.recv(5))
        assert frame[session.EVENTS] == [session.Event.START_TASK, session.Event.COMPLETE]
    finally:
        connection.close()


class _Peer:
    """The service's end of a session, whose receive gives ``outcome``: a message, or a raised error."""

    def __init__(self, outcome: bytes | Exception):
        self._outcome = outcome

    async def receive(self) -> bytes:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _race(service: TeleopService, peer: _Peer):
    wait = asyncio.to_thread(service._await_ready)
    return asyncio.run(service._answer_unless_the_peer_speaks(cast(server_wire.ServerConnection, peer), wait))


def test_a_disconnect_during_the_ready_wait_ends_the_wait_and_the_session(socket_path):
    service = TeleopService(socket_path, 60.0, 0.1, HOLD_SEC)
    assert _race(service, _Peer(PeerDisconnected('gone'))) is None
    assert service.phase is Phase.IDLE


def test_a_message_during_the_ready_wait_is_logged_and_ends_the_session(socket_path, caplog):
    service = TeleopService(socket_path, 60.0, 0.1, HOLD_SEC)
    assert _race(service, _Peer(b'too early')) is None
    assert 'sent a message while it waited' in caplog.text


def test_another_receive_failure_during_the_ready_wait_raises_after_the_wait_ends(socket_path):
    service = TeleopService(socket_path, 60.0, 0.1, HOLD_SEC)
    with pytest.raises(ValueError, match='binary'):
        _race(service, _Peer(ValueError('A session message must be a binary WebSocket frame')))
    assert service.phase is Phase.IDLE
