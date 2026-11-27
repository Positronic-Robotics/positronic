import asyncio
import contextlib
import json
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from unittest import mock

import numpy as np
import pytest

from positronic import geom, keys
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.command import CartesianDelta
from positronic.drivers.roboarm.models import (
    FRANKA_HAND_FRAME,
    GRIPPER_SITE,
    PANDA_HAND_FRAME,
    EEFrame,
    parse_ee_frames,
)
from positronic.offboard import server_utils
from positronic.policy.codecs import ACTION
from positronic.policy.processors import InferenceSchedule
from positronic.policy.spec import from_spec
from positronic.vendors import openpi
from positronic.vendors.openpi import codecs

pytest.importorskip('openpi_client')

from openpi_client import msgpack_numpy
from platform_client.policy_container import WARMUP_DEADLINE_S
from positronic_model_server.spec import ARGS, NAME, SEQ
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

from positronic.vendors.openpi.server import COMMANDS, PREALLOCATE_ENV, DeadlineWebsocketClientPolicy, OpenpiSubprocess


def _subprocess_env(monkeypatch) -> dict[str, str]:
    """The environment ``start`` gives the openpi subprocess."""
    monkeypatch.setattr(OpenpiSubprocess, '_wait_for_ready', lambda self: None)
    with mock.patch('subprocess.Popen') as popen:
        OpenpiSubprocess(
            checkpoint_dir='/checkpoints/exp/1000', config_name='pi05_positronic_lowmem', infer_timeout=120.0
        ).start()
    return popen.call_args.kwargs['env']


def test_the_subprocess_does_not_preallocate_the_gpu(monkeypatch):
    monkeypatch.delenv(PREALLOCATE_ENV, raising=False)

    assert _subprocess_env(monkeypatch)[PREALLOCATE_ENV] == 'false'


def test_the_preallocation_setting_of_the_environment_reaches_the_subprocess(monkeypatch):
    monkeypatch.setenv(PREALLOCATE_ENV, 'true')

    assert _subprocess_env(monkeypatch)[PREALLOCATE_ENV] == 'true'


@pytest.mark.parametrize('name', ['serve', 'ee', 'ee_joints', 'ee_traj', 'ee_joints_traj', 'ee_flip_grip'])
def test_a_pose_pipeline_that_states_no_frame_fails_to_build(name):
    with pytest.raises(TypeError, match='ee_frame'):
        COMMANDS[name].kwargs['pipeline'].instantiate()


@pytest.mark.parametrize(
    'name, offset', [('phail', FRANKA_HAND_FRAME), ('sim_stack', PANDA_HAND_FRAME), ('libero', PANDA_HAND_FRAME)]
)
def test_a_bound_checkpoint_states_the_frame_it_was_trained_in(name, offset):
    deployment = COMMANDS[name].kwargs['pipeline'].instantiate()
    frame = parse_ee_frames(deployment.meta()[roboarm_keys.EE_FRAMES])[keys.EE_POSE]
    assert frame == EEFrame(site=GRIPPER_SITE, offset=offset)


def _latency_estimate(deployment) -> float:
    """The lead, in seconds, that the deployment's client schedule declares."""
    stack = deployment.local.to_spec()[SEQ]
    (schedule,) = (node for node in stack if node[NAME] == InferenceSchedule.WIRE_NAME)
    return schedule[ARGS]['latency_estimate']


@pytest.mark.parametrize('name', ['droid', 'droid_jointpos'])
def test_a_session_inference_lead_requests_before_the_chunk_ends(name):
    pipeline = COMMANDS[name].kwargs['pipeline']

    assert _latency_estimate(pipeline.instantiate()) == 0.0
    assert _latency_estimate(pipeline.override_data(inference_lead=0.25).instantiate()) == 0.25


@pytest.mark.parametrize('lead', [-0.1, float('inf'), float('nan')])
def test_a_negative_or_infinite_inference_lead_fails_to_build(lead):
    with pytest.raises(ValueError, match='seconds'):
        COMMANDS['droid_jointpos'].kwargs['pipeline'].override_data(inference_lead=lead).instantiate()


@contextlib.contextmanager
def _openpi_server(before_answer: Callable[[], Awaitable[None]]) -> Iterator[int]:
    """The port of a server that speaks openpi's protocol. It awaits ``before_answer`` before each answer."""

    async def handler(connection: ServerConnection) -> None:
        await connection.send(msgpack_numpy.Packer().pack({}))
        async for message in connection:
            obs = msgpack_numpy.unpackb(message)
            await before_answer()
            with contextlib.suppress(ConnectionClosed):
                await connection.send(msgpack_numpy.Packer().pack({'echo': obs['value']}))

    async def start() -> Server:
        return await serve(handler, '127.0.0.1', 0)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    server = asyncio.run_coroutine_threadsafe(start(), loop).result(timeout=5)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        loop.call_soon_threadsafe(server.close)
        asyncio.run_coroutine_threadsafe(server.wait_closed(), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


@pytest.fixture
def short_keepalive(monkeypatch):
    monkeypatch.setattr(DeadlineWebsocketClientPolicy, 'PING_INTERVAL_S', 0.1)
    monkeypatch.setattr(DeadlineWebsocketClientPolicy, 'PING_TIMEOUT_S', 0.1)


@pytest.mark.usefixtures('short_keepalive')
def test_the_subprocess_client_closes_when_the_subprocess_answers_no_ping():
    async def block_the_event_loop() -> None:
        time.sleep(1.0)

    with _openpi_server(block_the_event_loop) as port:
        client = DeadlineWebsocketClientPolicy(host='127.0.0.1', port=port, infer_timeout=30.0)
        with pytest.raises(ConnectionClosedError, match='keepalive ping timeout'):
            client.infer({'value': 1})


@pytest.mark.usefixtures('short_keepalive')
def test_an_inference_that_outlasts_the_keepalive_window_returns_within_the_deadline():
    with _openpi_server(lambda: asyncio.sleep(0.6)) as port:
        client = DeadlineWebsocketClientPolicy(host='127.0.0.1', port=port, infer_timeout=5.0)
        try:
            assert client.infer({'value': 7}) == {'echo': 7}
        finally:
            client._ws.close()


@pytest.mark.usefixtures('short_keepalive')
def test_an_inference_past_the_deadline_fails_and_closes_the_connection():
    with _openpi_server(lambda: asyncio.sleep(1.5)) as port:
        client = DeadlineWebsocketClientPolicy(host='127.0.0.1', port=port, infer_timeout=0.5)
        with pytest.raises(TimeoutError, match=r'no inference result within infer_timeout=0\.5s'):
            client.infer({'value': 1})
        with pytest.raises(ConnectionClosed):
            client.infer({'value': 2})


def _first_answer_waits(wait: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
    """A ``before_answer`` that awaits ``wait`` before the first answer and answers every later one at once."""
    waited = threading.Event()

    async def before_answer() -> None:
        if not waited.is_set():
            waited.set()
            await wait()

    return before_answer


def _subprocess_on(port: int, infer_timeout: float) -> OpenpiSubprocess:
    return OpenpiSubprocess(
        checkpoint_dir='/checkpoints/exp/1000', config_name='pi05_droid', infer_timeout=infer_timeout, ws_port=port
    )


@pytest.mark.usefixtures('short_keepalive')
def test_the_inference_after_a_keepalive_timeout_reaches_the_subprocess_again():
    async def block_the_event_loop() -> None:
        time.sleep(1.0)

    with _openpi_server(_first_answer_waits(block_the_event_loop)) as port:
        subproc = _subprocess_on(port, infer_timeout=30.0)
        try:
            with pytest.raises(ConnectionClosedError, match='keepalive ping timeout'):
                subproc.client.infer({'value': 1})
            assert subproc.client.infer({'value': 2}) == {'echo': 2}
        finally:
            subproc.stop()


@pytest.mark.usefixtures('short_keepalive')
def test_the_inference_after_a_missed_deadline_reaches_the_subprocess_again():
    with _openpi_server(_first_answer_waits(lambda: asyncio.sleep(1.5))) as port:
        subproc = _subprocess_on(port, infer_timeout=0.5)
        try:
            with pytest.raises(TimeoutError, match='no inference result'):
                subproc.client.infer({'value': 1})
            assert subproc.client.infer({'value': 2}) == {'echo': 2}
        finally:
            subproc.stop()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('overrides', [{}, {'config_name': 'pi0_libero'}], ids=['default', 'config-override'])
def test_libero_server_converts_predictions_before_command_decoding(dtype, overrides):
    predictions = np.random.default_rng(42).uniform(-2, 2, (32, 7)).astype(dtype)
    predictions[:3] = np.array([-1, 0, 1])[:, None]
    original = predictions.copy()
    with (
        mock.patch('positronic.vendors.openpi.server.OpenpiSubprocess', autospec=True) as subprocess,
        mock.patch('positronic.vendors.openpi.server.warmup'),
    ):
        subprocess.return_value.client.infer.return_value = {openpi.ACTIONS: predictions}
        model = COMMANDS['libero'].kwargs['model'].override(**overrides).instantiate()
        chunk = model({}, session_id='test')

    decoder = from_spec(json.loads(json.dumps(codecs.libero_action.instantiate().to_spec())))
    decoded = decoder.decode(chunk)
    composed = codecs.libero.instantiate().decode(chunk)
    assert len(chunk) == len(decoded) == len(composed) == len(predictions)
    for row, record, action, full_action in zip(predictions, chunk, decoded, composed, strict=True):
        clipped = row.clip(-1, 1)
        physical = clipped[:6] * np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
        expected = geom.Transform3D(translation=physical[:3], rotation=geom.Rotation.from_rotvec(physical[3:]))
        np.testing.assert_array_equal(record[ACTION][:6], physical)
        assert record[ACTION][6] == (float(clipped[6]) + 1.0) / 2.0
        assert set(record) == {ACTION}
        for result in (action, full_action):
            command = result[keys.ROBOT_COMMAND]
            assert isinstance(command, CartesianDelta)
            assert command.mode is None
            np.testing.assert_array_equal(command.delta.as_matrix, expected.as_matrix)
            np.testing.assert_array_equal(command.frame.as_matrix, np.eye(4))
            assert result[keys.TARGET_GRIP] == (float(clipped[6]) + 1.0) / 2.0
    np.testing.assert_array_equal(predictions, original)


@pytest.mark.parametrize('preset', [name for name in COMMANDS if name != 'libero'])
def test_other_server_presets_preserve_native_predictions(preset):
    predictions = np.arange(24, dtype=np.float32).reshape(3, 8)
    with (
        mock.patch('positronic.vendors.openpi.server.OpenpiSubprocess', autospec=True) as subprocess,
        mock.patch('positronic.vendors.openpi.server.warmup'),
    ):
        subprocess.return_value.client.infer.return_value = {openpi.ACTIONS: predictions}
        model = COMMANDS[preset].kwargs['model'].override(checkpoints_dir='gs://checkpoints/test').instantiate()
        chunk = model({}, session_id='test')

    np.testing.assert_array_equal(np.stack([record[ACTION] for record in chunk]), predictions, strict=True)


class _Clock:
    """A monotonic clock that each sleep moves forward."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _start_ready_after(monkeypatch, seconds: float) -> None:
    """Start the subprocess on a fake clock, with a server that accepts connections `seconds` after the start."""
    clock = _Clock()
    monkeypatch.setattr(server_utils.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(server_utils.time, 'sleep', clock.sleep)
    monkeypatch.setattr(OpenpiSubprocess, '_is_ready', lambda self: clock.now >= seconds)
    with mock.patch('subprocess.Popen') as popen:
        popen.return_value.poll.return_value = None
        OpenpiSubprocess(checkpoint_dir='/checkpoints/exp/1000', config_name='pi05_droid', infer_timeout=120.0).start()


def test_a_subprocess_that_starts_slower_than_300_s_still_serves(monkeypatch):
    _start_ready_after(monkeypatch, 301.0)


def test_a_subprocess_that_does_not_start_within_the_published_deadline_fails(monkeypatch):
    with pytest.raises(RuntimeError, match='did not become ready'):
        _start_ready_after(monkeypatch, WARMUP_DEADLINE_S + 10.0)
