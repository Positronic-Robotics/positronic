import json
import threading
from unittest import mock

import numpy as np
import pytest

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianDelta
from positronic.policy.codecs import ACTION
from positronic.policy.spec import from_spec
from positronic.vendors import openpi
from positronic.vendors.openpi import codecs

pytest.importorskip('openpi_client')

from openpi_client import msgpack_numpy
from websockets.sync.server import ServerConnection, serve

from positronic.vendors.openpi.server import COMMANDS, PREALLOCATE_ENV, OpenpiSubprocess, SubprocessClient


def _subprocess_env(monkeypatch) -> dict[str, str]:
    """The environment ``start`` gives the openpi subprocess."""
    monkeypatch.setattr(OpenpiSubprocess, '_wait_for_ready', lambda self: None)
    with mock.patch('subprocess.Popen') as popen:
        OpenpiSubprocess(checkpoint_dir='/checkpoints/exp/1000', config_name='pi05_positronic_lowmem').start()
    return popen.call_args.kwargs['env']


def test_the_subprocess_does_not_preallocate_the_gpu(monkeypatch):
    monkeypatch.delenv(PREALLOCATE_ENV, raising=False)

    assert _subprocess_env(monkeypatch)[PREALLOCATE_ENV] == 'false'


def test_the_preallocation_setting_of_the_environment_reaches_the_subprocess(monkeypatch):
    monkeypatch.setenv(PREALLOCATE_ENV, 'true')

    assert _subprocess_env(monkeypatch)[PREALLOCATE_ENV] == 'true'


def _send_metadata_and_wait(connection: ServerConnection) -> None:
    connection.send(msgpack_numpy.Packer().pack({}))
    for _ in connection:
        pass


def test_the_subprocess_client_sends_no_keepalive_pings():
    with serve(_send_metadata_and_wait, '127.0.0.1', 0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        client = SubprocessClient(host='127.0.0.1', port=server.socket.getsockname()[1])
        try:
            assert client._ws.ping_interval is None
            assert client._ws.keepalive_thread is None
        finally:
            client._ws.close()
            server.shutdown()


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
