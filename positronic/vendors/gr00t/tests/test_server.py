from unittest.mock import Mock

import msgpack
import msgpack_numpy
import numpy as np
import pytest
import zmq

from positronic.offboard.server import PolicyServer
from positronic.vendors import gr00t
from positronic.vendors.gr00t import server as gr00t_server


def _source(monkeypatch, checkpoints: list[str]) -> gr00t_server.Gr00tSource:
    monkeypatch.setattr(gr00t_server, 'list_checkpoints', lambda _dir, prefix='': checkpoints)
    return gr00t_server.droid.override_data(**{'source.checkpoints_dir': 's3://bucket/exp'})().source


def test_zero_padded_checkpoints_are_served_under_the_id_they_advertise(monkeypatch):
    """The padding is the directory's, not the model's: a client asking for the advertised id must be
    recorded under that same id, or analysis splits one checkpoint in two."""
    source = _source(monkeypatch, ['checkpoint-005000', 'checkpoint-010000'])

    assert source.get_models() == ['5000', '10000']
    assert source.resolve('5000') == '5000'
    assert source.resolve(None) == '10000'
    # The raw suffix survives only where it is needed — reaching the directory.
    assert source._raw_for('5000') == '005000'


def test_msgpack_numpy_preserves_actions_and_camera_arrays():

    actions = {gr00t.JOINT_POSITION: np.arange(280, dtype=np.float32).reshape(1, 40, 7)}
    upstream_bytes = msgpack.packb((actions, {}), default=msgpack_numpy.encode)
    decoded, _ = gr00t_server.MsgSerializer.from_bytes(upstream_bytes)
    np.testing.assert_array_equal(decoded[gr00t.JOINT_POSITION], actions[gr00t.JOINT_POSITION])
    image = np.arange(180 * 320 * 3, dtype=np.uint8).reshape(1, 1, 180, 320, 3)
    encoded = gr00t_server.MsgSerializer.to_bytes({gr00t.VIDEO: image})
    np.testing.assert_array_equal(msgpack.unpackb(encoded, object_hook=msgpack_numpy.decode)[gr00t.VIDEO], image)


def test_serializer_rejects_pickle_bearing_arrays():

    with pytest.raises(TypeError, match='Object arrays'):
        gr00t_server.MsgSerializer.to_bytes(np.array([object()], dtype=object))
    for payload in ({b'nd': True, b'kind': b'O'}, {'nd': 1, 'kind': 'O'}):
        with pytest.raises(ValueError, match='Object arrays'):
            gr00t_server.MsgSerializer.from_bytes(msgpack.packb(payload))


def test_published_checkpoint_is_served_without_a_local_checkpoint_scan(monkeypatch):

    source = gr00t_server.droid().source
    assert source.get_models() == [gr00t.BASE_MODEL]
    assert source.resolve(None) == gr00t.BASE_MODEL


@pytest.mark.parametrize('config', [gr00t_server.droid, gr00t_server.droid_three_cameras])
@pytest.mark.parametrize('checkpoint_cameras', [2, 3])
def test_camera_mismatch_stops_the_backend_before_warmup(monkeypatch, config, checkpoint_cameras):
    source = config().source
    cameras = [gr00t.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE]
    if checkpoint_cameras == 3:
        cameras.append(gr00t.EXTERIOR_IMAGE_2)
    backend = Mock()
    backend.client.call_endpoint.return_value = {
        gr00t.VIDEO: {gr00t.DELTA_INDICES: [0], gr00t.MODALITY_KEYS: cameras},
        gr00t.STATE: {gr00t.DELTA_INDICES: [0], gr00t.MODALITY_KEYS: list(gr00t.STATE_DIMS)},
    }
    monkeypatch.setattr(gr00t_server, 'Gr00tSubprocess', Mock(return_value=backend))
    warmup = Mock()
    monkeypatch.setattr(gr00t_server, 'warmup', warmup)
    if len(source.video_keys) != checkpoint_cameras:
        with pytest.raises(ValueError, match='Checkpoint video keys'):
            source.load(gr00t.BASE_MODEL)
        warmup.assert_not_called()
        backend.stop.assert_called_once()
    else:
        policy = source.load(gr00t.BASE_MODEL)
        try:
            assert set(warmup.call_args.args[1][gr00t.VIDEO]) == set(cameras)
            backend.stop.assert_not_called()
        finally:
            policy.close()


def test_session_timing_overrides_preserve_source_equality():
    server = PolicyServer(gr00t_server.droid)
    variant = server._session_pipeline({'codec.fps': 10.0})
    assert variant.source == gr00t_server.droid().source


@pytest.mark.parametrize('failure', [zmq.Again(), zmq.ZMQError(zmq.EFSM)])
def test_client_can_ping_after_a_transport_failure(monkeypatch, failure):
    failed = Mock()
    failed.send.side_effect = failure
    recovered = Mock()
    recovered.recv.return_value = gr00t_server.MsgSerializer.to_bytes('pong')
    context = Mock()
    context.socket.side_effect = [failed, recovered]
    monkeypatch.setattr(gr00t_server.zmq, 'Context', lambda: context)
    client = gr00t_server.PolicyClient()
    try:
        assert not client.ping()
        assert client.ping()
        request = gr00t_server.MsgSerializer.from_bytes(recovered.send.call_args.args[0])
        assert request == {'endpoint': 'ping'}
    finally:
        client.close()
