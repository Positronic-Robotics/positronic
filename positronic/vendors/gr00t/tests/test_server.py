from unittest.mock import Mock

import msgpack
import msgpack_numpy
import numpy as np
import pytest
import zmq

from positronic.offboard import keys as offboard_keys
from positronic.policy.codec import ACTION
from positronic.vendors import gr00t
from positronic.vendors.gr00t import server as gr00t_server


def _modalities(cameras: list[str]) -> dict:
    return {
        gr00t.VIDEO: {gr00t.DELTA_INDICES: [0], gr00t.MODALITY_KEYS: cameras},
        gr00t.STATE: {gr00t.DELTA_INDICES: [0], gr00t.MODALITY_KEYS: list(gr00t.STATE_DIMS)},
        ACTION: {gr00t.DELTA_INDICES: list(range(40)), gr00t.MODALITY_KEYS: list(gr00t.STATE_DIMS)},
        gr00t.LANGUAGE: {gr00t.DELTA_INDICES: [0], gr00t.MODALITY_KEYS: [gr00t.TASK]},
    }


def _backend(monkeypatch, modalities: dict) -> Mock:
    """A gr00t subprocess that answers ``modalities``, and no warmup inference."""
    backend = Mock()
    backend.client.call_endpoint.return_value = modalities
    monkeypatch.setattr(gr00t_server, 'Gr00tSubprocess', Mock(return_value=backend))
    monkeypatch.setattr(gr00t_server, 'warmup', Mock())
    return backend


@pytest.mark.parametrize(
    ('checkpoint', 'expected', 'directory'),
    [
        (None, '10000', 'checkpoint-010000'),
        ('5000', '5000', 'checkpoint-005000'),
        ('005000', '5000', 'checkpoint-005000'),
    ],
)
def test_zero_padded_checkpoints_are_served_under_their_step(monkeypatch, checkpoint, expected, directory):
    """The padding is the directory's, not the model's: one checkpoint records one id, or analysis splits
    it in two."""
    names = ['checkpoint-005000', 'checkpoint-010000']
    monkeypatch.setattr(gr00t_server, 'list_checkpoints', lambda _dir, prefix='': names)
    downloaded = []
    monkeypatch.setattr(gr00t_server.pos3, 'download', lambda path, exclude=(): downloaded.append(path) or path)
    _backend(monkeypatch, _modalities([gr00t.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE]))

    model = gr00t_server.gr00t_model(model_source='s3://bucket/exp', checkpoint=checkpoint)

    assert model.meta()[offboard_keys.CHECKPOINT_ID] == expected
    # The raw suffix survives only where it is needed — reaching the directory.
    assert downloaded == [f's3://bucket/exp/{directory}']


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
    monkeypatch.setattr(gr00t_server, 'list_checkpoints', Mock(side_effect=AssertionError('scanned')))
    _backend(monkeypatch, _modalities([gr00t.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE]))

    assert gr00t_server.gr00t_model().meta()[offboard_keys.CHECKPOINT_ID] == gr00t.BASE_MODEL


@pytest.mark.parametrize('checkpoint_cameras', [2, 3])
def test_the_warmup_runs_in_the_checkpoints_own_cameras(monkeypatch, checkpoint_cameras):
    cameras = [gr00t.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE]
    if checkpoint_cameras == 3:
        cameras.append(gr00t.EXTERIOR_IMAGE_2)
    _backend(monkeypatch, _modalities(cameras))

    model = gr00t_server.gr00t_model()
    try:
        assert set(gr00t_server.warmup.call_args.args[1][gr00t.VIDEO]) == set(cameras)
    finally:
        model.close()


@pytest.mark.parametrize('modality', [gr00t.STATE, gr00t.LANGUAGE])
def test_a_checkpoint_the_warmup_cannot_serve_stops_the_subprocess(monkeypatch, modality):
    modalities = _modalities([gr00t.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE])
    modalities[modality][gr00t.MODALITY_KEYS] = ['incompatible_field']
    backend = _backend(monkeypatch, modalities)

    with pytest.raises(ValueError, match='Checkpoint .* key'):
        gr00t_server.gr00t_model()
    backend.stop.assert_called_once()


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
        assert client.ping() is gr00t_server.PingResult.FAILURE
        assert client.ping() is gr00t_server.PingResult.SUCCESS
        request = gr00t_server.MsgSerializer.from_bytes(recovered.send.call_args.args[0])
        assert request == {'endpoint': 'ping'}
    finally:
        client.close()
