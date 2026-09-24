from unittest.mock import Mock

import msgpack
import msgpack_numpy
import numpy as np
import pytest
import zmq

from positronic.offboard import keys as offboard_keys
from positronic.policy.codec import ACTION, GR00T_MODALITY
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


@pytest.mark.parametrize('config', [gr00t_server.droid, gr00t_server.droid_three_cameras])
@pytest.mark.parametrize('checkpoint_cameras', [2, 3])
def test_a_checkpoint_refuses_a_codec_with_other_cameras(monkeypatch, config, checkpoint_cameras):
    cameras = [gr00t.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE]
    if checkpoint_cameras == 3:
        cameras.append(gr00t.EXTERIOR_IMAGE_2)
    _backend(monkeypatch, _modalities(cameras))
    codec = config().codec

    model = gr00t_server.gr00t_model()
    try:
        # The warmup runs in the checkpoint's own cameras, whatever codec the server pairs it with.
        assert set(gr00t_server.warmup.call_args.args[1][gr00t.VIDEO]) == set(cameras)
        if len(codec.training_encoder.meta[GR00T_MODALITY][gr00t.VIDEO]) != checkpoint_cameras:
            with pytest.raises(ValueError, match='Checkpoint video keys'):
                model.check_codec(codec)
        else:
            model.check_codec(codec)
    finally:
        model.close()


@pytest.mark.parametrize('modality', [gr00t.STATE, ACTION, gr00t.LANGUAGE])
def test_same_camera_checkpoint_with_incompatible_modalities_is_refused(monkeypatch, modality):
    codec = gr00t_server.droid().codec
    declared = codec.training_encoder.meta[GR00T_MODALITY]
    modalities = {
        name: {gr00t.DELTA_INDICES: [0], gr00t.MODALITY_KEYS: list(declared[name])}
        for name in (gr00t.VIDEO, gr00t.STATE, ACTION)
    }
    modalities[gr00t.LANGUAGE] = {gr00t.DELTA_INDICES: [0], gr00t.MODALITY_KEYS: [gr00t.TASK]}
    modalities[modality][gr00t.MODALITY_KEYS] = ['incompatible_field']
    backend = _backend(monkeypatch, modalities)

    built = []
    with pytest.raises(ValueError, match='Checkpoint .* key'):
        built.append(gr00t_server.gr00t_model())
        built[0].check_codec(codec)
    for model in built:
        model.close()
    backend.stop.assert_called_once()


def test_a_codec_that_declares_no_modality_is_refused(monkeypatch):
    _backend(monkeypatch, _modalities([gr00t.EXTERIOR_IMAGE, gr00t.WRIST_IMAGE]))
    model = gr00t_server.gr00t_model()
    try:
        with pytest.raises(ValueError, match='declares its modality'):
            model.check_codec(None)
    finally:
        model.close()


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
