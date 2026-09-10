from unittest.mock import Mock

import msgpack
import msgpack_numpy
import numpy as np
import pytest
import zmq

from positronic.vendors import gr00t
from positronic.vendors.gr00t import server as gr00t_server


def _source(monkeypatch, checkpoints: list[str]) -> gr00t_server.Gr00tSource:
    monkeypatch.setattr(gr00t_server, 'list_checkpoints', lambda _dir, prefix='': checkpoints)
    return gr00t_server.Gr00tSource('s3://bucket/exp')


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

    source = gr00t_server.Gr00tSource()
    assert source.get_models() == [gr00t.BASE_MODEL]
    assert source.resolve(None) == gr00t.BASE_MODEL


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
