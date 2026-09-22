"""The roboarena client, driven against a fake connection."""

from unittest.mock import MagicMock, patch

import pytest
from positronic_wire import wire

from positronic.offboard import roboarena
from positronic.utils.serialization import deserialize, serialize


def _client(connection) -> roboarena.RoboarenaClient:
    client = roboarena.RoboarenaClient('a-partner-host', 8000)
    client._connection = connection
    return client


def test_a_reset_acknowledged_in_text_ends_the_session():
    """The bundled backend answers `reset successful` as a text frame, which the wire reports as the peer."""
    connection = MagicMock(**{'recv.side_effect': wire.PeerDisconnected('the server answered this error text: ok')})

    _client(connection).reset(session_id='an-episode')

    sent = deserialize(connection.send.call_args.args[0])
    assert sent == {roboarena.ENDPOINT: roboarena.RESET, roboarena.SESSION_ID: 'an-episode'}


def test_a_reset_acknowledged_in_a_frame_ends_the_session():
    connection = MagicMock(**{'recv.return_value': serialize({'ok': True})})

    _client(connection).reset()

    assert deserialize(connection.send.call_args.args[0]) == {roboarena.ENDPOINT: roboarena.RESET}


def test_a_readiness_poll_waits_far_less_than_a_handshake():
    """`wait_for_subprocess_ready` polls this between heartbeats, so a silent backend must not hold it."""
    assert roboarena.READY_PROBE_TIMEOUT_S < roboarena.HANDSHAKE_TIMEOUT_S

    client = roboarena.RoboarenaClient('a-partner-host', 8000)
    with patch.object(client, '_wire') as client_wire:
        client_wire.probe.return_value = None
        assert client.is_ready()
    assert client_wire.probe.call_args.args[2] == roboarena.READY_PROBE_TIMEOUT_S


def test_a_handshake_that_does_not_answer_closes_the_connection_it_opened():
    """Nothing else holds it: `_connection` is assigned after the read, so an unclosed one leaks."""
    connection = MagicMock(**{'recv.side_effect': TimeoutError('timed out')})
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.dial.return_value = connection
        with pytest.raises(TimeoutError):
            client.connect()

    connection.close.assert_called_once()


def test_an_inference_that_does_not_answer_drops_the_connection():
    """A reply arriving after this read gave up would be read by the NEXT inference as its own."""
    connection = MagicMock(**{'recv.side_effect': TimeoutError('timed out')})
    client = _client(connection)

    with pytest.raises(TimeoutError):
        client.infer({'observation/joint_position': 0})

    connection.close.assert_called_once()
    assert client._connection is None
