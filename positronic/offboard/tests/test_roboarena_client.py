"""The roboarena client, driven against a fake connection."""

from unittest.mock import MagicMock, patch

import pytest
from positronic_wire import roboarena as roboarena_wire
from positronic_wire import wire
from websockets.exceptions import ConnectionClosedError

from positronic.offboard import roboarena
from positronic.utils.serialization import deserialize, serialize


def _client(connection) -> roboarena.RoboarenaClient:
    client = roboarena.RoboarenaClient('a-partner-host', 8000)
    client._connection = connection
    return client


_ANNOUNCEMENT = serialize({'resolution': [180, 320]})

# Each way an exchange on an open connection fails, as the connection raises it.
_FAILURES = {
    'send-disconnect': ({'send.side_effect': wire.PeerDisconnected('gone')}, wire.PeerDisconnected),
    'read-timeout': ({'recv.side_effect': TimeoutError('timed out')}, TimeoutError),
    'read-disconnect': ({'recv.side_effect': wire.PeerDisconnected('gone')}, wire.PeerDisconnected),
    'read-text': ({'recv.side_effect': roboarena_wire.TextAnswer('CUDA out of memory')}, roboarena_wire.TextAnswer),
}


@pytest.mark.parametrize('failure', _FAILURES.values(), ids=_FAILURES.keys())
@pytest.mark.parametrize('verb', ['infer', 'reset'])
def test_a_failed_exchange_drops_the_connection_and_raises(verb, failure):
    """A reply after the failure stays queued, so no later exchange may read it as its own."""
    effects, raised = failure
    connection = MagicMock(**effects)
    client = _client(connection)

    with pytest.raises(raised):
        if verb == 'infer':
            client.infer({})
        else:
            client.reset(session_id='an-episode')

    connection.close.assert_called_once()
    assert client._connection is None


@pytest.mark.parametrize(
    'failure', [TimeoutError('timed out'), wire.PeerDisconnected('gone'), roboarena_wire.TextAnswer('x')]
)
def test_a_handshake_that_fails_closes_the_connection_it_opened_and_raises(failure):
    connection = MagicMock(**{'recv.side_effect': failure})
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.dial.return_value = connection
        with pytest.raises(type(failure)):
            client.connect()

    connection.close.assert_called_once()
    assert client._connection is None


def test_an_inference_with_no_connection_dials_one_and_keeps_it():
    connection = MagicMock(**{'recv.side_effect': [_ANNOUNCEMENT, serialize([0.0] * 8)]})
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.dial.return_value = connection
        assert client.infer({}) == [0.0] * 8

    assert client._connection is connection
    connection.close.assert_not_called()


def test_a_reset_with_no_connection_dials_one_and_sends_the_keyed_reset():
    """A failed inference drops the connection, and the backend still holds that session's history."""
    connection = MagicMock(**{'recv.side_effect': [_ANNOUNCEMENT, serialize({'ok': True})]})
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.dial.return_value = connection
        client.reset(session_id='an-episode')

    sent = deserialize(connection.send.call_args.args[0])
    assert sent == {roboarena.ENDPOINT: roboarena.RESET, roboarena.SESSION_ID: 'an-episode'}


def test_a_reset_with_no_connection_raises_when_the_server_refuses_the_dial():
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.dial.side_effect = wire.ConnectRefused(wire.Refusal.COLD, 'refused')
        with pytest.raises(wire.ConnectRefused):
            client.reset(session_id='an-episode')


def test_a_reset_acknowledged_in_text_ends_the_session():
    websocket = MagicMock(**{'recv.return_value': roboarena.RESET_ACKNOWLEDGEMENT})

    client = _client(roboarena_wire.RoboarenaClientConnection(websocket))

    client.reset(session_id='an-episode')

    sent = deserialize(websocket.send.call_args.args[0])
    assert sent == {roboarena.ENDPOINT: roboarena.RESET, roboarena.SESSION_ID: 'an-episode'}
    assert client._connection is None


def test_a_reset_answered_with_error_text_reaches_the_caller():
    """Error text propagates; it does not count as the reset acknowledgement."""
    websocket = MagicMock(**{'recv.return_value': 'CUDA out of memory'})

    client = _client(roboarena_wire.RoboarenaClientConnection(websocket))

    with pytest.raises(roboarena_wire.TextAnswer, match='CUDA out of memory'):
        client.reset(session_id='an-episode')

    assert client._connection is None


def test_an_error_text_that_ends_with_the_acknowledgement_reaches_the_caller():
    """Only the exact acknowledgement ends the reset. Text that ends with it is still a failure."""
    near_miss = f'expected {roboarena.RESET_ACKNOWLEDGEMENT}'
    websocket = MagicMock(**{'recv.return_value': near_miss})

    with pytest.raises(roboarena_wire.TextAnswer, match=near_miss):
        _client(roboarena_wire.RoboarenaClientConnection(websocket)).reset()


def test_a_reset_the_peer_never_answered_reaches_the_caller():
    closed = ConnectionClosedError(None, None)
    websocket = MagicMock(**{'recv.side_effect': closed})

    with pytest.raises(wire.PeerDisconnected) as ended:
        _client(roboarena_wire.RoboarenaClientConnection(websocket)).reset()

    assert ended.value.__cause__ is closed


def test_a_reset_acknowledged_in_a_frame_keeps_the_connection():
    connection = MagicMock(**{'recv.return_value': serialize({'ok': True})})
    client = _client(connection)

    client.reset()

    assert deserialize(connection.send.call_args.args[0]) == {roboarena.ENDPOINT: roboarena.RESET}
    connection.close.assert_not_called()
    assert client._connection is connection


def test_a_readiness_poll_waits_far_less_than_a_handshake():
    """Readiness reads on the short probe timeout, not the handshake's."""
    assert roboarena.READY_PROBE_TIMEOUT_S < roboarena.HANDSHAKE_TIMEOUT_S

    client = roboarena.RoboarenaClient('a-partner-host', 8000)
    with patch.object(client, '_wire') as client_wire:
        client_wire.probe.return_value = None
        assert client.is_ready()
    assert client_wire.probe.call_args.args[2] == roboarena.READY_PROBE_TIMEOUT_S


def test_readiness_raises_the_text_a_server_answers_in():
    """A backend that reports a failure is not a backend still starting, so a readiness poll does not retry it."""
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.return_value = 'CUDA out of memory'
        with pytest.raises(roboarena_wire.TextAnswer, match='CUDA out of memory'):
            client.is_ready()


def test_readiness_of_a_peer_that_closes_before_announcing_is_false():
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.side_effect = ConnectionClosedError(None, None)
        assert not client.is_ready()


def test_readiness_raises_a_final_refusal():
    """A retry does not change a permanent refusal, so a readiness poll does not wait out its deadline."""
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.probe.return_value = wire.Refusal.FINAL
        with pytest.raises(wire.ConnectRefused) as refused:
            client.is_ready()
    assert refused.value.refusal is wire.Refusal.FINAL


@pytest.mark.parametrize('refusal', [wire.Refusal.COLD, wire.Refusal.FORBIDDEN])
def test_readiness_of_a_refusal_a_retry_may_clear_is_false(refusal):
    client = roboarena.RoboarenaClient('a-partner-host', 8000)

    with patch.object(client, '_wire') as client_wire:
        client_wire.probe.return_value = refusal
        assert not client.is_ready()
