import json
import shutil
import socket
import struct
import threading

import pytest

from positronic.offboard.link_probe import (
    HEADER,
    READ_BYTES,
    _numeric_summary,
    _proc_queues,
    _ss_queues,
    network_facts,
    receive_one,
)


def _framed(payload: bytes) -> bytes:
    return struct.pack(HEADER, len(payload)) + payload


def test_a_read_is_reported_as_its_bytes_and_the_calls_it_took():
    sender, receiver = socket.socketpair()
    payload = b'x' * (300 * 1024)
    threading.Thread(target=sender.sendall, args=(_framed(payload),), daemon=True).start()
    try:
        report = receive_one(receiver, READ_BYTES)
    finally:
        sender.close()
        receiver.close()

    assert report['bytes'] == len(payload)
    assert report['reads'] == len(report['read_timeline'])
    assert sum(size for _, size in report['read_timeline']) == len(payload)
    assert report['read_span_ms'] >= 0.0


def test_a_transfer_of_no_bytes_is_refused_rather_than_timed():
    """There is no first byte to stamp, so the read has no span; say so instead of failing on an index."""
    sender, receiver = socket.socketpair()
    sender.sendall(struct.pack(HEADER, 0))
    try:
        with pytest.raises(ValueError, match='no bytes'):
            receive_one(receiver, READ_BYTES)
    finally:
        sender.close()
        receiver.close()


def _unread_connection(port_holder: list[int]) -> tuple[socket.socket, socket.socket, socket.socket]:
    """A connected pair whose server end is accepted and never read, so the bytes sit in its queue."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    port_holder.append(listener.getsockname()[1])
    client = socket.create_connection(('127.0.0.1', port_holder[0]))
    accepted, _ = listener.accept()
    return listener, client, accepted


def test_both_queue_readers_see_the_same_unread_bytes():
    """``ss`` prints a state column the readers index past; ``/proc/net/tcp`` is what answers without it."""
    held: list[int] = []
    listener, client, accepted = _unread_connection(held)
    try:
        client.sendall(b'q' * 1100)
        proc = _proc_queues(held[0])
        by_ss = _ss_queues(held[0])
        assert [row['recv_q'] for row in proc] == [1100]
        if by_ss is None:
            pytest.skip('ss is not installed here')
        assert any(row['recv_q'] == 1100 for row in by_ss)
    finally:
        client.close()
        accepted.close()
        listener.close()


@pytest.mark.skipif(shutil.which('ss') is None, reason='ss is not installed here')
def test_a_socket_with_nothing_queued_is_still_reported():
    """A queue that stays empty is the finding when a sender is blocked, so an empty read is not no read."""
    held: list[int] = []
    listener, client, accepted = _unread_connection(held)
    try:
        rows = _ss_queues(held[0])
        assert rows is not None and rows, 'an established socket on the port was not seen at all'
        assert all(row['recv_q'] == 0 for row in rows)
    finally:
        client.close()
        accepted.close()
        listener.close()


def test_the_namespace_reports_its_own_interfaces_and_buffers():
    reported = network_facts(peer=None)
    assert reported['net_namespace'].startswith('net:[')
    assert reported['interfaces']['lo']['mtu'] is not None
    assert reported['default_so_rcvbuf'] > 0
    json.dumps(reported)


def test_the_summary_skips_a_column_that_is_not_a_number():
    rows = [
        {'write_ms': 1.0, 'read_timeline': [(0.0, 10)], 'local': 'a'},
        {'write_ms': 3.0, 'read_timeline': [(0.0, 10)], 'local': 'b'},
    ]
    summary = _numeric_summary(rows)
    assert 'write_ms' in summary
    assert 'read_timeline' not in summary and 'local' not in summary
