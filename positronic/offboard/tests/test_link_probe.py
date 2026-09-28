import json
import logging
import os
import shutil
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from positronic.offboard import link_probe
from positronic.offboard.link_probe import (
    HEADER,
    READ_BYTES,
    _numeric_summary,
    _proc_queues,
    _queue_reader,
    _read_kernel_value,
    _serve_peer,
    _ss_queues,
    network_facts,
    receive_one,
    source,
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


class _ScriptedReceiver(socket.socket):
    """A socket whose ``recv`` returns the header, then ``chunks`` one per call."""

    def __init__(self, chunks: list[bytes]):
        super().__init__()
        self._pending = [struct.pack(HEADER, sum(len(chunk) for chunk in chunks)), *chunks]

    def recv(self, bufsize: int, flags: int = 0, /) -> bytes:
        return self._pending.pop(0)


def test_the_rate_counts_only_the_bytes_that_arrived_inside_the_span(monkeypatch):
    chunk = b'x' * READ_BYTES
    clock = iter([0, 10_000_000, 20_000_000])
    monkeypatch.setattr(link_probe, 'time', SimpleNamespace(perf_counter_ns=lambda: next(clock)))
    with _ScriptedReceiver([chunk, chunk, chunk]) as conn:
        report = receive_one(conn, READ_BYTES)
    assert report['read_span_ms'] == 20.0
    assert report['mib_per_sec'] == pytest.approx((2 * READ_BYTES / 2**20) / 0.020)


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


def _read_reply(conn: socket.socket) -> bytes:
    declared = struct.unpack(HEADER, conn.recv(struct.calcsize(HEADER), socket.MSG_WAITALL))[0]
    return conn.recv(declared, socket.MSG_WAITALL)


def test_a_transfer_that_lands_in_one_read_is_reported_and_the_sink_reads_on():
    """One read has no span and so no rate; the sink must report it and serve the next transfer."""
    sender, receiver = socket.socketpair()
    peer = threading.Thread(target=_serve_peer, args=(receiver, READ_BYTES, 0), daemon=True)
    peer.start()
    try:
        for _ in range(2):
            sender.sendall(_framed(b'x' * 100))
            report = json.loads(_read_reply(sender))
            assert report['bytes'] == 100
            assert report['reads'] == 1
            assert report['mib_per_sec'] is None
    finally:
        sender.close()
        peer.join(timeout=5.0)


@pytest.mark.parametrize(
    'sent',
    [struct.pack(HEADER, 0), struct.pack(HEADER, 100) + b'x' * 10],
    ids=['no bytes declared', 'closed inside a transfer'],
)
def test_a_peer_that_breaks_a_transfer_is_logged_as_an_error(sent, caplog):
    sender, receiver = socket.socketpair()
    sender.sendall(sent)
    sender.shutdown(socket.SHUT_WR)
    try:
        with caplog.at_level(logging.ERROR):
            _serve_peer(receiver, READ_BYTES, 0)
    finally:
        sender.close()
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_a_peer_that_closes_between_transfers_is_not_an_error(caplog):
    sender, receiver = socket.socketpair()
    sender.sendall(_framed(b'x' * 100))
    sender.shutdown(socket.SHUT_WR)
    try:
        with caplog.at_level(logging.ERROR):
            _serve_peer(receiver, READ_BYTES, 0)
        assert json.loads(_read_reply(sender))['bytes'] == 100
    finally:
        sender.close()
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


STALL_S = 0.2


def _source_against_a_sink(tmp_path: Path, kib: int) -> list[dict]:
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    peer = threading.Thread(target=lambda: _serve_peer(listener.accept()[0], READ_BYTES, 0), daemon=True)
    peer.start()
    out = tmp_path / 'rows.json'
    try:
        source.override(host='127.0.0.1', port=port, kib=kib, transfers=2, warmups=0, out=str(out)).instantiate()
    finally:
        peer.join(timeout=5.0)
        listener.close()
    return json.loads(out.read_text())


class _SlowToFrame(bytes):
    """A payload whose copy into a frame takes ``STALL_S``, so a timer around the copy shows it."""

    def __radd__(self, header: bytes) -> bytes:
        time.sleep(STALL_S)
        return header + bytes(self)


def test_write_ms_times_the_send_and_not_the_copy_into_the_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(link_probe.os, 'urandom', lambda count: _SlowToFrame(bytes(count)))
    rows = _source_against_a_sink(tmp_path, kib=64)
    assert [row['bytes'] for row in rows] == [64 * 1024] * 2
    assert all(row['write_ms'] < STALL_S * 1000 for row in rows)


def _slow_loads(raw: bytes):
    time.sleep(STALL_S)
    return json.loads(raw)


def test_report_ms_times_the_report_and_not_its_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(link_probe, 'json', SimpleNamespace(loads=_slow_loads, dumps=json.dumps))
    rows = _source_against_a_sink(tmp_path, kib=64)
    assert all(row['report_ms'] < STALL_S * 1000 for row in rows)


def _unread_connection(port_holder: list[int]) -> tuple[socket.socket, socket.socket, socket.socket]:
    """A connected pair whose server end is accepted and never read, so the bytes sit in its queue."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    port_holder.append(listener.getsockname()[1])
    client = socket.create_connection(('127.0.0.1', port_holder[0]))
    accepted, _ = listener.accept()
    return listener, client, accepted


@pytest.mark.skipif(shutil.which('ss') is None, reason='ss is not installed here')
def test_both_queue_readers_see_the_same_unread_bytes():
    """``ss`` prints a state column the readers index past; ``/proc/net/tcp`` answers without it."""
    held: list[int] = []
    listener, client, accepted = _unread_connection(held)
    try:
        client.sendall(b'q' * 1100)
        proc = _proc_queues(held[0])
        by_ss = _ss_queues(held[0])
        assert [row['recv_q'] for row in proc] == [1100]
        # The client's own socket has the port as its destination; only the server's end is read.
        assert [row['recv_q'] for row in by_ss] == [1100]
    finally:
        client.close()
        accepted.close()
        listener.close()


@pytest.mark.skipif(shutil.which('ss') is None, reason='ss is not installed here')
def test_a_socket_with_nothing_queued_is_still_reported():
    """An empty queue reports a blocked sender, so an empty read is not no read."""
    held: list[int] = []
    listener, client, accepted = _unread_connection(held)
    try:
        rows = _ss_queues(held[0])
        assert rows, 'an established socket on the port was not seen at all'
        assert all(row['recv_q'] == 0 for row in rows)
    finally:
        client.close()
        accepted.close()
        listener.close()


@pytest.mark.skipif(sys.platform != 'linux', reason='the namespace facts are read from /sys and /proc')
def test_the_namespace_reports_its_own_interfaces_and_buffers():
    reported = network_facts(peer=None)
    assert reported['net_namespace'].startswith('net:[')
    assert reported['interfaces']['lo']['mtu'] is not None
    assert reported['default_so_rcvbuf'] > 0
    json.dumps(reported)


@pytest.mark.skipif(sys.platform != 'linux', reason='the namespace facts are read from /sys and /proc')
def test_the_route_towards_a_peer_names_its_interface():
    reported = network_facts(peer='127.0.0.1')
    assert reported['local_address_towards_peer'] == '127.0.0.1'
    assert reported['interface_towards_peer'] == 'lo'
    assert reported['interfaces']['lo']['ipv4'] == '127.0.0.1'


def test_the_facts_refuse_a_system_without_proc_and_sys(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    with pytest.raises(OSError, match='run it on Linux'):
        network_facts(peer=None)


def test_the_summary_skips_a_column_that_is_not_a_number():
    rows = [
        {'write_ms': 1.0, 'read_timeline': [(0.0, 10)], 'local': 'a'},
        {'write_ms': 3.0, 'read_timeline': [(0.0, 10)], 'local': 'b'},
    ]
    summary = _numeric_summary(rows)
    assert 'write_ms' in summary
    assert 'read_timeline' not in summary and 'local' not in summary


def test_the_p95_of_twenty_transfers_is_not_their_maximum():
    rows = [{'write_ms': float(value)} for value in range(1, 21)]
    median, p95, maximum = (float(cell) for cell in _numeric_summary(rows).splitlines()[1].split()[1:])
    assert (median, p95, maximum) == (10.5, 19.1, 20.0)


def test_one_transfer_summarises_as_itself():
    assert _numeric_summary([{'write_ms': 4.0}]).splitlines()[1].split()[1:] == ['4.0', '4.0', '4.0']


def test_a_rate_that_one_transfer_lacks_summarises_over_the_transfers_that_have_it():
    rows = [{'mib_per_sec': 1.0}, {'mib_per_sec': None}, {'mib_per_sec': 3.0}]
    assert _numeric_summary(rows).splitlines()[1].split() == ['mib_per_sec', '2.0', '2.9', '3.0']


def test_a_rate_that_the_first_transfer_lacks_is_still_summarised():
    rows = [{'write_ms': 1.0, 'mib_per_sec': None}, {'write_ms': 2.0, 'mib_per_sec': 4.0}]
    assert _numeric_summary(rows).splitlines()[2].split() == ['mib_per_sec', '4.0', '4.0', '4.0']


def test_a_rate_that_no_transfer_has_is_not_summarised():
    rows = [{'write_ms': 1.0, 'mib_per_sec': None}, {'write_ms': 2.0, 'mib_per_sec': None}]
    assert 'mib_per_sec' not in _numeric_summary(rows)


def test_a_missing_ss_falls_back_to_the_kernel_table(monkeypatch, caplog):
    """A reader that cannot run must name itself at the start, not read as an empty queue per sample."""
    monkeypatch.setenv('PATH', '')
    with caplog.at_level(logging.ERROR):
        assert _queue_reader(9100) is _proc_queues
    assert any(record.levelno == logging.ERROR and 'ss refused' in record.message for record in caplog.records)


@pytest.mark.skipif(shutil.which('ss') is None, reason='ss is not installed here')
def test_ss_is_the_reader_where_it_answers():
    """The fallback must not be the path every run quietly takes."""
    assert _queue_reader(9100) is _ss_queues


def test_a_kernel_without_the_setting_reports_it_absent(tmp_path):
    """A setting this kernel does not carry is a legitimate absence, and the facts still print."""
    assert _read_kernel_value(tmp_path / 'no_such_setting') is None


@pytest.mark.skipif(os.geteuid() == 0, reason='root bypasses the permission bits this asserts on')
def test_a_setting_that_cannot_be_read_is_not_reported_absent(tmp_path):
    """A namespace refusing /proc/sys must raise; reported as null it reads as an absent setting."""
    refused = tmp_path / 'refused'
    refused.write_text('4096 131072 6291456\n')
    refused.chmod(0o000)
    with pytest.raises(PermissionError):
        _read_kernel_value(refused)
