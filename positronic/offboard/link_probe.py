"""Where the bytes of one transfer go, timed at both ends of the link.

A session's round trip divides into what the server reports spending and what it does not. This
measures the rest: the link, and whether the receiver drains it. It carries no policy and no model,
so it answers before a model is loaded, and it runs wherever the receiver runs — a container's
network namespace is not its host's.

``sink`` reads, and reports the read: when the first byte landed, when the last one did, and every
read between them. ``source`` writes, and reports how long its own write took to return — on a
websocket that is the uplink, and a write outlasting its own bytes means the far end is not
draining. ``watch`` samples a socket's receive queue while either runs, which is what tells a slow
path from a late reader. ``facts`` reports the MTU, the namespace and the buffer sizes of wherever
it runs.

Usage
    # In the container, before anything else is up:
    uv run --locked python -m positronic.offboard.link_probe sink --port=9100
    uv run --locked python -m positronic.offboard.link_probe watch --port=9100 --out=recvq.jsonl
    uv run --locked python -m positronic.offboard.link_probe facts

    # From the client, against the sink above and again against one on the host:
    uv run --locked python -m positronic.offboard.link_probe source \\
        --host=<host> --port=9100 --kib=750 --transfers=5
"""

import json
import os
import socket
import statistics
import struct
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
from configuronic.cli import CommandTree

from pimm.logging import init_logging

# One transfer is a length, that many bytes, then the sink's report under a length of its own. Both
# ends read this module, so the header is written once.
HEADER = '!Q'
HEADER_BYTES = struct.calcsize(HEADER)

# What one ``recv`` asks for. asyncio's selector transport reads 256 KiB at a time; a sink reading the
# same amount sees the same number of wakeups for the same payload.
READ_BYTES = 256 * 1024


def _send_framed(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(struct.pack(HEADER, len(payload)) + payload)


def _read_exactly(conn: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = conn.recv(min(remaining, READ_BYTES))
        if not chunk:
            raise ConnectionError(f'the peer closed with {remaining} of {count} bytes still owed')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def receive_one(conn: socket.socket, read_bytes: int) -> dict[str, Any]:
    """Read one transfer, and report the read: the span the bytes took, and every ``recv`` inside it.

    ``reads`` is ``[offset_ms, size]`` per call, offset from the first byte. Evenly spaced reads are a
    path delivering slowly; a gap and then a burst is a receiver that was not scheduled.
    """
    declared = struct.unpack(HEADER, _read_exactly(conn, HEADER_BYTES))[0]
    if declared == 0:
        raise ValueError('the peer declared a transfer of no bytes; there is no read to time')
    reads: list[tuple[float, int]] = []
    received = 0
    first_ns = 0
    while received < declared:
        chunk = conn.recv(min(declared - received, read_bytes))
        now = time.perf_counter_ns()
        if not chunk:
            raise ConnectionError(f'the peer closed with {declared - received} of {declared} bytes still owed')
        if not reads:
            first_ns = now
        reads.append(((now - first_ns) / 1e6, len(chunk)))
        received += len(chunk)
    sizes = [size for _, size in reads]
    span_ms = reads[-1][0]
    return {
        'bytes': received,
        'read_span_ms': span_ms,
        'reads': len(reads),
        'read_bytes_median': statistics.median(sizes),
        'read_bytes_max': max(sizes),
        'mib_per_sec': (received / 2**20) / (span_ms / 1000.0) if span_ms > 0 else None,
        'read_timeline': reads,
    }


class GilHog:
    """Threads spinning in Python, so the reader competes for the interpreter while it reads.

    A CUDA call or a busy model holds the GIL the same way. ``threads`` of 0 raises none.
    """

    def __init__(self, threads: int):
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._spin, daemon=True) for _ in range(threads)]
        for thread in self._threads:
            thread.start()

    def _spin(self) -> None:
        counter = 0
        while not self._stop.is_set():
            counter += 1

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)


def _serve_peer(conn: socket.socket, read_bytes: int, busy_threads: int) -> None:
    """Read transfers off one connection until the peer goes, reporting each back to its sender."""
    try:
        while True:
            report = receive_one(conn, read_bytes)
            report['read_bytes'] = read_bytes
            report['busy_threads'] = busy_threads
            _send_framed(conn, json.dumps(report).encode())
            print(
                f'  {report["bytes"] / 1024:.0f} KiB in {report["read_span_ms"]:.1f} ms over '
                f'{report["reads"]} read(s), {report["mib_per_sec"]:.1f} MiB/s',
                flush=True,
            )
    except (ConnectionError, OSError, ValueError) as e:
        print(f'  peer gone: {e}', flush=True)
    finally:
        conn.close()


@cfn.config(host='0.0.0.0', port=9100, read_bytes=READ_BYTES, busy_threads=0)
def sink(host: str, port: int, read_bytes: int, busy_threads: int):
    """Read transfers and report each one. Runs where the receiver runs, and answers one peer at a time.

    ``busy_threads`` raises that many Python threads spinning beside the read, so a run says whether a
    receiver competing for the interpreter is enough to stall a read on its own.
    """
    hog = GilHog(busy_threads)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)
    print(f'sink on {host}:{port}, reading {read_bytes} at a time, {busy_threads} busy thread(s)', flush=True)
    try:
        while True:
            conn, peer = listener.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f'peer {peer[0]}:{peer[1]}', flush=True)
            _serve_peer(conn, read_bytes, busy_threads)
    except KeyboardInterrupt:
        print('sink stopped', flush=True)
    finally:
        hog.close()
        listener.close()


def _incompressible(count: int) -> bytes:
    """``count`` bytes nothing along the path can shrink, so the wire carries what the caller asked for."""
    return os.urandom(count)


def _numeric_summary(rows: list[dict[str, Any]]) -> str:
    """Median, p95 and max of every numeric column in ``rows``, one column per line."""
    columns = [name for name, value in rows[0].items() if isinstance(value, int | float)]
    width = max(len(name) for name in columns)
    lines = [f'{"":<{width}}  {"median":>10}  {"p95":>10}  {"max":>10}']
    for name in columns:
        values = sorted(float(row[name]) for row in rows)
        p95 = values[min(int(0.95 * len(values)), len(values) - 1)]
        lines.append(f'{name:<{width}}  {statistics.median(values):10.1f}  {p95:10.1f}  {max(values):10.1f}')
    return '\n'.join(lines)


@cfn.config(host='127.0.0.1', port=9100, kib=750, transfers=5, warmups=1, out=None)
def source(host: str, port: int, kib: int, transfers: int, warmups: int, out: str | None):
    """Write ``kib`` to a sink ``transfers`` times on one connection, and report both ends of each.

    ``write_ms`` is this end's own: the time ``sendall`` took to return. On a websocket that is the
    uplink span a session reports, so a ``write_ms`` far above what the link needs for ``kib`` says the
    far end did not drain it. Every figure beside it comes back from the sink.
    """
    payload = _incompressible(kib * 1024)
    rows = []
    with socket.create_connection((host, port), timeout=300.0) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f'source -> {host}:{port}, {kib} KiB x {transfers} (+{warmups} warm-up)', flush=True)
        for attempt in range(warmups + transfers):
            started = time.perf_counter_ns()
            _send_framed(conn, payload)
            written = time.perf_counter_ns()
            reported = json.loads(_read_exactly(conn, struct.unpack(HEADER, _read_exactly(conn, HEADER_BYTES))[0]))
            answered = time.perf_counter_ns()
            if attempt < warmups:
                continue
            timeline = reported.pop('read_timeline')
            rows.append({
                'write_ms': (written - started) / 1e6,
                'report_ms': (answered - written) / 1e6,
                **reported,
                'read_timeline': timeline,
            })
    print('\n' + _numeric_summary(rows))
    if out is not None:
        Path(out).write_text(json.dumps(rows, indent=1))
        print(f'\nper-transfer rows -> {out}')


def _proc_queues(port: int) -> list[dict[str, Any]]:
    """Every established socket on ``port``, with its queues, from ``/proc/net/tcp``.

    The kernel's own table, so it answers in an image that carries no tooling at all.
    """
    lines = Path('/proc/net/tcp').read_text().splitlines()[1:]
    try:
        lines += Path('/proc/net/tcp6').read_text().splitlines()[1:]
    except FileNotFoundError:
        # A namespace with IPv6 off carries no tcp6 table, and then the tcp one is the whole answer.
        pass
    rows = []
    for line in lines:
        fields = line.split()
        local, state, queues = fields[1], fields[3], fields[4]
        # `01` is ESTABLISHED; a listener's queues count backlog, not bytes.
        if state != '01' or int(local.rsplit(':', 1)[1], 16) != port:
            continue
        send_q, recv_q = (int(part, 16) for part in queues.split(':'))
        rows.append({'local': local, 'recv_q': recv_q, 'send_q': send_q})
    return rows


def _ss_queues(port: int) -> list[dict[str, Any]]:
    """The same queues through ``ss -tim``, with the TCP info it prints under each socket.

    Raises when ``ss`` is absent or refuses, carrying what it said.
    """
    done = subprocess.run(
        ['ss', '-tim', f'( sport = :{port} or dport = :{port} )'],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=True,
    )
    rows = []
    for line in done.stdout.splitlines()[1:]:
        fields = line.split()
        if line[:1].isspace():
            # The indented line under a socket: `cubic wscale:7,7 rtt:3.5/1.7 bytes_received:786432 ...`
            info = dict(field.split(':', 1) for field in fields if ':' in field)
            if rows:
                rows[-1].update({key: info[key] for key in ('rtt', 'cwnd', 'bytes_received', 'retrans') if key in info})
        # `State Recv-Q Send-Q Local Peer`. A `state established` filter would drop the first column and
        # shift every index here, so the state is filtered on rather than asked for.
        elif len(fields) >= 5 and fields[0] == 'ESTAB':
            rows.append({'local': fields[3], 'peer': fields[4], 'recv_q': int(fields[1]), 'send_q': int(fields[2])})
    return rows


def _queue_reader(port: int) -> Callable[[int], list[dict[str, Any]]]:
    """``ss`` where it answers, and the kernel's own table where it does not.

    Chosen once, by running ``ss``, and the refusal is printed. A failure after the choice raises
    rather than reading as an empty queue for that sample.
    """
    try:
        _ss_queues(port)
    except (OSError, subprocess.SubprocessError) as refused:
        print(f'ss refused ({refused}); reading /proc/net/tcp instead', flush=True)
        return _proc_queues
    print('reading queues through ss', flush=True)
    return _ss_queues


@cfn.config(port=9100, interval_ms=20, seconds=60.0, out=None)
def watch(port: int, interval_ms: int, seconds: float, out: str | None):
    """Sample the receive queue of every established socket on ``port``, for ``seconds``.

    This is what tells a slow path from a late reader. Bytes piling up in ``recv_q`` while the sender
    is still writing mean the receiver is not draining them; a queue that stays near empty while the
    sender blocks means the bytes are not arriving. Run it in the receiver's namespace, against the
    sink's port or the policy server's.
    """
    reader = _queue_reader(port)
    print(f'watching port {port} every {interval_ms} ms for {seconds:.0f}s', flush=True)
    samples: list[dict[str, Any]] = []
    started = time.perf_counter_ns()
    deadline = started + int(seconds * 1e9)
    while time.perf_counter_ns() < deadline:
        rows = reader(port)
        at_ms = (time.perf_counter_ns() - started) / 1e6
        samples.extend({'at_ms': at_ms, **row} for row in rows)
        time.sleep(interval_ms / 1000.0)
    if not samples:
        print(f'no established socket on port {port} in the whole window', flush=True)
        return
    queues = [sample['recv_q'] for sample in samples]
    busy = [sample for sample in samples if sample['recv_q'] > 0]
    # What one round cost, not what was asked for: reading through `ss` spawns a process per sample, so
    # `interval_ms` is a floor. A queue that empties faster than this cadence is invisible here.
    steps = [b['at_ms'] - a['at_ms'] for a, b in zip(samples, samples[1:], strict=False) if b['at_ms'] > a['at_ms']]
    print(
        f'{len(samples)} sample(s): recv_q max {max(queues)} B, median {statistics.median(queues):.0f} B, '
        f'non-empty in {len(busy)} of them; sampled every {statistics.median(steps) if steps else 0:.1f} ms'
    )
    if out is not None:
        Path(out).write_text('\n'.join(json.dumps(sample) for sample in samples) + '\n')
        print(f'per-sample rows -> {out}')


def _sysctl(path: str) -> str | None:
    """One kernel setting, or ``None`` where this kernel has no such file.

    Any other read failure raises: a namespace that refuses ``/proc/sys`` is a fact about the
    namespace, and this is the tool that reports those.
    """
    try:
        return Path(path).read_text().strip()
    except FileNotFoundError:
        return None


def network_facts(peer: str | None) -> dict[str, Any]:
    """What this namespace does to a transfer: its interfaces, its buffers, its congestion control.

    A container has a namespace of its own, so none of this is the host's. The MTU is the figure to
    read first: an overlay or a tunnel carries less than an ethernet link, and a sender that does not
    learn it retransmits its way through every transfer.
    """
    interfaces = {}
    for entry in sorted(Path('/sys/class/net').iterdir()):
        interfaces[entry.name] = {'mtu': _sysctl(f'{entry}/mtu'), 'operstate': _sysctl(f'{entry}/operstate')}
    facts: dict[str, Any] = {
        'net_namespace': Path('/proc/self/ns/net').readlink().name,
        'interfaces': interfaces,
        'tcp_rmem': _sysctl('/proc/sys/net/ipv4/tcp_rmem'),
        'tcp_wmem': _sysctl('/proc/sys/net/ipv4/tcp_wmem'),
        'rmem_max': _sysctl('/proc/sys/net/core/rmem_max'),
        'wmem_max': _sysctl('/proc/sys/net/core/wmem_max'),
        'tcp_congestion_control': _sysctl('/proc/sys/net/ipv4/tcp_congestion_control'),
        'tcp_slow_start_after_idle': _sysctl('/proc/sys/net/ipv4/tcp_slow_start_after_idle'),
    }
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        facts['default_so_rcvbuf'] = probe.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        facts['default_so_sndbuf'] = probe.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    if peer is not None:
        # A connect-less UDP socket takes the route the kernel would use, naming the interface a
        # transfer to ``peer`` leaves by — and so the MTU above that actually applies.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect((peer, 9))
            facts['local_address_towards_peer'] = route.getsockname()[0]
    return facts


@cfn.config(peer=None)
def facts(peer: str | None):
    """Print what this namespace does to a transfer. ``peer`` names a host to report the route towards."""
    print(json.dumps(network_facts(peer), indent=1))


COMMANDS: CommandTree = {'sink': sink, 'source': source, 'watch': watch, 'facts': facts}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
