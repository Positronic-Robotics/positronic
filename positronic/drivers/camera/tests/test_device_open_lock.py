"""Camera opens are serialized across processes, because a vendor SDK loses the device when two overlap."""

import multiprocessing as mp
import time
from contextlib import nullcontext
from pathlib import Path

from positronic.drivers.camera import device_open_lock

# Spawn, not fork: a forked child inherits the parent's open fds, and with them the lock.
CTX = mp.get_context('spawn')

SUCCESS = 'SUCCESS'
CAMERA_NOT_DETECTED = 'CAMERA NOT DETECTED'
OPEN_SECONDS = 0.2
PATIENCE_SECONDS = 30.0


def _open(in_flight) -> str:
    """Stands in for `sl.Camera.open`: it loses the device whenever a second open overlaps it."""
    with in_flight.get_lock():
        in_flight.value += 1
        overlapped = in_flight.value > 1
    time.sleep(OPEN_SECONDS)
    with in_flight.get_lock():
        in_flight.value -= 1
    return CAMERA_NOT_DETECTED if overlapped else SUCCESS


def _open_racing(barrier, in_flight, results, lock_path: Path | None) -> None:
    barrier.wait(timeout=PATIENCE_SECONDS)
    with nullcontext() if lock_path is None else device_open_lock(lock_path):
        results.put(_open(in_flight))


def _open_at_the_same_instant(lock_path: Path | None, cameras: int = 2) -> list[str]:
    barrier = CTX.Barrier(cameras)
    in_flight = CTX.Value('i', 0)
    results = CTX.Queue()
    procs = [CTX.Process(target=_open_racing, args=(barrier, in_flight, results, lock_path)) for _ in range(cameras)]
    for p in procs:
        p.start()
    try:
        return [results.get(timeout=PATIENCE_SECONDS) for _ in range(cameras)]
    finally:
        for p in procs:
            p.join(timeout=PATIENCE_SECONDS)


def _hold_the_lock(lock_path: Path, holding) -> None:
    with device_open_lock(lock_path):
        holding.set()
        time.sleep(PATIENCE_SECONDS)


def _another_process_acquires_within(lock_path: Path, seconds: float) -> bool:
    acquired = CTX.Event()
    p = CTX.Process(target=_hold_the_lock, args=(lock_path, acquired))
    p.start()
    try:
        return acquired.wait(timeout=seconds)
    finally:
        p.kill()
        p.join(timeout=PATIENCE_SECONDS)


def test_two_opens_at_the_same_instant_lose_the_device():
    """Pins that `_open` reproduces the SDK, so the serialization test below cannot pass vacuously."""
    assert CAMERA_NOT_DETECTED in _open_at_the_same_instant(lock_path=None)


def test_the_lock_lets_only_one_open_run_at_a_time(tmp_path: Path):
    assert _open_at_the_same_instant(tmp_path / 'open.lock') == [SUCCESS, SUCCESS]


def test_a_live_holder_keeps_another_process_out(tmp_path: Path):
    lock_path = tmp_path / 'open.lock'
    holding = CTX.Event()
    holder = CTX.Process(target=_hold_the_lock, args=(lock_path, holding))
    holder.start()
    try:
        assert holding.wait(timeout=PATIENCE_SECONDS)
        assert not _another_process_acquires_within(lock_path, seconds=2.0)
    finally:
        holder.kill()
        holder.join(timeout=PATIENCE_SECONDS)


def test_a_killed_holder_does_not_strand_the_lock(tmp_path: Path):
    """A camera killed mid-open must not lock the bus for every later launch."""
    lock_path = tmp_path / 'open.lock'
    holding = CTX.Event()
    holder = CTX.Process(target=_hold_the_lock, args=(lock_path, holding))
    holder.start()
    assert holding.wait(timeout=PATIENCE_SECONDS)
    holder.kill()
    holder.join(timeout=PATIENCE_SECONDS)

    assert _another_process_acquires_within(lock_path, seconds=PATIENCE_SECONDS)
