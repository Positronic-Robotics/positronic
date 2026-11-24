import itertools
import os
import threading
import time
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler

import numpy as np
import pytest

from positronic.policy.image_worker import ImageWorker, ImageWorkerError

pytestmark = pytest.mark.timeout(120)


def _echo(tree, extra):
    return {'pid': os.getpid(), 'tree': tree, 'extra': extra, 'bytes': b'\x01' * 1000}


@pytest.fixture(scope='module')
def worker():
    with ImageWorker() as worker:
        yield worker


def test_a_call_runs_in_another_process_and_returns_its_arrays_and_bytes(worker):
    frames = np.random.default_rng(0).integers(0, 256, (4, 60, 80, 3), dtype=np.uint8)
    tree = {'image': frames, 'state': [np.arange(3.0), 2.5], 'task': 'pick', 'empty': np.zeros((0, 3))}

    answer = worker.call(_echo, tree, 'extra')

    assert answer['pid'] != os.getpid()
    np.testing.assert_array_equal(answer['tree']['image'], frames)
    np.testing.assert_array_equal(answer['tree']['state'][0], np.arange(3.0))
    assert answer['tree']['state'][1] == 2.5
    assert answer['tree']['task'] == 'pick'
    assert answer['tree']['empty'].shape == (0, 3)
    assert answer['extra'] == 'extra'
    assert answer['bytes'] == b'\x01' * 1000


def test_the_arrays_go_through_shared_memory_and_the_pipe_carries_kilobytes(worker, monkeypatch):
    pickled_sizes = []
    send, recv = Connection.send, Connection.recv

    def counted_send(conn, obj):
        pickled_sizes.append(len(ForkingPickler.dumps(obj)))
        send(conn, obj)

    def counted_recv(conn):
        obj = recv(conn)
        pickled_sizes.append(len(ForkingPickler.dumps(obj)))
        return obj

    monkeypatch.setattr(Connection, 'send', counted_send)
    monkeypatch.setattr(Connection, 'recv', counted_recv)
    frames = np.ones((25, 600, 960, 3), dtype=np.uint8)

    answer = worker.call(_echo, {'image': frames}, None)

    np.testing.assert_array_equal(answer['tree']['image'], frames)
    assert max(pickled_sizes) < 4096 < frames.nbytes


def test_a_larger_call_after_a_smaller_one_returns_all_of_its_arrays(worker):
    for size in (8, 512, 16):
        frames = np.full((3, size, size, 3), size % 251, dtype=np.uint8)
        np.testing.assert_array_equal(worker.call(_echo, {'image': frames}, None)['tree']['image'], frames)


def _fail(tree):
    raise ValueError(f'no frames in {sorted(tree)}')


def test_an_error_in_the_function_reaches_the_caller_with_its_type_and_message(worker):
    with pytest.raises(ValueError, match=r"no frames in \['image'\]"):
        worker.call(_fail, {'image': np.zeros((2, 2, 3), dtype=np.uint8)})
    assert worker.call(_echo, {}, None)['extra'] is None


def _end_process(tree):
    os._exit(3)


def test_a_process_that_ends_fails_the_call_and_the_next_call_starts_a_new_process():
    with ImageWorker() as worker:
        first = worker.call(_echo, {}, None)['pid']
        with pytest.raises(ImageWorkerError, match='ended'):
            worker.call(_end_process, {'image': np.zeros((4, 4, 3), dtype=np.uint8)})
        assert worker.call(_echo, {}, None)['pid'] not in (first, os.getpid())


def _sleep(tree):
    time.sleep(60)


def test_a_process_that_does_not_answer_in_time_fails_the_call_and_the_next_call_starts_a_new_process():
    with ImageWorker(timeout_sec=0.5) as worker:
        first = worker.call(_echo, {}, None)['pid']
        started = time.monotonic()
        with pytest.raises(ImageWorkerError, match='did not answer a call in 0.5 s'):
            worker.call(_sleep, {})
        assert time.monotonic() - started < 10
        assert worker.call(_echo, {}, None)['pid'] not in (first, os.getpid())


def _hold_interpreter_lock(tree):
    started = time.monotonic()
    sum(range(tree['count']))  # One C call: it keeps the interpreter lock until it returns.
    return time.monotonic() - started


def test_a_loop_in_this_process_keeps_its_ticks_while_a_call_holds_the_interpreter_lock(worker):
    worker.call(_echo, {}, None)
    held: list[float] = []
    caller = threading.Thread(target=lambda: held.append(worker.call(_hold_interpreter_lock, {'count': 60_000_000})))
    ticks = [time.monotonic()]
    caller.start()
    while caller.is_alive():
        time.sleep(0.005)
        ticks.append(time.monotonic())
    caller.join()

    assert held[0] > 0.2
    assert max(b - a for a, b in itertools.pairwise(ticks)) < held[0] / 2
