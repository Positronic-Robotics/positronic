"""Implementation of multiprocessing channels."""

import logging
import multiprocessing as mp
import multiprocessing.shared_memory
import multiprocessing.managers
import sys
from queue import Empty, Full
import time
import traceback
from typing import Any, Callable, List, Tuple

from .core import Message, SignalEmitter, SignalReader, system_clock


class QueueEmitter(SignalEmitter):

    def __init__(self, queue: mp.Queue):
        self._queue = queue

    def emit(self, data: Any, ts: int = 0) -> bool:
        try:
            self._queue.put_nowait(Message(data, ts))
            return True
        except Full:
            # Queue is full, try to remove old message and try again
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(Message(data, ts))
                return True
            except (Empty, Full):
                return False


class QueueReader(SignalReader):

    def __init__(self, queue: mp.Queue):
        self._queue = queue
        self._last_value = None

    def read(self) -> Message | None:
        try:
            self._last_value = self._queue.get_nowait()
        except Empty:
            pass
        return self._last_value


class SMCompliant:
    @classmethod
    def buf_size(cls) -> int:
        return 0

    def bind_to_buffer(self, buffer: memoryview, lock: mp.Lock) -> None:
        raise NotImplementedError()

    def bind_from_buffer(self, buffer: memoryview, lock: mp.Lock) -> None:
        raise NotImplementedError()


class SMEmitter(SignalEmitter):
    def __init__(self, sm_name: str, ts_value: mp.Value, lock: mp.Lock):
        self._sm = None
        self._sm_name = sm_name
        self._lock = lock
        self._ts_value = ts_value
        self._last_value = None

    def emit(self, data: SMCompliant, ts: int = 0) -> bool:
        ts = ts or system_clock()

        if self._sm is None:
            self._sm = mp.SharedMemory(name=self._sm_name, create=True, size=self.buf_size())
            self.bind_to_buffer(self._sm.buf, self._lock)

        if self._last_value is None:
            self._last_value = data
        else:
            assert data == self._last_value, "SMEmitter can only emit the same object multiple times"

        with self._lock:
            self._ts_value.value = ts
        return True


class SMReader(SignalReader):
    def __init__(self, sm_name: str, ts_value: mp.Value, lock: mp.Lock):
        self._sm = None
        self._sm_name = sm_name
        self._ts_value = ts_value
        self._lock = lock

    def read(self) -> Message | None:
        if self._sm is None:
            self._sm = mp.SharedMemory(name=self._sm_name, create=False)
            self.bind_from_buffer(self._sm.buf, self._lock)

        with self._lock:
            if self._ts_value.value == 0:
                return None
            return Message(data=self._last_value, ts=self._ts_value.value)


class EventReader(SignalReader):

    def __init__(self, event: mp.Event):
        self._event = event

    def read(self) -> Message | None:
        return Message(data=self._event.is_set(), ts=system_clock())


def _bg_wrapper(run_func: Callable, stop_event: mp.Event, name: str):
    try:
        run_func(EventReader(stop_event))
    except KeyboardInterrupt:
        # Silently handle KeyboardInterrupt in background processes
        pass
    except Exception:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ERROR in background process '{name}':", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        logging.error(f"Error in control system {name}:\n{traceback.format_exc()}")
    finally:
        stop_event.set()


class World:
    """Utility class to bind and run control loops."""

    def __init__(self):
        # TODO: stop_signal should be a shared variable, since we should be able to track if background
        # processes are still running
        self._stop_event = mp.Event()
        self.background_processes = []
        self._manager = mp.Manager()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Stopping background processes...", flush=True)
        self._stop_event.set()
        time.sleep(0.1)

        print(f"Waiting for {len(self.background_processes)} background processes to terminate...", flush=True)
        for process in self.background_processes:
            process.join(timeout=3)
            if process.is_alive():
                print(f'Process {process.name} (pid {process.pid}) did not respond, terminating...', flush=True)
                process.terminate()
                process.join(timeout=2)  # Give it a moment to terminate
                if process.is_alive():
                    print(f'Process {process.name} (pid {process.pid}) still alive, killing...', flush=True)
                    process.kill()
            print(f'Process {process.name} (pid {process.pid}) finished', flush=True)
            process.close()

    def pipe(self, maxsize: int = 0) -> Tuple[SignalEmitter, SignalReader]:
        q = self._manager.Queue(maxsize=maxsize)
        return QueueEmitter(q), QueueReader(q)

    def start(self, *background_loops: List[Callable]):
        """Starts background control loops. Can be called multiple times for different control loops."""
        for bg_loop in background_loops:
            if hasattr(bg_loop, '__self__'):
                name = f"{bg_loop.__self__.__class__.__name__}.{bg_loop.__name__}"
            else:
                name = getattr(bg_loop, '__name__', 'anonymous')
            p = mp.Process(target=_bg_wrapper, args=(bg_loop, self._stop_event, name), daemon=True, name=name)
            p.start()
            self.background_processes.append(p)
            print(f"Started background process {name} (pid {p.pid})", flush=True)

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()
