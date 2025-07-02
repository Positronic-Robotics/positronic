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

    def bind_to_buffer(self, buffer: Any, lock: mp.Lock) -> None:
        raise NotImplementedError()

    @classmethod
    def create_from_memoryview(cls, buffer: memoryview, lock: mp.Lock) -> None:
        raise NotImplementedError()


class ZeroCopySMEmitter(SignalEmitter):

    def __init__(self, data_type: type[SMCompliant], sm: mp.shared_memory.SharedMemory,
                 ts_value: mp.Value, lock: mp.Lock):
        self._sm = sm
        self._lock = lock
        self._ts_value = ts_value
        self._out_data = None
        self._data_type = data_type

    def emit(self, data: SMCompliant, ts: int = 0) -> bool:
        assert isinstance(data, self._data_type), f"Data type mismatch: {type(data)} != {self._data_type}"
        ts = ts or system_clock()

        if self._out_data is None:
            data.bind_to_buffer(self._sm.buf, self._lock)
            self._out_data = data
        else:
            assert data is self._out_data, "SMEmitter can only emit the same object multiple times"

        with self._lock:
            self._ts_value.value = ts
        return True

    def close(self):
        """Release references to shared memory to allow proper cleanup."""
        if self._out_data is not None:
            self._out_data._buffer = None
            self._out_data = None


class ZeroCopySMReader(SignalReader):

    def __init__(self, data_type: type[SMCompliant], sm: mp.shared_memory.SharedMemory,
                 ts_value: mp.Value, lock: mp.Lock):
        self._sm = sm
        self._ts_value = ts_value
        self._lock = lock
        self._data_type = data_type
        self._out_value = self._data_type.create_from_memoryview(sm.buf.toreadonly(), lock)

    def read(self) -> Message | None:
        with self._lock:
            if self._ts_value.value == 0:
                return None
            return Message(data=self._out_value, ts=self._ts_value.value)

    def close(self):
        """Release references to shared memory to allow proper cleanup."""
        if self._out_value is not None:
            self._out_value._buffer = None
            self._out_value = None


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
        self._shared_memories = []
        self._sm_emitters_readers = []

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

        for emitter, reader in self._sm_emitters_readers:
            emitter.close()
            reader.close()

        for sm in self._shared_memories:
            sm.close()
            sm.unlink()

    def pipe(self, maxsize: int = 0) -> Tuple[SignalEmitter, SignalReader]:
        """Create a queue-based communication channel between processes.

        Args:
            maxsize: Maximum queue size (0 for unlimited)

        Returns:
            Tuple of (emitter, reader) for inter-process communication
        """
        q = self._manager.Queue(maxsize=maxsize)
        return QueueEmitter(q), QueueReader(q)

    def zero_copy_sm(self, data_type: type[SMCompliant]) -> Tuple[SignalEmitter, SignalReader]:
        """Create a zero-copy shared memory channel for efficient data sharing.

        Args:
            data_type: SMCompliant type that defines the shared data structure

        Returns:
            Tuple of (emitter, reader) for zero-copy inter-process communication
        """
        sm = multiprocessing.shared_memory.SharedMemory(create=True, size=data_type.buf_size())
        self._shared_memories.append(sm)
        ts_value = self._manager.Value('Q', 0)
        lock = self._manager.Lock()
        emitter = ZeroCopySMEmitter(data_type, sm, ts_value, lock)
        reader = ZeroCopySMReader(data_type, sm, ts_value, lock)
        self._sm_emitters_readers.append((emitter, reader))
        return (emitter, reader)

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
