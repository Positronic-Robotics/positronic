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

import numpy as np

from .core import Message, SignalEmitter, SignalReader, system_clock


class QueueEmitter(SignalEmitter):

    def __init__(self, queue: mp.Queue):
        self._queue = queue

    def emit(self, data: Any, ts: int | None = None) -> bool:
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


class SharedMemoryEmitter(SignalEmitter):
    def __init__(self, shared_memory_manager: mp.managers.SharedMemoryManager, lock: mp.Lock):
        self._shared_memory_manager = shared_memory_manager
        self._shared_memory = None
        self._shared_memory_array = None
        self._ts = None
        self._lock = lock

    def emit(self, data: Any, ts: int | None = None) -> bool:
        with self._lock:
            if self._shared_memory is None:
                assert isinstance(data, np.ndarray), "Only numpy arrays could be emitted to shared memory"
                self._shared_memory = self._shared_memory_manager.SharedMemory(size=data.nbytes)
                self._shared_memory_array = np.ndarray(data.shape, dtype=data.dtype, buffer=self._shared_memory.buf)
            self._shared_memory_array[:] = data
            self._ts = ts
        return True


class SharedMemoryReader(SignalReader):
    def __init__(self, emitter: SharedMemoryEmitter, lock: mp.Lock):
        self._emitter = emitter
        self._lock = lock
        self._non_writable_array = None

    def read(self) -> Message | None:
        with self._lock:
            if self._emitter._shared_memory_array is None:
                return None
            if self._non_writable_array is None:
                self._non_writable_array = np.ndarray(
                    shape=self._emitter._shared_memory_array.shape,
                    dtype=self._emitter._shared_memory_array.dtype,
                    buffer=self._emitter._shared_memory_array.data,
                )
                self._non_writable_array.flags.writeable = False

        return Message(data=self._non_writable_array, ts=self._emitter._ts)


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
        self._shared_memory_manager = mp.managers.SharedMemoryManager()

    def __enter__(self):
        self._shared_memory_manager.start()
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
    
    def fixed_size_pipe(self):
        lock = self._manager.Lock()
        emitter = SharedMemoryEmitter(self._shared_memory_manager, lock)
        reader = SharedMemoryReader(emitter, lock)
        return emitter, reader

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
