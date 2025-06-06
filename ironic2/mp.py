"""Implementation of multiprocessing channels."""

import logging
import multiprocessing as mp
import sys
from queue import Empty, Full
import threading
import traceback
from typing import Callable, List, Tuple

from .core import Message, NoValue, NoValueType, SignalEmitter, SignalReader, system_clock, SharedSignal, is_true


class _SharedEvent(SharedSignal):
    def __init__(self, event: mp.Event):
        self._event = event

    def emit(self, message: Message) -> bool:
        self._event.set()
        return True

    def value(self) -> Message | NoValueType:
        return Message(data=self._event.is_set(), ts=system_clock())


class QueueEmitter(SignalEmitter):

    def __init__(self, queue: mp.Queue):
        self._queue = queue

    def emit(self, message: Message) -> bool:
        if self._queue.full():
            self._queue.get_nowait()
        try:
            self._queue.put_nowait(message)
            return True
        except Full:
            return False


class QueueReader(SignalReader):

    def __init__(self, queue: mp.Queue):
        self._queue = queue
        self._last_value = NoValue

    def value(self) -> Message | NoValueType:
        try:
            self._last_value = self._queue.get_nowait()
        except Empty:
            pass
        return self._last_value


class SharedQueue(SharedSignal):
    def __init__(self, queue: mp.Queue):
        self._queue = queue
        self._last_value = NoValue

    def emit(self, message: Message) -> bool:
        if self._queue.full():
            self._queue.get_nowait()
        try:
            self._queue.put_nowait(message)
            return True
        except Full:
            return False

    def value(self) -> Message | NoValueType:
        return self._last_value


def _bg_wrapper(run_func: Callable, should_stop: SharedSignal, name: str):
    try:
        run_func(should_stop)
    except KeyboardInterrupt:
        # Silently handle KeyboardInterrupt in background processes
        pass
    except Exception:
        # Print to stderr for immediate visibility
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ERROR in background process '{name}':", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        logging.error(f"Error in control system {name}:\n{traceback.format_exc()}")
        should_stop.emit(Message(True, system_clock()))


class MPWorld:

    def __init__(self, stop_signal: SharedSignal | None = None):
        # TODO: stop_signal should be a shared variable, since we should be able to track if background
        # processes are still running
        if stop_signal is None:
            event = mp.Event()
            self._should_stop = _SharedEvent(event)
        else:
            self._should_stop = stop_signal
        self.background_processes = []

    def pipe(self) -> Tuple[SignalEmitter, SignalReader]:
        q = mp.Queue()
        return QueueEmitter(q), QueueReader(q)

    def run(self, *background_loops: List[Callable]):
        bg_processes: List[mp.Process] = []
        for bg_loop in background_loops:
            name = getattr(bg_loop, '__name__', 'anonymous')
            p = mp.Process(target=_bg_wrapper, args=(bg_loop, self._should_stop, name), daemon=True)
            p.start()
            bg_processes.append(p)

    def stop(self):
        self._should_stop.emit(Message(True, system_clock()))
        for process in self.background_processes:
            process.join(timeout=3)

    @property
    def should_stop(self) -> bool:
        return is_true(self._should_stop)


class SharedVariable(SignalReader, SignalEmitter):
    def __init__(self, value):
        self._value = value

    def emit(self, value):
        self._value = value

    def value(self):
        return self._value


class ThreadWorld:

    def __init__(self,):
        self._should_stop = threading.Event()
        self.background_threads = []

    def pipe(self) -> Tuple[SignalEmitter, SignalReader]:
        var = SharedVariable(NoValue)
        return var, var

    def run(self, *background_loops: List[Callable]):

        bg_threads: List[threading.Thread] = []
        for bg_loop in background_loops:
            name = getattr(bg_loop, '__name__', 'anonymous')
            t = threading.Thread(target=_bg_wrapper, args=(bg_loop, self._should_stop, name), daemon=True)
            t.start()
            bg_threads.append(t)

    def stop(self):
        self._should_stop.set()
        for thread in self.background_threads:
            thread.join(timeout=3)