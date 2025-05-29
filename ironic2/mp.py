"""Implementation of multiprocessing channels."""

import logging
import multiprocessing as mp
import signal
import sys
from multiprocessing import Queue
from queue import Empty, Full
import traceback
from types import SimpleNamespace
from typing import Callable

from ironic2.channel import (CommunicationProvider, Message, NoValue, NoValueType, SignalEmitter, SignalReader,
                             system_clock)


class _EventReader(SignalReader):

    def __init__(self, event: mp.Event):
        self._event = event

    def value(self) -> Message | NoValueType:
        print('+' if self._event.is_set() else '-', end='', flush=True)
        return Message(data=self._event.is_set(), ts=system_clock())


class _Queue:

    class Emitter(SignalEmitter):

        def __init__(self, queue: '_Queue'):
            self._queue = queue

        def emit(self, message: Message) -> bool:
            if self._queue._queue.full():
                self._queue._queue.get_nowait()
            try:
                self._queue._queue.put_nowait(message)
                return True
            except Full:
                return False

    class Reader(SignalReader):

        def __init__(self, queue: '_Queue'):
            self._queue = queue
            self._last_value = NoValue

        def value(self) -> Message | NoValueType:
            try:
                self._last_value = self._queue._queue.get_nowait()
            except Empty:
                pass
            return self._last_value

    def __init__(self, max_size: int = 1):
        self._queue = Queue(max_size)

    def new_emitter(self) -> Emitter:
        return _Queue.Emitter(self)

    def new_reader(self) -> Reader:
        return _Queue.Reader(self)


class _Provider(CommunicationProvider):

    def __init__(self, stop_event: mp.Event):
        self.interface = SimpleNamespace()
        self.stop_event = stop_event

    def emitter(self, name: str, **kwargs) -> SignalEmitter:
        if kwargs:
            logging.warning(f"Ignoring kwargs for {name}: {kwargs}")

        q = _Queue()
        setattr(self.interface, name, q.new_reader())
        return q.new_emitter()

    def reader(self, name: str, **kwargs) -> SignalReader:
        if kwargs:
            logging.warning(f"Ignoring kwargs for {name}: {kwargs}")

        q = _Queue()
        setattr(self.interface, name, q.new_reader())
        return q.new_reader()

    def should_stop(self) -> SignalReader:
        return _EventReader(self.stop_event)

    def interface(self) -> SimpleNamespace:
        return self.interface


def _bg_wrapper(run_func: Callable, stop_event: mp.Event, name: str):
    try:
        run_func()
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
        stop_event.set()


class MPWorld:

    def __init__(self):
        self.stopped = mp.Event()
        self.background_processes = []

    def new_provider(self) -> _Provider:
        return _Provider(self.stopped)

    def add_background_control_system(self, control_system, *args, **kwargs) -> SimpleNamespace:
        provider = self.new_provider()
        cs = control_system(provider, *args, **kwargs)

        self.background_processes.append(
            mp.Process(target=_bg_wrapper, args=(cs.run, self.stopped, control_system.__name__), daemon=True))
        return provider.interface

    def run(self, main_loop):

        def signal_handler(_signum, _frame):
            print("\nProgram interrupted by user, stopping...")
            self.stopped.set()
            print("Stopping background processes...")
            for process in self.background_processes:
                process.join(timeout=3)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        for process in self.background_processes:
            process.start()
        try:
            main_loop(_EventReader(self.stopped))
        except KeyboardInterrupt:
            print("\nProgram interrupted by user, stopping...")
        finally:
            self.stopped.set()
            for process in self.background_processes:
                process.join(timeout=3)
