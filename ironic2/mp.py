"""Implementation of multiprocessing channels."""

import logging
import multiprocessing as mp
import signal
import sys
from queue import Empty, Full
import traceback
from typing import Callable, List, Tuple

from ironic2.ironic2 import (Message, NoValue, NoValueType, SignalEmitter, SignalReader, system_clock)


class _EventReader(SignalReader):

    def __init__(self, event: mp.Event):
        self._event = event

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


def _bg_wrapper(run_func: Callable, should_stop: mp.Event, name: str):
    try:
        run_func(_EventReader(should_stop))
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
        should_stop.set()


class MPWorld:

    def __init__(self):
        self._should_stop = mp.Event()
        self.background_processes = []

    def pipe(self) -> Tuple[SignalEmitter, SignalReader]:
        q = mp.Queue()
        return QueueEmitter(q), QueueReader(q)

    def run(self, main_loop: Callable, *background_loops: List[Callable]):

        def signal_handler(_signum, _frame):
            print("\nProgram interrupted by user, stopping...")
            self._should_stop.set()
            print("Stopping background processes...")
            for process in self.background_processes:
                process.join(timeout=3)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        bg_processes: List[mp.Process] = []
        for bg_loop in background_loops:
            p = mp.Process(target=_bg_wrapper, args=(bg_loop, self._should_stop, bg_loop.__name__), daemon=True)
            p.start()
            bg_processes.append(p)
        try:
            main_loop(_EventReader(self._should_stop))
        except KeyboardInterrupt:
            print("\nProgram interrupted by user, stopping...")
        finally:
            self._should_stop.set()
            for process in bg_processes:
                process.join(timeout=3)
