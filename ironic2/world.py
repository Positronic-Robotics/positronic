"""Implementation of multiprocessing channels."""

import logging
import multiprocessing as mp
import signal
import sys
from queue import Empty, Full
import traceback
from typing import Any, Callable, List, Tuple

from .core import Message, NoValue, NoValueType, SignalEmitter, SignalReader, system_clock


class QueueEmitter(SignalEmitter):

    def __init__(self, queue: mp.Queue):
        self._queue = queue

    def emit(self, data: Any, ts: int | None = None) -> bool:
        if self._queue.full():
            self._queue.get_nowait()
        try:
            self._queue.put_nowait(Message(data, ts))
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


class EventReader(SignalReader):

    def __init__(self, event: mp.Event):
        self._event = event

    def value(self) -> Message | NoValueType:
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

    def pipe(self) -> Tuple[SignalEmitter, SignalReader]:
        q = mp.Queue()
        return QueueEmitter(q), QueueReader(q)

    def start(self, *background_loops: List[Callable]):
        """Starts background control loops. Can be called multiple times for different control loops."""
        bg_processes: List[mp.Process] = []
        for bg_loop in background_loops:
            name = getattr(bg_loop, '__name__', 'anonymous')
            p = mp.Process(target=_bg_wrapper, args=(bg_loop, self._stop_event, name), daemon=True)
            p.start()
            bg_processes.append(p)

    def run(self, main_loop: Callable):
        """Optional utility function to run the main loop and handle Ctrl+C nicely.
           You must start background control loops that you depend on before calling this method.
        """

        def signal_handler(_signum, _frame):
            print("\nProgram interrupted by user, stopping...")
            self._stop_event.set()
            print("Stopping background processes...")
            for process in self.background_processes:
                process.join(timeout=3)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            main_loop(EventReader(self._stop_event))
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._stop_event.set()
        for process in self.background_processes:
            process.join(timeout=3)

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()
