"""Implementation of multiprocessing channels."""

import asyncio
import logging
import multiprocessing as mp
import sys
import traceback
from queue import Empty, Full
from typing import Any, Callable, List, Tuple

from .core import Clock, Message, SignalEmitter, SignalReader


class QueueEmitter(SignalEmitter):

    def __init__(self, queue: mp.Queue, clock: Clock):
        self._queue = queue
        self._clock = clock

    def emit(self, data: Any, ts: int | None = None) -> bool:
        if ts is None:
            ts = self._clock.now_ns()
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


class EventReader(SignalReader):

    def __init__(self, event: mp.Event, clock: Clock):
        self._event = event
        self._clock = clock

    def read(self) -> Message | None:
        return Message(data=self._event.is_set(), ts=self._clock.now_ns())


def _bg_wrapper(run_func: Callable, stop_event: mp.Event, name: str, clock: Clock):
    try:
        if asyncio.iscoroutinefunction(run_func):
            asyncio.run(run_func(EventReader(stop_event, clock), clock))
        else:
            run_func(EventReader(stop_event, clock), clock)
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

    def __init__(self, clock: Clock):
        # TODO: stop_signal should be a shared variable, since we should be able to track if background
        # processes are still running
        self._stop_event = mp.Event()
        self.background_processes = []
        self._manager = mp.Manager()
        self._clock = clock

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Stopping background processes...", flush=True)
        self._stop_event.set()

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
        return QueueEmitter(q, self._clock), QueueReader(q)

    def start(self, *background_loops: List[Callable]):
        """Starts background control loops. Can be called multiple times for different control loops."""
        for bg_loop in background_loops:
            if hasattr(bg_loop, '__self__'):
                name = f"{bg_loop.__self__.__class__.__name__}.{bg_loop.__name__}"
            else:
                name = getattr(bg_loop, '__name__', 'anonymous')
            p = mp.Process(target=_bg_wrapper,
                           args=(bg_loop, self._stop_event, name, self._clock),
                           daemon=True,
                           name=name)
            p.start()
            self.background_processes.append(p)
            print(f"Started background process {name} (pid {p.pid})", flush=True)

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()
