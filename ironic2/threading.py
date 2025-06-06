import logging
import sys
import threading
import traceback
from typing import Callable, List, Tuple

from ironic2 import is_true

from .core import Message, NoValue, NoValueType, SignalEmitter, SignalReader, SharedSignal, system_clock


class _SharedEvent(SharedSignal):
    def __init__(self, event: threading.Event):
        self._event = event

    def emit(self, message: Message) -> bool:
        self._event.set()
        return True

    def value(self) -> Message | NoValueType:
        return Message(data=self._event.is_set(), ts=system_clock())

class SharedVariable(SharedSignal):
    def __init__(self, value):
        self._value = value

    def emit(self, value):
        self._value = value

    def value(self):
        return self._value


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

class ThreadWorld:

    def __init__(self, stop_signal: SharedSignal | None = None):
        if stop_signal is None:
            self._should_stop = _SharedEvent(threading.Event())
        else:
            self._should_stop = stop_signal
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
        self._should_stop.emit(Message(True, system_clock()))
        for thread in self.background_threads:
            thread.join(timeout=3)

    @property
    def should_stop(self) -> bool:
        return is_true(self._should_stop.value())