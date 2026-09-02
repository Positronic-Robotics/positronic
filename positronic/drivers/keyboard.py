import logging
import select
import sys
import termios
import tty
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pimm

logger = logging.getLogger(__name__)


def _pressed_key() -> str | None:
    """The key just typed, or ``None`` where nobody typed one. It never waits."""
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    return sys.stdin.read(1) if ready else None


@contextmanager
def key_reader() -> Iterator[Callable[[], str | None] | None]:
    """Put the terminal in cbreak mode, and yield a reader of one key press at a time.

    Yields ``None`` where stdin is not a terminal: there is no key to read, and what a run without an
    operator does is the caller's to decide.
    """
    if not sys.stdin.isatty():
        yield None
        return
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield _pressed_key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class KeyboardControl(pimm.ControlSystem):
    """Reads the terminal a key at a time and emits each one. ``quit_key`` returns, which stops the world.

    ``_each_round`` is what a key does. A subclass answers it to do something else, and gets it every round,
    key or no key, so work of its own between presses has somewhere to run.
    """

    def __init__(self, quit_key: str | None = None):
        self.quit_key = quit_key
        self.keyboard_inputs = pimm.ControlSystemEmitter[str](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        with key_reader() as read_key:
            if read_key is None:
                logger.warning('no key can be read: stdin is not a terminal')
                return
            while not should_stop.value:
                key = read_key()
                if key is not None and key == self.quit_key:
                    return
                self._each_round(key)
                yield pimm.Sleep(0.01)

    def _each_round(self, key: str | None) -> None:
        if key is not None:
            self.keyboard_inputs.emit(key)
