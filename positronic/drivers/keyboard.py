import select
import sys
import termios
import tty
from collections.abc import Callable, Iterator
from contextlib import contextmanager


def _pressed_key() -> str | None:
    """The key just typed, or ``None`` when nobody typed one. It never waits."""
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    return sys.stdin.read(1) if ready else None


@contextmanager
def key_reader() -> Iterator[Callable[[], str | None] | None]:
    """Put the terminal in cbreak mode, and yield a reader of one key press at a time.

    Yields ``None`` where stdin is not a terminal: there is nothing to read a key from, and the caller
    decides what a run without an operator does.
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
