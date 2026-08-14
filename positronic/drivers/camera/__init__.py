import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# A file rather than a multiprocessing primitive because the openers are separate processes that share no
# parent state, and `flock` is dropped by the kernel when its holder exits, so a camera killed mid-open
# cannot strand it.
DEVICE_OPEN_LOCK_PATH = Path(tempfile.gettempdir()) / 'positronic-camera-open.lock'


@contextmanager
def device_open_lock(path: Path = DEVICE_OPEN_LOCK_PATH) -> Iterator[None]:
    """Hold exclusive access to camera device enumeration for the duration of the block.

    Two SDKs enumerating the bus at the same instant lose the devices and report them as undetected.
    """
    # Read-only, so a lock file another account created is still openable; `flock` needs no write access.
    fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)
