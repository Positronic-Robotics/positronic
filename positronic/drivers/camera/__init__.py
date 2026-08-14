import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# A file rather than a multiprocessing primitive because the openers are separate processes that share no
# parent state, and `flock` is dropped by the kernel when its holder exits, so a camera killed mid-open
# cannot strand it. Absolute and fixed rather than read from the environment, because openers that resolve
# different paths -- a differing `TMPDIR`, a unit with `PrivateTmp=` -- take different locks and serialize
# nothing between them. `/run/lock` is tmpfs, so no ownership survives a reboot.
DEVICE_OPEN_LOCK_PATH = Path('/run/lock/positronic-camera-open.lock')

# `flock` needs an open descriptor and no write access, so every account on the host needs read and nothing
# more.
_LOCK_FILE_MODE = 0o666


def _open_for_locking(path: Path) -> int:
    """Open the lock file read-only, creating it readable by every account the first time.

    A file already there is opened without `O_CREAT`, because `fs.protected_regular` refuses an
    `O_CREAT` open of a file owned by neither the opener nor the directory, and every directory a
    host-global lock can live in is world-writable and sticky.
    """
    try:
        return os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        pass
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_EXCL, _LOCK_FILE_MODE)
    except FileExistsError:
        return os.open(path, os.O_RDONLY)  # another opener created it between the two opens
    os.fchmod(fd, _LOCK_FILE_MODE)  # `umask` masks the create mode, and every account needs read
    return fd


@contextmanager
def device_open_lock(path: Path = DEVICE_OPEN_LOCK_PATH) -> Iterator[None]:
    """Hold exclusive access to camera device enumeration for the duration of the block.

    Two SDKs enumerating the bus at the same instant lose the devices and report them as undetected.
    """
    fd = _open_for_locking(path)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)
