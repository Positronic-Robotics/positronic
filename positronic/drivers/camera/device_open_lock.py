import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Fixed and absolute: openers resolving different paths -- a differing `TMPDIR`, a `PrivateTmp=` unit -- take
# different locks and serialize nothing. `/run/lock` is tmpfs, so no ownership survives a reboot.
DEVICE_OPEN_LOCK_PATH = Path('/run/lock/positronic-camera-open.lock')

# `flock` needs a descriptor, not write access, so every account needs read and nothing more.
_LOCK_FILE_MODE = 0o666


def _install_lock_file(path: Path) -> None:
    """Link the lock file into place already carrying its mode."""
    fd, staged = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        # Moded before it is reachable: `umask` masks a create mode, so a create killed before a later
        # `fchmod` would strand a file no other account can open.
        os.fchmod(fd, _LOCK_FILE_MODE)
        try:
            os.link(staged, path)
        except FileExistsError:
            pass  # lost the race; `flock` binds the inode, so the winner's file is the lock
    finally:
        os.close(fd)
        os.unlink(staged)


def _open_for_locking(path: Path) -> int:
    """Open the lock file read-only, installing it the first time."""
    # Never `O_CREAT` on a file that exists: `fs.protected_regular` refuses it when another account owns the
    # file in a world-writable sticky directory, which is every home a host-global lock has.
    try:
        return os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        _install_lock_file(path)
    return os.open(path, os.O_RDONLY)


@contextmanager
def device_open_lock(path: Path = DEVICE_OPEN_LOCK_PATH) -> Iterator[None]:
    """Hold exclusive access to camera device enumeration.

    Two SDKs enumerating the bus at the same instant lose the devices and report them as undetected.
    """
    fd = _open_for_locking(path)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)
