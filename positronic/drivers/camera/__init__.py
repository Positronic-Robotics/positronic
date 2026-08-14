import fcntl
import os
import tempfile
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


def _install_lock_file(path: Path) -> None:
    """Link the lock file into place already readable by every account.

    The mode is set before the file is reachable under `path` -- `umask` masks a requested create
    mode, and a process killed between the create and a later `fchmod` would leave a file no other
    account can open until the next reboot. `mkstemp` takes its directory from `path`, not from the
    environment.
    """
    fd, staged = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        os.fchmod(fd, _LOCK_FILE_MODE)
        try:
            os.link(staged, path)
        except FileExistsError:
            pass  # another opener linked one first; `flock` binds the inode, so theirs is the lock
    finally:
        os.close(fd)
        os.unlink(staged)


def _open_for_locking(path: Path) -> int:
    """Open the lock file read-only, installing it the first time.

    A file already there is opened without `O_CREAT`, because `fs.protected_regular` refuses an
    `O_CREAT` open of a file owned by neither the opener nor the directory, and every directory a
    host-global lock can live in is world-writable and sticky.
    """
    try:
        return os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        _install_lock_file(path)
    return os.open(path, os.O_RDONLY)


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
