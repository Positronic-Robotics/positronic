"""Shared machinery for benchmark env servers launched as subprocesses.

A benchmark's launcher keeps only what identifies it — the shape of its ``_spawn`` (PEP 723 script vs uv
project, its ``PYTHONPATH`` entries) and its pinned source checkout — and composes these pieces.
"""

import fcntl
import shutil
import socket
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


def ensure_pinned_checkout(repo_url: str, commit: str, dest: Path, *, lfs: bool = False) -> Path:
    """Clone ``repo_url`` into ``dest``, force it onto ``commit``, and return ``dest``.

    The pin is enforced on every call: a cache sitting at another revision, or at the pinned commit but with
    locally modified tracked files, would otherwise be imported as-is, mismatching committed fixtures and the
    assumptions pinned against the commit. With ``lfs``, ``git lfs pull`` runs each time too, since a clone made
    without git-lfs installed carries only pointer stubs. Callers sharing ``dest`` (a fan-out of eval jobs
    mounting one cache filesystem) serialize on a lock file beside it — concurrent forced checkouts of the same
    worktree would otherwise collide on ``.git/index.lock`` or reset files from under a starting process.
    """
    if lfs and shutil.which('git-lfs') is None:
        raise RuntimeError("git-lfs is required to fetch the checkout's LFS assets; install it and re-run")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest.parent / f'{dest.name}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (dest / '.git').exists():
            subprocess.run(['git', 'clone', repo_url, str(dest)], check=True)
        head = subprocess.run(
            ['git', '-C', str(dest), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True
        ).stdout.strip()
        if head != commit:
            subprocess.run(['git', '-C', str(dest), 'fetch', 'origin', commit], check=True)
        # Force onto the pin unconditionally: a cache already at the pinned commit but with locally modified
        # tracked files would otherwise import altered benchmark code or assets despite the pin.
        subprocess.run(['git', '-C', str(dest), 'checkout', '-f', commit], check=True)
        if lfs:
            subprocess.run(['git', '-C', str(dest), 'lfs', 'pull'], check=True)
    return dest


def terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@contextmanager
def serve_subprocess(spawn: Callable[[str, int], subprocess.Popen], host: str) -> Iterator[tuple[str, int]]:
    """Run an env-server subprocess for the body's lifetime, yielding its ``(host, port)``.

    The single owner of the subprocess: ``RemoteEnvControlSystem`` enters it to tie the subprocess to the
    World run, and a plain client (e.g. an e2e demo replay) enters it directly to talk over the socket without
    a World. The task spec rides the reset token, so the subprocess needs only its address — it serves
    whatever task the first reset asks for. The port is picked before the spawn; the client's connect retry
    covers the gap until the server binds it.

    TODO: a subprocess that dies at startup goes unnoticed until the client's connect deadline — nothing
    surfaces its exit during the retry wait.
    """
    port = free_port()
    proc = spawn(host, port)
    try:
        yield host, port
    finally:
        terminate(proc)
