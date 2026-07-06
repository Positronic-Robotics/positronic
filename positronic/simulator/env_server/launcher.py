"""Shared machinery for benchmark env servers launched as subprocesses.

A benchmark's launcher keeps only what identifies it — the shape of its ``_spawn`` (PEP 723 script vs uv
project, its ``PYTHONPATH`` entries) and its pinned source checkout — and composes these pieces.
"""

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

    The pin is enforced on every run, not just the first: a cache cloned earlier (or by hand) at another
    revision would otherwise be imported as-is, mismatching committed fixtures and the assumptions pinned
    against the commit. With ``lfs``, ``git lfs pull`` also runs every time — a clone made without git-lfs
    installed left pointer stubs behind.
    """
    if lfs and shutil.which('git-lfs') is None:
        raise RuntimeError("git-lfs is required to fetch the checkout's LFS assets; install it and re-run")
    if not (dest / '.git').exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', repo_url, str(dest)], check=True)
    head = subprocess.run(
        ['git', '-C', str(dest), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != commit:
        subprocess.run(['git', '-C', str(dest), 'fetch', 'origin', commit], check=True)
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
