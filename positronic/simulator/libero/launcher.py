"""Launches the LIBERO env server as a subprocess and owns its lifetime.

positronic starts the server: the env runs in its own 3.10 interpreter via ``uv run --no-project env.py``,
which reads the script's PEP 723 deps and builds an isolated environment. The positronic-free ``env_server``
package and a LIBERO source checkout are placed on ``PYTHONPATH`` so ``env.py`` imports the dumb
``server``/``protocol`` and ``libero`` without dragging in positronic. A free localhost port is picked here and
handed to ``RemoteEnvControlSystem``; the subprocess is terminated when the run ends.
"""

import os
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pimm

_ENV_SCRIPT = Path(__file__).parent / 'env.py'
_ENV_SERVER_DIR = Path(__file__).parents[1] / 'env_server'

_LIBERO_REPO = 'https://github.com/Lifelong-Robot-Learning/LIBERO.git'
_LIBERO_COMMIT = '8f1084e3132a39270c3a13ebe37270a43ece2a01'
_LIBERO_CACHE = Path.home() / '.cache' / 'positronic' / 'libero'
_LIBERO_SRC = _LIBERO_CACHE / 'src'
_LIBERO_CONFIG = _LIBERO_CACHE / 'config'


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


def _ensure_libero_src() -> Path:
    """Clone LIBERO at the pinned commit into the cache and return its repo root for ``PYTHONPATH``.

    LIBERO declares ``install_requires=[]`` and ships ``libero`` as a PEP 420 namespace package (no top-level
    ``__init__.py``), so ``find_packages`` builds an empty wheel — it is importable only with the repo root on
    ``sys.path``, and the bddl/init-state/asset files the env reads live in the repo tree, not a wheel.
    """
    if not (_LIBERO_SRC / '.git').exists():
        _LIBERO_SRC.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', _LIBERO_REPO, str(_LIBERO_SRC)], check=True)
    head = subprocess.run(
        ['git', '-C', str(_LIBERO_SRC), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True
    ).stdout.strip()
    # Enforce the pin on every run, not just the first: a cache cloned earlier (or by hand) at another revision
    # would otherwise be imported as-is, mismatching the committed fixture and the controller assumptions.
    if head != _LIBERO_COMMIT:
        subprocess.run(['git', '-C', str(_LIBERO_SRC), 'fetch', 'origin', _LIBERO_COMMIT], check=True)
        subprocess.run(['git', '-C', str(_LIBERO_SRC), 'checkout', _LIBERO_COMMIT], check=True)
    return _LIBERO_SRC


def _spawn(
    suite: str, task_id: int, camera_resolution: int, control_mode: str, host: str, port: int
) -> subprocess.Popen:
    command = [
        'uv',
        'run',
        '--no-project',
        str(_ENV_SCRIPT),
        '--host',
        host,
        '--port',
        str(port),
        '--suite',
        suite,
        '--task-id',
        str(task_id),
        '--camera-resolution',
        str(camera_resolution),
        '--control-mode',
        control_mode,
    ]
    # LIBERO writes a config of repo-relative asset paths into ``LIBERO_CONFIG_PATH`` on first import; pin it beside
    # the checkout so it can never go stale against a ``~/.libero`` from an earlier, differently-located clone.
    env = {
        **os.environ,
        'PYTHONPATH': os.pathsep.join([str(_ENV_SERVER_DIR), str(_ensure_libero_src())]),
        'LIBERO_CONFIG_PATH': str(_LIBERO_CONFIG),
    }
    return subprocess.Popen(command, env=env)


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@contextmanager
def serve_libero(
    suite: str, task_id: int, camera_resolution: int, control_mode: str, host: str = 'localhost'
) -> Iterator[tuple[str, int]]:
    """Run the LIBERO env-server subprocess for the body's lifetime, yielding its ``(host, port)``.

    The synchronous counterpart of ``LiberoServer`` (which ties the subprocess to a pimm run loop): for a
    plain client that talks to the server over the socket without a World, e.g. the e2e demo replay.
    """
    port = _free_port()
    proc = _spawn(suite, task_id, camera_resolution, control_mode, host, port)
    try:
        yield host, port
    finally:
        _terminate(proc)


class LiberoServer(pimm.ControlSystem):
    """The LIBERO env-server subprocess: started on construction, killed when the run ends.

    It is a control system only so the World tears it down — it owns no ports and does no work each turn;
    ``RemoteEnvControlSystem`` does the talking over the socket at ``host``/``port``.
    """

    def __init__(self, suite: str, task_id: int, camera_resolution: int, control_mode: str, host: str = 'localhost'):
        self.host = host
        self.port = _free_port()
        self._proc = _spawn(suite, task_id, camera_resolution, control_mode, host, self.port)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        try:
            while not should_stop.value:
                yield pimm.Sleep(1.0)
        finally:
            _terminate(self._proc)
