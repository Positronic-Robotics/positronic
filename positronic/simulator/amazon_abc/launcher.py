"""Launches the ABC env server as a subprocess and owns its lifetime.

positronic starts the server: the env runs in an interpreter of its own, built once beside the pinned ABC
checkout, because ABC needs MuJoCo 3.8 where positronic locks 3.5. The positronic-free ``env_server`` package,
this package and the checkout ride ``PYTHONPATH`` so ``env.py`` imports the dumb ``server``/``protocol``, the
shared ``mapping``, and ``abc_sim`` without dragging in positronic.
"""

import fcntl
import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from pathlib import Path

from positronic.simulator.env_server.launcher import ensure_pinned_checkout, serve_subprocess

_ENV_SCRIPT = Path(__file__).parent / 'env.py'
_ENV_SERVER_DIR = Path(__file__).parents[1] / 'env_server'
_MAPPING_DIR = Path(__file__).parent

_ABC_REPO = 'https://github.com/amazon-far/abc.git'
_ABC_COMMIT = '6c467cebcecf16a4dce79e6fd87a7ca2281c3ef0'
_ABC_CACHE = Path.home() / '.cache' / 'positronic' / 'abc'
_ABC_SRC = _ABC_CACHE / 'src'

# ABC declares ``requires-python = ">=3.10"``, so uv would otherwise inherit positronic's interpreter.
_ABC_PYTHON = '3.12'
# What the simulator and the asset installer import, at ABC's own bounds. Installing the project instead would
# pull its CUDA torch and mujoco-warp, which only the policy and the batched renderer need.
_ABC_DEPS = ('mujoco~=3.8.0', 'gymnasium>=1.1', 'numpy', 'tyro')
# The isolated env server requires these independently of ABC's dependencies.
_WIRE_DEPS = ('websockets>=15.0.1', 'msgpack')

# ABC's own asset installer: it resolves a task to the packages its scene loads, and verifies each archive
# against the manifest it ships. Assets are untracked, so forcing the checkout onto the pin leaves them.
_PREPARE_SCRIPT = 'prepare.py'
_PREPARE_ONE_TASK = '--sim-task'
_PREPARE_EVERY_TASK = '--sim'


@contextmanager
def _checkout_lock() -> Iterator[None]:
    """Prevent concurrent checkout, installation and asset download in the shared cache."""
    _ABC_CACHE.mkdir(parents=True, exist_ok=True)
    with open(_ABC_CACHE / 'setup.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def ensure_abc(tasks: Sequence[str] | None) -> Path:
    """Return the Python executable of the prepared ABC environment, with ``tasks``' assets installed.

    ``tasks`` of ``None`` installs every asset package, which is what a sweep over the whole catalogue needs.
    """
    venv = _ABC_SRC / '.venv'
    with _checkout_lock():
        src = ensure_pinned_checkout(_ABC_REPO, _ABC_COMMIT, _ABC_SRC)
        if not venv.exists():
            subprocess.run(['uv', 'venv', '--python', _ABC_PYTHON, str(venv)], check=True)
        python = venv / 'bin' / 'python'
        subprocess.run(
            ['uv', 'pip', 'install', *_ABC_DEPS, *_WIRE_DEPS], env={**os.environ, 'VIRTUAL_ENV': str(venv)}, check=True
        )
        assets = [_PREPARE_EVERY_TASK] if tasks is None else [_PREPARE_ONE_TASK, *tasks]
        subprocess.run([str(python), _PREPARE_SCRIPT, *assets], cwd=str(src), check=True)
    return python


_GL_BACKEND_ENV = 'MUJOCO_GL'
PYTHONPATH_ENV = 'PYTHONPATH'


def abc_subprocess_env() -> dict[str, str]:
    """Subprocess environment with the server's Python paths and GL backend."""
    return {
        **os.environ,
        PYTHONPATH_ENV: os.pathsep.join([str(_ENV_SERVER_DIR), str(_MAPPING_DIR), str(_ABC_SRC)]),
        _GL_BACKEND_ENV: os.environ.get(_GL_BACKEND_ENV, 'cgl' if sys.platform == 'darwin' else 'egl'),
    }


def _spawn(host: str, port: int, tasks: Sequence[str] | None) -> subprocess.Popen:
    python = ensure_abc(tasks)
    command = [str(python), str(_ENV_SCRIPT), '--host', host, '--port', str(port)]
    return subprocess.Popen(command, env=abc_subprocess_env())


def serve_abc(tasks: Sequence[str] | None, host: str = 'localhost') -> AbstractContextManager[tuple[str, int]]:
    """The ABC env server as a ``serve`` context manager (the ``serve_subprocess`` contract).

    ``tasks`` names the tasks whose assets the server needs; the scene each one builds is drawn from them.
    """
    return serve_subprocess(partial(_spawn, tasks=tasks), host)
