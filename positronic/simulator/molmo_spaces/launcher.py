"""Launch MolmoSpaces in an isolated Python environment.

``MLSPACES_ASSETS_DIR`` must point to the asset packs. ``MUJOCO_GL`` defaults to EGL on Linux and CGL on macOS;
CPU rendering on Linux requires OSMesa or software EGL.
"""

import fcntl
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.launcher import ensure_pinned_checkout, serve_subprocess
from positronic.simulator.molmo_spaces import mapping

_ENV_SCRIPT = Path(__file__).parent / 'env.py'
_ENV_SERVER_DIR = Path(__file__).parents[1] / 'env_server'
_MAPPING_DIR = Path(__file__).parent

_MOLMO_REPO = 'https://github.com/allenai/molmospaces.git'
_MOLMO_COMMIT = 'c2f1b583f087e1d3994e1377574843b759d9d0f8'
_MOLMO_SRC = Path.home() / '.cache' / 'positronic' / 'molmospaces' / 'src'

# Version constraints compensate for MolmoSpaces' missing lockfile; the file documents how to regenerate them.
_MOLMO_CONSTRAINTS = Path(__file__).parent / 'molmo_constraints.txt'

# MolmoSpaces requires Python 3.11. Filament benchmarks need the alternative mujoco-filament extra.
_MOLMO_PYTHON = '3.11'
_MOLMO_EXTRA = 'mujoco'

# The isolated env server requires these independently of MolmoSpaces' dependencies.
_WIRE_DEPS = ('websockets>=15.0.1', 'msgpack')


@contextmanager
def _checkout_lock() -> Iterator[None]:
    """Prevent concurrent checkout and installation in the shared cache."""
    _MOLMO_SRC.parent.mkdir(parents=True, exist_ok=True)
    with open(_MOLMO_SRC.parent / 'setup.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def ensure_molmo_venv() -> Path:
    """Return the Python executable after preparing the pinned MolmoSpaces environment."""
    venv = _MOLMO_SRC / '.venv'
    with _checkout_lock():
        src = ensure_pinned_checkout(_MOLMO_REPO, _MOLMO_COMMIT, _MOLMO_SRC)
        if not venv.exists():
            subprocess.run(['uv', 'venv', '--python', _MOLMO_PYTHON, str(venv)], check=True)
        # uv sync also resolves upstream's unused curobo extra, which requires CUDA to build.
        subprocess.run(
            ['uv', 'pip', 'install', '-c', str(_MOLMO_CONSTRAINTS), '-e', f'.[{_MOLMO_EXTRA}]', *_WIRE_DEPS],
            cwd=str(src),
            env={**os.environ, 'VIRTUAL_ENV': str(venv)},
            check=True,
        )
    return venv / 'bin' / 'python'


_GL_BACKEND_ENV = 'MUJOCO_GL'


def molmo_subprocess_env() -> dict[str, str]:
    """Subprocess environment with the server's Python paths and GL backend."""
    return {
        **os.environ,
        'PYTHONPATH': os.pathsep.join([str(_ENV_SERVER_DIR), str(_MAPPING_DIR)]),
        _GL_BACKEND_ENV: os.environ.get(_GL_BACKEND_ENV, 'cgl' if sys.platform == 'darwin' else 'egl'),
    }


def _spawn(host: str, port: int) -> subprocess.Popen:
    if not os.environ.get(mapping.ASSETS_DIR_ENV):
        raise ValueError(f'{mapping.ASSETS_DIR_ENV} must point at the MolmoSpaces asset packs')
    python = ensure_molmo_venv()
    command = [str(python), str(_ENV_SCRIPT), protocol.OPT_HOST, host, protocol.OPT_PORT, str(port)]
    return subprocess.Popen(command, env=molmo_subprocess_env())


def serve_molmo_spaces(host: str = 'localhost') -> AbstractContextManager[tuple[str, int]]:
    """Run a MolmoSpaces server for the context's lifetime and yield its address."""
    return serve_subprocess(_spawn, host)
