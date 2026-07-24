"""Launches the MolmoSpaces env server as a subprocess and owns its lifetime.

positronic starts the server: the env runs in MolmoSpaces' own interpreter — a per-checkout ``.venv`` with the
``molmospaces[mujoco]`` stack (mujoco ~=3.5, the resource-manager asset layer, torch) installed into it, far too
heavy and Python-version-pinned (3.11) to share positronic's venv. The positronic-free ``env_server`` package and
this package's ``mapping`` module ride ``PYTHONPATH`` so ``env.py`` imports the dumb ``server``/``protocol`` and
the pure wire mappings without dragging in positronic; ``molmo_spaces`` resolves from the venv.

MolmoSpaces renders MuJoCo scenes, so the server needs a GL backend (``MUJOCO_GL``) and its asset packs
(``MLSPACES_ASSETS_DIR``): ``MUJOCO_GL`` defaults to ``egl`` (GPU) here and both env vars pass through from the
caller, so a GPU-less box can override ``MUJOCO_GL=osmesa`` for CPU software rendering.
"""

import fcntl
import os
import subprocess
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from positronic.simulator.env_server.launcher import ensure_pinned_checkout, serve_subprocess

_ENV_SCRIPT = Path(__file__).parent / 'env.py'
_ENV_SERVER_DIR = Path(__file__).parents[1] / 'env_server'
_MAPPING_DIR = Path(__file__).parent  # ``mapping.py`` — imported flat by env.py, positronic-free

_MOLMO_REPO = 'https://github.com/allenai/molmospaces.git'
_MOLMO_COMMIT = 'c2f1b583f087e1d3994e1377574843b759d9d0f8'
_MOLMO_SRC = Path.home() / '.cache' / 'positronic' / 'molmospaces' / 'src'

# MolmoSpaces pins Python 3.11 and installs its MuJoCo renderer stack via the ``mujoco`` extra (classic renderer,
# mujoco ~=3.5). ``mujoco-filament`` is the alternative for bench-v2 filament scenes; the classic renderer is the
# eval default.
_MOLMO_PYTHON = '3.11'
_MOLMO_EXTRA = 'mujoco'


@contextmanager
def _checkout_lock() -> Iterator[None]:
    """Serialize checkout + ``uv sync`` across processes sharing the cache, so a warm-cache fan-out of eval jobs
    mounting one ``~/.cache/positronic/molmospaces`` filesystem does not race a forced checkout against a sync."""
    _MOLMO_SRC.parent.mkdir(parents=True, exist_ok=True)
    with open(_MOLMO_SRC.parent / 'setup.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _spawn(host: str, port: int, benchmark_dir: str) -> subprocess.Popen:
    venv = _MOLMO_SRC / '.venv'
    with _checkout_lock():
        src = ensure_pinned_checkout(_MOLMO_REPO, _MOLMO_COMMIT, _MOLMO_SRC)
        # Install the stack before spawning: a cold first install far exceeds the client's connect deadline,
        # which should only cover the sim's boot. Install the ``mujoco`` extra explicitly into a venv the way
        # MolmoSpaces' own image does, rather than ``uv sync`` — which also resolves the ``curobo`` extra, a CUDA
        # build that needs a GPU toolchain and is not on the eval task path. Both steps are idempotent and fast
        # when warm. MolmoSpaces ships no uv.lock, so the install re-resolves on every fresh box.
        if not venv.exists():
            subprocess.run(['uv', 'venv', '--python', _MOLMO_PYTHON, str(venv)], check=True)
        subprocess.run(
            ['uv', 'pip', 'install', '-e', f'.[{_MOLMO_EXTRA}]'],
            cwd=str(src),
            env={**os.environ, 'VIRTUAL_ENV': str(venv)},
            check=True,
        )
    command = [
        str(venv / 'bin' / 'python'),
        str(_ENV_SCRIPT),
        '--host',
        host,
        '--port',
        str(port),
        '--benchmark_dir',
        str(benchmark_dir),
    ]
    env = {
        **os.environ,
        'PYTHONPATH': os.pathsep.join([str(_ENV_SERVER_DIR), str(_MAPPING_DIR)]),
        # GPU OpenGL by default; a GPU-less box exports MUJOCO_GL=osmesa, or relies on mesa's software EGL, for
        # CPU rendering.
        'MUJOCO_GL': os.environ.get('MUJOCO_GL', 'egl'),
    }
    return subprocess.Popen(command, env=env)


def serve_molmo_spaces(benchmark_dir: str, host: str = 'localhost') -> AbstractContextManager[tuple[str, int]]:
    """The MolmoSpaces env server as a ``serve`` context manager (the ``serve_subprocess`` contract).

    ``benchmark_dir`` (a dir holding ``benchmark.json``) is fixed for the run; the reset token selects the
    episode within it, so one task-agnostic server serves every trial.
    """
    return serve_subprocess(lambda host, port: _spawn(host, port, benchmark_dir), host)
