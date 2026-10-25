"""Launches the ABC env server as a subprocess and owns its lifetime.

The server runs in an interpreter of its own: ABC needs MuJoCo 3.8 and positronic locks 3.5.
"""

import subprocess
from collections.abc import Sequence
from contextlib import AbstractContextManager
from functools import partial
from pathlib import Path

from positronic.simulator.env_server.launcher import (
    SERVER_DEPS,
    ensure_pinned_checkout,
    ensure_venv,
    mujoco_server_env,
    serve_subprocess,
    setup_lock,
    uv_pip_install,
)

_ENV_SCRIPT = Path(__file__).parent / 'env.py'
_ENV_SERVER_DIR = Path(__file__).parents[1] / 'env_server'
_MAPPING_DIR = Path(__file__).parent
_APERTURE_DIR = Path(__file__).parents[2] / 'drivers' / 'gripper'

_ABC_REPO = 'https://github.com/amazon-far/abc.git'
_ABC_COMMIT = '6c467cebcecf16a4dce79e6fd87a7ca2281c3ef0'
_ABC_CACHE = Path.home() / '.cache' / 'positronic' / 'abc'
_ABC_SRC = _ABC_CACHE / 'src'

# ABC declares ``requires-python = ">=3.10"``, so uv would otherwise inherit positronic's interpreter.
_ABC_PYTHON = '3.12'
# Installing the ABC project itself would pull its CUDA torch and mujoco-warp.
_ABC_DEPS = ('mujoco~=3.8.0', 'gymnasium>=1.1', 'numpy', 'tyro')
# ABC's contact-rich tasks score differently on another MuJoCo, so every version is pinned.
_ABC_CONSTRAINTS = Path(__file__).parent / 'abc_constraints.txt'

_PREPARE_SCRIPT = Path('prepare.py')
_PREPARE_ONE_TASK = '--sim-task'
_PREPARE_EVERY_TASK = '--sim'
# `--sim` leaves out the RoboCasa packs that inhand_transfer needs; `--sim-task` adds them for that task.
_PREPARE_ROBOCASA = '--sim-robocasa'


def ensure_abc(tasks: Sequence[str] | None) -> Path:
    """The Python executable of the prepared ABC environment; ``None`` tasks installs every asset package."""
    with setup_lock(_ABC_CACHE):
        src = ensure_pinned_checkout(_ABC_REPO, _ABC_COMMIT, _ABC_SRC)
        venv = _ABC_SRC / '.venv'
        python = ensure_venv(venv, _ABC_PYTHON)
        uv_pip_install(venv, ['-c', str(_ABC_CONSTRAINTS), *_ABC_DEPS, *SERVER_DEPS])
        assets = [_PREPARE_EVERY_TASK, _PREPARE_ROBOCASA] if tasks is None else [_PREPARE_ONE_TASK, *tasks]
        subprocess.run([str(python), str(_PREPARE_SCRIPT), *assets], cwd=str(src), check=True)
    return python


def _spawn(host: str, port: int, tasks: Sequence[str] | None) -> subprocess.Popen:
    python = ensure_abc(tasks)
    command = [str(python), str(_ENV_SCRIPT), '--host', host, '--port', str(port)]
    return subprocess.Popen(command, env=mujoco_server_env([_ENV_SERVER_DIR, _MAPPING_DIR, _APERTURE_DIR, _ABC_SRC]))


def serve_abc(tasks: Sequence[str] | None, host: str = 'localhost') -> AbstractContextManager[tuple[str, int]]:
    """Run an ABC server for the context's lifetime and yield its address, with ``tasks``' assets prepared."""
    return serve_subprocess(partial(_spawn, tasks=tasks), host)
