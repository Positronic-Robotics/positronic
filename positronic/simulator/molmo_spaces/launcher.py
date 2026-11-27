"""Launch MolmoSpaces in an isolated Python environment.

``MLSPACES_ASSETS_DIR`` must point to the asset packs. ``MUJOCO_GL`` defaults to EGL on Linux and CGL on macOS;
CPU rendering on Linux requires OSMesa or software EGL.
"""

import argparse
import functools
import os
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path

from positronic_evals.molmo_spaces import RIGHT_VIEW_BLOCKED_PATH

from positronic.simulator.env_server.launcher import (
    SERVER_DEPS,
    ensure_pinned_checkout,
    ensure_venv,
    mujoco_server_env,
    serve_subprocess,
    setup_lock,
    uv_pip_install,
)
from positronic.simulator.molmo_spaces import mapping

_ENV_SCRIPT = Path(__file__).parent / 'env.py'
_ENV_SERVER_DIR = Path(__file__).parents[1] / 'env_server'
_MAPPING_DIR = Path(__file__).parent

_MOLMO_REPO = 'https://github.com/allenai/molmospaces.git'
_MOLMO_COMMIT = '67acef9a17bb8212a18aadfc8f62ff6171d84891'
_MOLMO_SRC = Path.home() / '.cache' / 'positronic' / 'molmospaces' / 'src'

# Version constraints compensate for MolmoSpaces' missing lockfile; the file documents how to regenerate them.
_MOLMO_CONSTRAINTS = Path(__file__).parent / 'molmo_constraints.txt'
# Packages that upstream requires and the eval never imports, removed so that the install needs no C compiler.
_MOLMO_OVERRIDES = Path(__file__).parent / 'molmo_overrides.txt'

# MolmoSpaces requires Python 3.11. Filament benchmarks need the alternative mujoco-filament extra.
# The robot configs import the G1 robot at module level, so every install needs the g1_humanoid extra.
_MOLMO_PYTHON = '3.11'
_MOLMO_EXTRAS = 'mujoco,g1_humanoid'


def ensure_molmo_venv() -> Path:
    """Return the Python executable after preparing the pinned MolmoSpaces environment."""
    with setup_lock(_MOLMO_SRC.parent):
        src = ensure_pinned_checkout(_MOLMO_REPO, _MOLMO_COMMIT, _MOLMO_SRC)
        venv = _MOLMO_SRC / '.venv'
        python = ensure_venv(venv, _MOLMO_PYTHON)
        # uv sync also resolves upstream's unused curobo extra, which requires CUDA to build.
        pins = ['-c', str(_MOLMO_CONSTRAINTS), '--override', str(_MOLMO_OVERRIDES)]
        uv_pip_install(venv, [*pins, '-e', f'.[{_MOLMO_EXTRAS}]', *SERVER_DEPS], cwd=src)
    return python


def molmo_subprocess_env() -> dict[str, str]:
    """Subprocess environment with the server's Python paths and GL backend."""
    return mujoco_server_env([_ENV_SERVER_DIR, _MAPPING_DIR])


def _spawn(hook_root: Path | None, host: str, port: int) -> subprocess.Popen:
    if not os.environ.get(mapping.ASSETS_DIR_ENV):
        raise ValueError(f'{mapping.ASSETS_DIR_ENV} must point at the MolmoSpaces asset packs')
    python = ensure_molmo_venv()
    command = [str(python), str(_ENV_SCRIPT), '--host', host, '--port', str(port)]
    command += ['--right-view-blocked', str(RIGHT_VIEW_BLOCKED_PATH)]
    if hook_root is not None:
        command += ['--hook-root', str(hook_root)]
    return subprocess.Popen(command, env=molmo_subprocess_env())


def serve_molmo_spaces(
    host: str = 'localhost', hook_root: Path | None = None
) -> AbstractContextManager[tuple[str, int]]:
    """Run a MolmoSpaces server for the context's lifetime and yield its address.

    The server loads only the hook files under ``hook_root``, and none without it.
    """
    return serve_subprocess(functools.partial(_spawn, hook_root), host)


def main() -> None:
    """Run a MolmoSpaces server that evals dial with ``--eval.env_server``, one run after another, until Ctrl-C."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--hook-root', type=Path, help='the directory of the hook files that an eval can name')
    args = parser.parse_args()
    raise SystemExit(_spawn(args.hook_root, args.host, args.port).wait())


if __name__ == '__main__':
    main()
