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
_MOLMO_COMMIT = '713fd12ab593c3bbb4abfaa76622e14249aa36b3'
_MOLMO_SRC = Path.home() / '.cache' / 'positronic' / 'molmospaces' / 'src'

# Version constraints compensate for MolmoSpaces' missing lockfile; the file documents how to regenerate them.
_MOLMO_CONSTRAINTS = Path(__file__).parent / 'molmo_constraints.txt'
# Packages that upstream requires and the eval never imports, removed so that the install needs no C compiler.
_MOLMO_OVERRIDES = Path(__file__).parent / 'molmo_overrides.txt'

# MolmoSpaces requires Python 3.11. Filament benchmarks need the alternative mujoco-filament extra.
_MOLMO_PYTHON = '3.11'
_MOLMO_EXTRA = 'mujoco'


def ensure_molmo_venv() -> Path:
    """Return the Python executable after preparing the pinned MolmoSpaces environment."""
    with setup_lock(_MOLMO_SRC.parent):
        src = ensure_pinned_checkout(_MOLMO_REPO, _MOLMO_COMMIT, _MOLMO_SRC)
        venv = _MOLMO_SRC / '.venv'
        python = ensure_venv(venv, _MOLMO_PYTHON)
        # uv sync also resolves upstream's unused curobo extra, which requires CUDA to build.
        pins = ['-c', str(_MOLMO_CONSTRAINTS), '--override', str(_MOLMO_OVERRIDES)]
        uv_pip_install(venv, [*pins, '-e', f'.[{_MOLMO_EXTRA}]', *SERVER_DEPS], cwd=src)
    return python


def molmo_subprocess_env() -> dict[str, str]:
    """Subprocess environment with the server's Python paths and GL backend."""
    return mujoco_server_env([_ENV_SERVER_DIR, _MAPPING_DIR])


def _spawn(hook: Path | None, hook_sha256: str | None, host: str, port: int) -> subprocess.Popen:
    if not os.environ.get(mapping.ASSETS_DIR_ENV):
        raise ValueError(f'{mapping.ASSETS_DIR_ENV} must point at the MolmoSpaces asset packs')
    python = ensure_molmo_venv()
    command = [str(python), str(_ENV_SCRIPT), '--host', host, '--port', str(port)]
    command += ['--right-view-blocked', str(RIGHT_VIEW_BLOCKED_PATH)]
    if hook is not None:
        command += ['--hook', str(hook)]
    if hook_sha256 is not None:
        command += ['--hook-sha256', hook_sha256]
    return subprocess.Popen(command, env=molmo_subprocess_env())


def serve_molmo_spaces(
    host: str = 'localhost', hook: Path | None = None, hook_sha256: str | None = None
) -> AbstractContextManager[tuple[str, int]]:
    """Run a MolmoSpaces server for the context's lifetime and yield its address.

    ``hook`` is a file that defines ``HOOK``, an ``env_hook.EnvHook`` that changes each episode. With
    ``hook_sha256``, the server refuses a hook file with another digest.
    """
    return serve_subprocess(functools.partial(_spawn, hook, hook_sha256), host)


def main() -> None:
    """Run a MolmoSpaces server that an eval dials with ``--eval.env_server``, until that eval run ends."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--hook', dest='hook', type=Path, help='a hook file that defines HOOK, an env_hook.EnvHook')
    args = parser.parse_args()
    hook = None if args.hook is None else args.hook.expanduser().resolve()
    raise SystemExit(_spawn(hook, None, args.host, args.port).wait())


if __name__ == '__main__':
    main()
