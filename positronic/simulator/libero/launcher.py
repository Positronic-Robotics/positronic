"""Launches the LIBERO env server as a subprocess and owns its lifetime.

positronic starts the server: the env runs in its own 3.10 interpreter via ``uv run --no-project env.py``,
which reads the script's PEP 723 deps and builds an isolated environment. The positronic-free ``env_server``
package and a LIBERO source checkout are placed on ``PYTHONPATH`` so ``env.py`` imports the dumb
``server``/``protocol`` and ``libero`` without dragging in positronic.
"""

import os
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path

from positronic.simulator.env_server.launcher import ensure_pinned_checkout, serve_subprocess

_ENV_SCRIPT = Path(__file__).parent / 'env.py'
_ENV_SERVER_DIR = Path(__file__).parents[1] / 'env_server'

_LIBERO_REPO = 'https://github.com/Lifelong-Robot-Learning/LIBERO.git'
_LIBERO_COMMIT = '8f1084e3132a39270c3a13ebe37270a43ece2a01'
_LIBERO_CACHE = Path.home() / '.cache' / 'positronic' / 'libero'
_LIBERO_CONFIG = _LIBERO_CACHE / 'config'


def _ensure_libero_src() -> Path:
    """The pinned LIBERO checkout, importable only from its repo root: LIBERO declares ``install_requires=[]``
    and ships ``libero`` as a PEP 420 namespace package (no top-level ``__init__.py``), so ``find_packages``
    builds an empty wheel — and the bddl/init-state/asset files the env reads live in the repo tree, not a wheel.
    """
    return ensure_pinned_checkout(_LIBERO_REPO, _LIBERO_COMMIT, _LIBERO_CACHE / 'src')


def _spawn(host: str, port: int) -> subprocess.Popen:
    command = ['uv', 'run', '--no-project', str(_ENV_SCRIPT), '--host', host, '--port', str(port)]
    # LIBERO writes a config of repo-relative asset paths into ``LIBERO_CONFIG_PATH`` on first import; pin it beside
    # the checkout so it can never go stale against a ``~/.libero`` from an earlier, differently-located clone.
    env = {
        **os.environ,
        'PYTHONPATH': os.pathsep.join([str(_ENV_SERVER_DIR), str(_ensure_libero_src())]),
        'LIBERO_CONFIG_PATH': str(_LIBERO_CONFIG),
    }
    return subprocess.Popen(command, env=env)


def serve_libero(host: str = 'localhost') -> AbstractContextManager[tuple[str, int]]:
    """The LIBERO env server as a ``serve`` context manager (the ``serve_subprocess`` contract)."""
    return serve_subprocess(_spawn, host)
