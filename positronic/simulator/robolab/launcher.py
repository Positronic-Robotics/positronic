"""Launches the RoboLab env server as a subprocess and owns its lifetime.

positronic starts the server: the env runs in RoboLab's own interpreter via ``uv run --project <checkout>``,
which resolves the Isaac Lab / Isaac Sim dependency stack from the pinned RoboLab checkout (far too heavy for
a PEP 723 script header). The positronic-free ``env_server`` package rides ``PYTHONPATH`` so ``env.py`` imports
the dumb ``server``/``protocol`` without dragging in positronic; ``robolab`` itself resolves from the uv
project.
"""

import os
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path

from positronic.simulator.env_server.launcher import ensure_pinned_checkout, serve_subprocess

_ENV_SCRIPT = Path(__file__).parent / 'env.py'
_ENV_SERVER_DIR = Path(__file__).parents[1] / 'env_server'

_ROBOLAB_REPO = 'https://github.com/NVLabs/RoboLab.git'
_ROBOLAB_COMMIT = '7d45d74904eade3b578a8eb1f2f9f89bc3d40326'
_ROBOLAB_SRC = Path.home() / '.cache' / 'positronic' / 'robolab' / 'src'

# RoboLab ships no uv.lock, so ``uv run --project`` re-resolves its dependencies on every fresh box and a
# day-fresh release can break the install (cffi 2.1.0 published a macOS-only wheel and took down Linux
# resolves). Known-broken releases are appended to the checkout's own ``constraint-dependencies``.
# TODO: the dependency tree is still unfrozen — any unpinned release can shift the resolve.
_CONSTRAINTS = ('cffi != 2.1.0',)
_CONSTRAINTS_ANCHOR = 'constraint-dependencies = ['


def _ensure_robolab_src() -> Path:
    """The pinned RoboLab checkout with its LFS assets (Isaac Lab cannot load a USD from a pointer stub),
    its ``constraint-dependencies`` extended with ``_CONSTRAINTS``."""
    src = ensure_pinned_checkout(_ROBOLAB_REPO, _ROBOLAB_COMMIT, _ROBOLAB_SRC, lfs=True)
    # Re-patch from pristine every run, so the edit is idempotent and survives the pin moving.
    subprocess.run(['git', '-C', str(src), 'checkout', '--', 'pyproject.toml'], check=True)
    pyproject = src / 'pyproject.toml'
    text = pyproject.read_text()
    if _CONSTRAINTS_ANCHOR not in text:
        raise RuntimeError(f'RoboLab pyproject.toml has no {_CONSTRAINTS_ANCHOR!r} to extend')
    additions = ''.join(f'"{constraint}", ' for constraint in _CONSTRAINTS)
    pyproject.write_text(text.replace(_CONSTRAINTS_ANCHOR, _CONSTRAINTS_ANCHOR + additions, 1))
    return src


def _spawn(host: str, port: int) -> subprocess.Popen:
    src = _ensure_robolab_src()
    command = ['uv', 'run', '--project', str(src), str(_ENV_SCRIPT), '--host', host, '--port', str(port), '--headless']
    # Isaac Sim prompts for its EULA on stdin at first launch; the server is headless, so accept it here.
    env = {**os.environ, 'PYTHONPATH': str(_ENV_SERVER_DIR), 'OMNI_KIT_ACCEPT_EULA': 'Y'}
    return subprocess.Popen(command, env=env)


def serve_robolab(host: str = 'localhost') -> AbstractContextManager[tuple[str, int]]:
    """The RoboLab env server as a ``serve`` context manager (the ``serve_subprocess`` contract)."""
    return serve_subprocess(_spawn, host)
