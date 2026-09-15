"""Launches the MolmoSpaces env server as a subprocess and owns its lifetime.

positronic starts the server: the env runs in MolmoSpaces' own interpreter — a per-checkout ``.venv`` with the
``molmospaces[mujoco]`` stack (mujoco ~=3.5, the resource-manager asset layer, torch) installed into it, far too
heavy and Python-version-pinned (3.11) to share positronic's venv. The positronic-free ``env_server`` package and
this package's ``mapping`` module ride ``PYTHONPATH`` so ``env.py`` imports the dumb ``server``/``protocol`` and
the pure wire mappings without dragging in positronic; ``molmo_spaces`` resolves from the venv.

MolmoSpaces renders MuJoCo scenes, so the server needs a GL backend (``MUJOCO_GL``) and its asset packs
(``MLSPACES_ASSETS_DIR``). Both env vars pass through from the caller; unset, ``MUJOCO_GL`` takes the backend
the host platform offers — ``egl`` (GPU) on Linux, ``cgl`` on macOS, which has no EGL and rejects it. A
GPU-less Linux box overrides with ``MUJOCO_GL=osmesa`` for CPU software rendering.
"""

import fcntl
import os
import subprocess
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.launcher import ensure_pinned_checkout, serve_subprocess
from positronic.simulator.molmo_spaces import mapping

_ENV_SCRIPT = Path(__file__).parent / 'env.py'
_ENV_SERVER_DIR = Path(__file__).parents[1] / 'env_server'
_MAPPING_DIR = Path(__file__).parent  # ``mapping.py`` — imported flat by env.py, positronic-free

_MOLMO_REPO = 'https://github.com/allenai/molmospaces.git'
_MOLMO_COMMIT = 'c2f1b583f087e1d3994e1377574843b759d9d0f8'
_MOLMO_SRC = Path.home() / '.cache' / 'positronic' / 'molmospaces' / 'src'

# MolmoSpaces ships no lockfile, so a bare install re-resolves every transitive dep on each fresh box. This
# constraints file pins the full resolution (a frozen known-good venv, minus molmo-spaces' own editable line),
# fed to the install via ``-c`` so the pinned commit always builds the same environment. Regenerate it when
# ``_MOLMO_COMMIT`` bumps — see the file header.
_MOLMO_CONSTRAINTS = Path(__file__).parent / 'molmo_constraints.txt'

# MolmoSpaces pins Python 3.11 and installs its MuJoCo renderer stack via the ``mujoco`` extra (classic renderer,
# mujoco ~=3.5). ``mujoco-filament`` is the alternative for bench-v2 filament scenes; the classic renderer is the
# eval default.
_MOLMO_PYTHON = '3.11'
_MOLMO_EXTRA = 'mujoco'

# ``env.py`` imports positronic's ``env_server`` off PYTHONPATH, which needs ``websockets`` (the wire server) and
# ``msgpack`` (the frame codec). MolmoSpaces currently pulls both, but that is incidental to its own deps — install
# them explicitly so env_server's wire contract holds even if MolmoSpaces drops them. Constraints mirror positronic's.
_WIRE_DEPS = ('websockets>=15.0.1', 'msgpack')


@contextmanager
def _checkout_lock() -> Iterator[None]:
    """Serialize checkout + ``uv sync`` across processes sharing the cache, so a warm-cache fan-out of eval jobs
    mounting one ``~/.cache/positronic/molmospaces`` filesystem does not race a forced checkout against a sync."""
    _MOLMO_SRC.parent.mkdir(parents=True, exist_ok=True)
    with open(_MOLMO_SRC.parent / 'setup.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def ensure_molmo_venv() -> Path:
    """The MolmoSpaces venv python, after ensuring the pinned checkout and its installed stack exist.

    Install the stack before returning: a cold first install far exceeds any client's connect deadline, which
    should only cover the sim's boot. Install the ``mujoco`` extra explicitly into a venv the way MolmoSpaces'
    own image does, rather than ``uv sync`` — which also resolves the ``curobo`` extra, a CUDA build that needs a
    GPU toolchain and is not on the eval task path. Both steps are idempotent and fast when warm. MolmoSpaces
    ships no uv.lock, so ``molmo_constraints.txt`` pins the transitive resolution (``-c``) for a reproducible env.
    """
    venv = _MOLMO_SRC / '.venv'
    with _checkout_lock():
        src = ensure_pinned_checkout(_MOLMO_REPO, _MOLMO_COMMIT, _MOLMO_SRC)
        if not venv.exists():
            subprocess.run(['uv', 'venv', '--python', _MOLMO_PYTHON, str(venv)], check=True)
        subprocess.run(
            ['uv', 'pip', 'install', '-c', str(_MOLMO_CONSTRAINTS), '-e', f'.[{_MOLMO_EXTRA}]', *_WIRE_DEPS],
            cwd=str(src),
            env={**os.environ, 'VIRTUAL_ENV': str(venv)},
            check=True,
        )
    return venv / 'bin' / 'python'


def molmo_subprocess_env() -> dict[str, str]:
    """The environment a molmo-venv script runs under: the positronic-free ``env_server``/``mapping`` on
    PYTHONPATH and a GL backend. GPU OpenGL by default; a GPU-less box exports MUJOCO_GL=osmesa, or relies on
    mesa's software EGL, for CPU rendering."""
    return {
        **os.environ,
        'PYTHONPATH': os.pathsep.join([str(_ENV_SERVER_DIR), str(_MAPPING_DIR)]),
        mapping.GL_BACKEND_ENV: os.environ.get(mapping.GL_BACKEND_ENV, mapping.GL_BACKEND_DEFAULT),
    }


def _spawn(host: str, port: int, benchmark_dir: Path, task_horizon_steps: int | None) -> subprocess.Popen:
    # env.py exits on these before it binds the port. Check them here, where the failure can name the missing
    # precondition instead of reaching the caller as a bare pre-bind exit status.
    if not os.environ.get(mapping.ASSETS_DIR_ENV):
        raise ValueError(f'{mapping.ASSETS_DIR_ENV} must point at the MolmoSpaces asset packs')
    if not benchmark_dir.is_dir():
        raise ValueError(f'benchmark dir {benchmark_dir} does not exist')
    python = ensure_molmo_venv()
    command = [
        str(python),
        str(_ENV_SCRIPT),
        protocol.OPT_HOST,
        host,
        protocol.OPT_PORT,
        str(port),
        mapping.OPT_BENCHMARK_DIR,
        str(benchmark_dir),
    ]
    if task_horizon_steps is not None:
        command += [mapping.OPT_TASK_HORIZON_STEPS, str(task_horizon_steps)]
    return subprocess.Popen(command, env=molmo_subprocess_env())


def serve_molmo_spaces(
    benchmark_dir: Path, host: str = 'localhost', task_horizon_steps: int | None = None
) -> AbstractContextManager[tuple[str, int]]:
    """The MolmoSpaces env server as a ``serve`` context manager (the ``serve_subprocess`` contract).

    ``benchmark_dir`` (a dir holding ``benchmark.json``) is fixed for the run; the reset token selects the
    episode within it, so one task-agnostic server serves every trial. ``task_horizon_steps`` optionally overrides
    the benchmark's own horizon (mirroring MolmoSpaces' ``--task_horizon_steps``); ``None`` reads it per episode.
    """
    return serve_subprocess(lambda host, port: _spawn(host, port, benchmark_dir, task_horizon_steps), host)
