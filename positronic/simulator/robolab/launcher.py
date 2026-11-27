"""Launches the RoboLab env server as a subprocess and owns its lifetime.

The server runs ``env.py`` in the interpreter of the pinned RoboLab checkout, through ``uv run --project``. The
positronic-free ``env_server`` package is on its ``PYTHONPATH``; positronic is not importable there.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from contextlib import AbstractContextManager
from functools import partial
from pathlib import Path

from positronic.drivers.roboarm.models import bundled_franka_model
from positronic.simulator.env_server.launcher import ensure_pinned_checkout, serve_subprocess, setup_lock, terminate
from positronic.simulator.env_server.protocol import encode
from positronic.simulator.env_server.server import DEFAULT_JOIN_DEADLINE
from positronic.simulator.robolab import keys
from positronic.simulator.robolab.task_catalog import METADATA as TASK_LIB_METADATA

_ENV_SCRIPT = Path(__file__).parent / 'env.py'
_ENV_SERVER_DIR = Path(__file__).parents[1] / 'env_server'

_ROBOLAB_REPO = 'https://github.com/NVLabs/RoboLab.git'
_ROBOLAB_COMMIT = 'dc3383888b7279b4256312268a35a5f9046c44d3'  # v0.3.3
_ROBOLAB_SRC = Path.home() / '.cache' / 'positronic' / 'robolab' / 'src'

# RoboLab admits Python 3.11 and 3.12, and its lock holds ``isaaclab`` 2.2 (the ``isaac50`` stack) for cp311 only.
_ROBOLAB_PYTHON = '3.11'

# RoboLab installs Isaac only through an extra; ``isaac50`` is the IsaacSim 5.0 / IsaacLab 2.2 stack env.py drives.
_ROBOLAB_ISAAC_EXTRA = 'isaac50'

# RoboLab builds ``flatdict`` with ``[tool.uv.extra-build-dependencies]``, which uv makes stable in 0.10.0 (changelog).
_UV_MIN_VERSION = (0, 10, 0)


def _require_uv_version() -> None:
    found = subprocess.run(['uv', '--version'], check=True, capture_output=True, text=True).stdout.strip()
    match = re.match(r'uv (\d+)\.(\d+)\.(\d+)', found)
    if match is None or tuple(int(part) for part in match.groups()) < _UV_MIN_VERSION:
        minimum = '.'.join(str(part) for part in _UV_MIN_VERSION)
        raise RuntimeError(f'the RoboLab setup needs uv {minimum} or newer, and found {found!r}')


def _spawn(
    host: str,
    port: int,
    robot: str,
    max_envs: int,
    join_deadline: float = DEFAULT_JOIN_DEADLINE,
    task_libs: tuple[Path, ...] = (),
) -> subprocess.Popen:
    # A missing library fails here, before the Isaac boot, rather than in the server after it.
    for lib in task_libs:
        if not (lib / TASK_LIB_METADATA).is_file():
            raise FileNotFoundError(f'{lib} is no task library: it has no {TASK_LIB_METADATA}')
    _require_uv_version()
    # One setup at a time, so no checkout resets files under ``uv sync``; the hold is one install, which a waiter needs.
    with setup_lock(_ROBOLAB_SRC.parent):
        # With its LFS assets: Isaac Lab cannot load a USD from a pointer stub.
        src = ensure_pinned_checkout(_ROBOLAB_REPO, _ROBOLAB_COMMIT, _ROBOLAB_SRC, lfs=True)
        # Before the spawn: a cold install (~15 GB) far exceeds the connect deadline. ``--frozen`` installs the
        # shipped uv.lock and resolves nothing; the spawn passes ``--no-sync``, so no install runs outside the lock.
        subprocess.run(
            [
                'uv',
                'sync',
                '--frozen',
                '--project',
                str(src),
                '--python',
                _ROBOLAB_PYTHON,
                '--extra',
                _ROBOLAB_ISAAC_EXTRA,
            ],
            check=True,
        )
    command = [
        'uv',
        'run',
        '--no-sync',
        '--project',
        str(src),
        '--python',
        _ROBOLAB_PYTHON,
        str(_ENV_SCRIPT),
        '--host',
        host,
        '--port',
        str(port),
        '--robot',
        robot,
        '--max-envs',
        str(max_envs),
        '--join-deadline',
        str(join_deadline),
        *(arg for lib in task_libs for arg in ('--task-lib', str(lib))),
        '--headless',
    ]
    # Isaac Sim prompts for its EULA on stdin at first launch; the server is headless, so accept it here.
    env = {**os.environ, 'PYTHONPATH': str(_ENV_SERVER_DIR), 'OMNI_KIT_ACCEPT_EULA': 'Y'}
    if robot == keys.DROID:
        # env.py cannot build the DROID model, so it reads this encoded copy and sends it as ``robot_meta``.
        # One temp file per spawn, off the shared cache, where a concurrent rewrite could tear a reader.
        fd, meta_path = tempfile.mkstemp(prefix='robolab_robot_meta_', suffix='.bin')
        with os.fdopen(fd, 'wb') as meta_file:
            meta_file.write(encode(bundled_franka_model()))
        env['ROBOLAB_ROBOT_META'] = meta_path
    return subprocess.Popen(command, env=env)


def serve_robolab(
    *, host: str = 'localhost', robot: str = keys.DROID, max_envs: int = 1, task_libs: tuple[Path, ...] = ()
) -> AbstractContextManager[tuple[str, int]]:
    """The RoboLab env server as a ``serve`` context manager (the ``serve_subprocess`` contract).

    ``robot`` names the robot that the server puts in the task scenes, one of ``keys.ROBOTS``. Each reset token names
    the camera set that the server renders, one of ``keys.ROBOT_CAMERA_SETS[robot]``.

    ``max_envs`` is the most clones of the scene the one Isaac process steps together. Each batch of evals gets as
    many clones as the batch size its first reset carries.

    ``task_libs`` are task library directories on this host (see ``task_catalog``); their tasks run by name.
    """
    return serve_subprocess(partial(_spawn, robot=robot, max_envs=max_envs, task_libs=task_libs), host)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Serve RoboLab to several evals, one scene clone each, batch after batch until it is stopped.'
    )
    parser.add_argument('--robot', default=keys.DROID, choices=keys.ROBOTS)
    parser.add_argument('--max-envs', type=int, required=True, help='the most scene clones, one per eval of a batch')
    parser.add_argument(
        '--join-deadline',
        type=float,
        default=DEFAULT_JOIN_DEADLINE,
        help='seconds the first reset of a batch waits for the clones whose evals have not asked for it',
    )
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument(
        '--task-lib', type=Path, action='append', default=[], help='a task library directory; repeatable'
    )
    args = parser.parse_args()
    proc = _spawn(args.host, args.port, args.robot, args.max_envs, args.join_deadline, tuple(args.task_lib))
    try:
        sys.exit(proc.wait())
    finally:
        terminate(proc)


if __name__ == '__main__':
    main()
