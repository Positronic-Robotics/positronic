"""Launches the RoboLab env server as a subprocess and owns its lifetime.

positronic starts the server: the env runs in RoboLab's own interpreter via ``uv run --project <checkout>``,
which installs the Isaac Lab / Isaac Sim dependency stack from the lock of the pinned RoboLab checkout (far too
heavy for a PEP 723 script header). The positronic-free ``env_server`` package rides ``PYTHONPATH`` so ``env.py``
imports the dumb ``server``/``protocol`` without dragging in positronic; ``robolab`` itself resolves from the uv
project.
"""

import argparse
import os
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

_ENV_SCRIPT = Path(__file__).parent / 'env.py'
_ENV_SERVER_DIR = Path(__file__).parents[1] / 'env_server'

_ROBOLAB_REPO = 'https://github.com/NVLabs/RoboLab.git'
_ROBOLAB_COMMIT = 'dc3383888b7279b4256312268a35a5f9046c44d3'  # v0.3.3
_ROBOLAB_SRC = Path.home() / '.cache' / 'positronic' / 'robolab' / 'src'

# RoboLab admits Python 3.11 and 3.12, and its lock holds ``isaaclab`` 2.2 (the ``isaac50`` stack) for cp311 only.
_ROBOLAB_PYTHON = '3.11'

# RoboLab installs Isaac only through an extra; ``isaac50`` is the IsaacSim 5.0 / IsaacLab 2.2 stack env.py drives.
_ROBOLAB_ISAAC_EXTRA = 'isaac50'


def _spawn(
    host: str, port: int, cameras: str, max_envs: int, join_deadline: float = DEFAULT_JOIN_DEADLINE
) -> subprocess.Popen:
    # Only the setup is one at a time: a concurrent checkout would reset files from under ``uv sync``. A waiter needs
    # the same checkout and install, so it waits no longer than its own setup would take. The per-eval ``env.py``
    # runs outside the lock, so evals still fan out.
    with setup_lock(_ROBOLAB_SRC.parent):
        # The pinned checkout with its LFS assets: Isaac Lab cannot load a USD from a pointer stub.
        src = ensure_pinned_checkout(_ROBOLAB_REPO, _ROBOLAB_COMMIT, _ROBOLAB_SRC, lfs=True)
        # Install the dependency stack before spawning: a cold first install (~15 GB of Isaac wheels) far
        # exceeds the client's connect deadline, which should only ever cover Isaac boot. Idempotent and fast
        # when warm. ``--frozen`` installs the uv.lock that RoboLab ships and resolves nothing, so a box with no
        # index access installs from its cache, and a new release on an index cannot change the install. The
        # spawn below passes ``--no-sync`` (``uv run`` re-syncs by default), so no install runs outside this lock.
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
        '--cameras',
        cameras,
        '--max-envs',
        str(max_envs),
        '--join-deadline',
        str(join_deadline),
        '--headless',
    ]
    # The DROID rig's model (URDF + meshes + gripper) for the viewer and offline IK. env.py runs in RoboLab's
    # interpreter and cannot build it, so it is serialized here (wire codec) and env.py emits it as
    # ``robot_meta``. A fresh temp file per spawn, in the container-local tmpdir — never the shared cache
    # filesystem, where a concurrent eval's rewrite could tear a reader mid-decode.
    # The rig's own model, unaltered: RoboLab measures and drives at ``robolab_eef``, and ``RobolabAdapter``
    # converts between that and the model's ``GRIPPER_SITE`` on both the observation and command paths.
    robot_meta = bundled_franka_model()
    fd, meta_path = tempfile.mkstemp(prefix='robolab_robot_meta_', suffix='.bin')
    with os.fdopen(fd, 'wb') as meta_file:
        meta_file.write(encode(robot_meta))
    # Isaac Sim prompts for its EULA on stdin at first launch; the server is headless, so accept it here.
    env = {
        **os.environ,
        'PYTHONPATH': str(_ENV_SERVER_DIR),
        'OMNI_KIT_ACCEPT_EULA': 'Y',
        'ROBOLAB_ROBOT_META': meta_path,
    }
    return subprocess.Popen(command, env=env)


def serve_robolab(
    cameras: str, host: str = 'localhost', *, max_envs: int = 1
) -> AbstractContextManager[tuple[str, int]]:
    """The RoboLab env server as a ``serve`` context manager (the ``serve_subprocess`` contract).

    ``cameras`` names the set in ``keys.CAMERA_SETS`` the server renders; RoboLab bakes it into the
    registered task, so one server serves one set.

    ``max_envs`` is the most clones of the scene the one Isaac process steps together. Each batch of evals gets as
    many clones as the batch size its first reset carries.
    """
    return serve_subprocess(partial(_spawn, cameras=cameras, max_envs=max_envs), host)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Serve RoboLab to several evals, one scene clone each, batch after batch until it is stopped.'
    )
    parser.add_argument('--cameras', default=keys.WRIST_LEFT_RIGHT, choices=sorted(keys.CAMERA_SETS))
    parser.add_argument('--max-envs', type=int, required=True, help='the most scene clones, one per eval of a batch')
    parser.add_argument(
        '--join-deadline',
        type=float,
        default=DEFAULT_JOIN_DEADLINE,
        help='seconds the first reset of a batch waits for the clones whose evals have not asked for it',
    )
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    proc = _spawn(args.host, args.port, args.cameras, args.max_envs, args.join_deadline)
    try:
        sys.exit(proc.wait())
    finally:
        terminate(proc)


if __name__ == '__main__':
    main()
