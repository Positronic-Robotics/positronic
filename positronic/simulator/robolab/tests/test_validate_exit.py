"""The exit status of ``validate.py`` when a check raises.

The Isaac app swallows an uncaught exception: the process then exits with status 0. A stub ``env`` module does the
same with ``sys.excepthook``, so the script runs here with no Isaac Lab.
"""

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

ROBOLAB = Path(__file__).resolve().parents[1]
ENV_SERVER = ROBOLAB.parent / 'env_server'

_STUBS = {
    'env.py': """
        import os
        import sys
        from types import SimpleNamespace

        import keys

        sys.excepthook = lambda *exc: os._exit(0)
        args = SimpleNamespace(robot=keys.DROID, cameras=keys.DEFAULT_CAMERA_SETS[keys.DROID], max_envs=1)
        simulation_app = SimpleNamespace(close=lambda: None)


        class DroidEnv:
            def __init__(self):
                raise RuntimeError('the env failed to build')


        RobolabEnv = YamEnv = DroidEnv
        """,
    'torch.py': 'def __getattr__(name):\n    return lambda *args, **kwargs: None\n',
    'isaaclab/__init__.py': '',
    'isaaclab/utils/__init__.py': '',
    'isaaclab/utils/math.py': 'matrix_from_quat = quat_apply = quat_inv = quat_mul = None\n',
    'robolab/__init__.py': '',
    'robolab/core/__init__.py': '',
    'robolab/core/sensors/__init__.py': '',
    'robolab/core/sensors/contact_sensor_utils.py': 'get_batch_contact_sensor = None\n',
    'robolab/robots/__init__.py': '',
    'robolab/robots/droid.py': 'EEF_OFFSET_ROT = None\n',
}


def test_a_check_that_raises_gives_a_nonzero_exit(tmp_path):
    for name in ('validate.py', 'keys.py', 'yam_spec.py'):
        shutil.copy(ROBOLAB / name, tmp_path / name)
    for name, text in _STUBS.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(textwrap.dedent(text))
    run = subprocess.run(
        [sys.executable, str(tmp_path / 'validate.py')],
        env={'PYTHONPATH': f'{tmp_path}:{ENV_SERVER}'},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 1, run.stderr
    assert 'the env failed to build' in run.stderr
