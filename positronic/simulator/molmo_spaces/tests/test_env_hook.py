"""Loading the hook file of a MolmoSpaces env server."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from positronic.simulator.molmo_spaces import env_hook, mapping
from positronic.simulator.molmo_spaces.launcher import molmo_subprocess_env


def test_no_hook_file_gives_the_default_hook_which_reads_the_grip_as_molmospaces_does():
    hook = env_hook.load(None)
    assert type(hook) is env_hook.EnvHook
    qpos = np.array([0.4, 0.4], dtype=np.float32)
    assert hook.grip(qpos) == mapping.normalize_grip_qpos(qpos)


# Loads the hook file of argv[1] as the server does, in an interpreter where positronic does not import.
_LOAD_AS_THE_SERVER = """
import sys
from pathlib import Path

sys.modules['positronic'] = None
import env_hook

hook = env_hook.load(Path(sys.argv[1]))
print(type(hook).__name__, hook.grip([0.0]))
"""

_CLASS_HOOK = """
from env_hook import EnvHook

class HalfGrip(EnvHook):
    def grip(self, gripper_qpos):
        return 0.5

HOOK = HalfGrip()
"""

_DATACLASS_HOOK = """
from __future__ import annotations

from dataclasses import dataclass

from env_hook import EnvHook

@dataclass
class HalfGrip(EnvHook):
    value: float = 0.5

    def grip(self, gripper_qpos):
        return self.value

HOOK = HalfGrip()
"""


@pytest.mark.parametrize('body', [_CLASS_HOOK, _DATACLASS_HOOK], ids=['class', 'dataclass'])
def test_the_server_interpreter_loads_a_hook_file_with_the_documented_import(tmp_path, body):
    path = tmp_path / 'site_hook.py'
    path.write_text(textwrap.dedent(body))
    loaded = subprocess.run(
        [sys.executable, '-c', _LOAD_AS_THE_SERVER, str(path)],
        env=molmo_subprocess_env(),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.split() == ['HalfGrip', '0.5']


@pytest.fixture
def hook_path(tmp_path):
    """A hook file path whose module the test forgets after it loads."""
    path = tmp_path / 'site_hook.py'
    yield path
    sys.modules.pop(path.stem, None)


@pytest.mark.parametrize('body', ['', 'HOOK = 3\n'])
def test_a_hook_file_without_an_env_hook_is_refused(hook_path, body):
    hook_path.write_text(body)
    with pytest.raises(ValueError, match='must define HOOK'):
        env_hook.load(hook_path)


def test_a_hook_file_with_the_name_of_a_loaded_module_is_refused(tmp_path):
    path = tmp_path / 'json.py'
    path.write_text('HOOK = None\n')
    with pytest.raises(ValueError, match='rename the hook file'):
        env_hook.load(path)
