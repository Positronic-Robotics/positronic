"""Loading the hook file of a MolmoSpaces env server."""

import numpy as np
import pytest

from positronic.simulator.molmo_spaces import env_hook, mapping


def test_no_hook_file_gives_the_default_hook_which_reads_the_grip_as_molmospaces_does():
    hook = env_hook.load(None)
    assert type(hook) is env_hook.EnvHook
    qpos = np.array([0.4, 0.4], dtype=np.float32)
    assert hook.grip(qpos) == mapping.normalize_grip_qpos(qpos)


def test_a_hook_file_gives_its_hook(tmp_path):
    path = tmp_path / 'site_hook.py'
    path.write_text(
        'from positronic.simulator.molmo_spaces.env_hook import EnvHook\n'
        '\n'
        'class HalfGrip(EnvHook):\n'
        '    def grip(self, gripper_qpos):\n'
        '        return 0.5\n'
        '\n'
        'HOOK = HalfGrip()\n'
    )
    hook = env_hook.load(path)
    assert type(hook).__name__ == 'HalfGrip'
    assert hook.grip(np.zeros(2)) == 0.5


@pytest.mark.parametrize('body', ['', 'HOOK = 3\n'])
def test_a_hook_file_without_an_env_hook_is_refused(tmp_path, body):
    path = tmp_path / 'site_hook.py'
    path.write_text(body)
    with pytest.raises(ValueError, match='must define HOOK'):
        env_hook.load(path)
