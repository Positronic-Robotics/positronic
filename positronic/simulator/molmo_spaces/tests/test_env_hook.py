"""Loading the hook file of a MolmoSpaces env server."""

import hashlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from positronic.simulator.molmo_spaces import env_hook, mapping
from positronic.simulator.molmo_spaces.launcher import molmo_subprocess_env


def test_no_hook_file_gives_the_default_hook_which_reads_the_grip_as_molmospaces_does():
    hook, record = env_hook.load(None)
    assert type(hook) is env_hook.EnvHook
    assert record is None
    qpos = np.array([0.4, 0.4], dtype=np.float32)
    assert hook.grip(qpos) == mapping.normalize_grip_qpos(qpos)


# Loads the hook file of argv[1] as the server does, in an interpreter where positronic does not import.
_LOAD_AS_THE_SERVER = """
import sys
from pathlib import Path

sys.modules['positronic'] = None
import env_hook

hook = env_hook.load(Path(sys.argv[1])).hook
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


_GRIP_HOOK = (
    'from positronic.simulator.molmo_spaces.env_hook import EnvHook\n\n'
    'class Half(EnvHook):\n'
    '    def grip(self, gripper_qpos):\n'
    '        return 0.5\n\n'
    'HOOK = Half()\n'
)


def test_a_hook_file_with_the_expected_digest_loads(hook_path):
    hook_path.write_text(_GRIP_HOOK)
    digest = hashlib.sha256(hook_path.read_bytes()).hexdigest()

    hook, record = env_hook.load(hook_path, digest)

    assert hook.grip([0.0]) == 0.5
    assert record == f'{hook_path.name}@sha256:{digest}'


def test_a_hook_file_that_changed_after_its_digest_was_recorded_is_refused(hook_path):
    hook_path.write_text(_GRIP_HOOK)
    digest = hashlib.sha256(hook_path.read_bytes()).hexdigest()
    hook_path.write_text(_GRIP_HOOK.replace('0.5', '0.25'))

    with pytest.raises(ValueError, match='changed after the eval recorded its digest'):
        env_hook.load(hook_path, digest)


@pytest.mark.parametrize('grip', [0.0, 0.5, 1.0, 2.0])
def test_the_default_hook_sends_the_grip_command_as_molmospaces_does(grip):
    hook, _ = env_hook.load(None)
    assert hook.grip_actuator(grip) == mapping.grip_command_to_actuator(grip)


def _request(path, digest: str) -> dict:
    return mapping.HookFile(path, digest).request()


def test_a_request_without_a_hook_file_gets_the_default_hook():
    hook, record = env_hook.Hooks(None).of({mapping.TOKEN_EPISODE_INDEX: 0})
    assert type(hook) is env_hook.EnvHook
    assert record is None


def test_a_hook_file_loads_at_its_first_request_and_later_requests_with_its_digest_get_that_hook(hook_path):
    hook_path.write_text(_GRIP_HOOK)
    digest = hashlib.sha256(hook_path.read_bytes()).hexdigest()
    hooks = env_hook.Hooks(hook_path.parent)

    first = hooks.of(_request(hook_path, digest))
    hook_path.write_text('HOOK = None\n')
    again = hooks.of(_request(hook_path, digest))

    assert again.hook is first.hook
    assert again.record == f'{hook_path.name}@sha256:{digest}'
    assert hooks.of({}).record is None


def test_two_hook_files_with_the_same_bytes_load_as_two_hooks_that_each_record_their_file(tmp_path):
    other = tmp_path / 'other_hook.py'
    first = tmp_path / 'first_hook.py'
    for path in (first, other):
        path.write_text(_GRIP_HOOK)
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    hooks = env_hook.Hooks(tmp_path)
    try:
        records = [hooks.of(_request(path, digest)).record for path in (first, other)]
    finally:
        for path in (first, other):
            sys.modules.pop(path.stem, None)

    assert records == [f'first_hook.py@sha256:{digest}', f'other_hook.py@sha256:{digest}']


def test_a_second_digest_of_a_loaded_hook_file_name_is_refused(hook_path):
    hook_path.write_text(_GRIP_HOOK)
    hooks = env_hook.Hooks(hook_path.parent)
    hooks.of(_request(hook_path, hashlib.sha256(hook_path.read_bytes()).hexdigest()))
    hook_path.write_text(_GRIP_HOOK.replace('0.5', '0.25'))

    with pytest.raises(ValueError, match='rename the hook file'):
        hooks.of(_request(hook_path, hashlib.sha256(hook_path.read_bytes()).hexdigest()))


@pytest.mark.parametrize('broken', ['raise ImportError("no such pack")\n', 'HOOK = None\n'], ids=['raises', 'no-hook'])
def test_a_hook_file_that_failed_to_load_loads_after_it_is_fixed(hook_path, broken):
    hook_path.write_text(broken)
    hooks = env_hook.Hooks(hook_path.parent)
    with pytest.raises((ImportError, ValueError)):
        hooks.of(_request(hook_path, hashlib.sha256(hook_path.read_bytes()).hexdigest()))
    hook_path.write_text(_GRIP_HOOK)

    fixed = hooks.of(_request(hook_path, hashlib.sha256(hook_path.read_bytes()).hexdigest()))

    assert fixed.hook.grip([0.0]) == 0.5


def test_a_hook_file_under_the_hook_root_of_the_server_loads(hook_path):
    hook_path.write_text(_GRIP_HOOK)
    digest = hashlib.sha256(hook_path.read_bytes()).hexdigest()

    hook, record = env_hook.Hooks(hook_path.parent.parent).of(_request(hook_path, digest))

    assert hook.grip([0.0]) == 0.5
    assert record == f'{hook_path.name}@sha256:{digest}'


def test_a_server_without_a_hook_root_refuses_every_hook_file(hook_path):
    hook_path.write_text(_GRIP_HOOK)
    with pytest.raises(ValueError, match='has no hook root'):
        env_hook.Hooks(None).of(_request(hook_path, hashlib.sha256(hook_path.read_bytes()).hexdigest()))
    assert hook_path.stem not in sys.modules


@pytest.mark.parametrize('link', [False, True], ids=['outside', 'link-out'])
def test_a_hook_file_outside_the_hook_root_of_the_server_is_refused(hook_path, link):
    hook_path.write_text(_GRIP_HOOK)
    root = hook_path.parent / 'hooks'
    root.mkdir()
    named = root / hook_path.name if link else hook_path
    if link:
        named.symlink_to(hook_path)
    with pytest.raises(ValueError, match='is not under the hook root'):
        env_hook.Hooks(root).of(_request(named, hashlib.sha256(hook_path.read_bytes()).hexdigest()))
    assert hook_path.stem not in sys.modules
