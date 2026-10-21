import importlib.metadata
import json
import re
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import pytest

from positronic import utils
from positronic.utils.credentials import CredentialInRunMetadata
from positronic.utils.git import get_git_state, get_package_checkout, get_package_git_state

WHEEL_COMMIT = '08b08698e11f081c5c8b3f82b44835be3f331a60'


class _Distribution:
    version = '0.0.0'

    def __init__(self, direct_url: dict | None):
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == 'direct_url.json'
        return None if self._direct_url is None else json.dumps(self._direct_url)


def install_as(monkeypatch, direct_url: dict | None) -> None:
    monkeypatch.setattr(importlib.metadata, 'distribution', lambda name: _Distribution(direct_url))


def vcs_wheel() -> dict:
    return {
        'url': 'https://github.com/Positronic-Robotics/positronic.git',
        'vcs_info': {'vcs': 'git', 'commit_id': WHEEL_COMMIT, 'requested_revision': 'main'},
    }


def git_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    env = {'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t', 'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t'}

    def run(*args: str) -> None:
        subprocess.run(['git', '-C', str(path), *args], check=True, capture_output=True, env=env)

    run('init', '-q')
    (path / 'f').write_text(str(path))
    run('add', 'f')
    run('commit', '-q', '-m', 'init')
    state = get_git_state(workdir=path)
    assert state is not None
    return str(state['commit'])


@pytest.fixture
def cwd_repo(tmp_path, monkeypatch) -> str:
    """The process runs inside a git checkout that is not the installed positronic."""
    head = git_repo(tmp_path / 'cwd')
    monkeypatch.chdir(tmp_path / 'cwd')
    return head


def test_a_wheel_built_from_a_vcs_url_names_the_commit_it_was_built_from(cwd_repo, monkeypatch):
    install_as(monkeypatch, vcs_wheel())

    state = get_package_git_state()

    assert state is not None
    assert state == {
        'commit': WHEEL_COMMIT,
        'dirty': False,
        'url': 'https://github.com/Positronic-Robotics/positronic.git',
        'requested_revision': 'main',
    }
    assert state['commit'] != cwd_repo
    assert get_package_checkout() is None


def test_an_editable_install_names_its_checkout_not_the_working_directory(cwd_repo, tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    head = git_repo(checkout)
    install_as(monkeypatch, {'url': checkout.as_uri(), 'dir_info': {'editable': True}})

    state = get_package_git_state()

    assert state is not None
    assert state == get_git_state(workdir=checkout)
    assert state['commit'] == head != cwd_repo
    assert get_package_checkout() == checkout


def test_an_install_with_no_origin_has_no_revision(cwd_repo, monkeypatch):
    install_as(monkeypatch, None)
    assert get_package_git_state() is None

    install_as(monkeypatch, {'url': 'file:///nowhere', 'archive_info': {}})
    assert get_package_git_state() is None

    mercurial = {'url': 'https://example.invalid/positronic', 'vcs_info': {'vcs': 'hg', 'commit_id': 'a' * 40}}
    install_as(monkeypatch, mercurial)
    assert get_package_git_state() is None


def test_run_metadata_records_the_installed_revision_and_no_diff_for_a_wheel(cwd_repo, monkeypatch):
    install_as(monkeypatch, vcs_wheel())
    Path('f').write_text('changed')

    metadata = utils.run_metadata(add_uv_lock=False)

    assert metadata['git.positronic']['commit'] == WHEEL_COMMIT
    assert 'git.positronic.diff' not in metadata
    assert metadata['git.current']['commit'] == cwd_repo
    assert metadata['git.current']['dirty'] is True


# Split, so that a diff which holds this line does not look like it holds a credential.
TOKEN = 'q7RfX2mKp' + '9WzL4vNc8' + 'YbT3hJd'


def test_run_metadata_with_a_credential_in_the_command_is_refused_and_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['pos', '--policy.api_key', TOKEN])

    with pytest.raises(CredentialInRunMetadata) as exc:
        utils.save_run_metadata(tmp_path / 'out', add_uv_lock=False)

    assert not (tmp_path / 'out').exists()
    assert TOKEN not in ''.join(traceback.format_exception(exc.value))


def test_writing_run_metadata_with_a_credential_in_the_command_is_refused(tmp_path):
    with pytest.raises(CredentialInRunMetadata):
        utils.write_run_metadata(tmp_path / 'out', {'command': ['pos', '--policy.api_key', TOKEN]})

    assert not (tmp_path / 'out').exists()


def test_run_metadata_writes_a_marker_in_place_of_a_diff_line_with_a_credential(cwd_repo, tmp_path, monkeypatch):
    install_as(monkeypatch, vcs_wheel())
    monkeypatch.setattr(sys, 'argv', ['pos'])
    Path('f').write_text(f'API_TOKEN = "{TOKEN}"\nx = 1\n')

    written = utils.save_run_metadata(tmp_path / 'out', patterns=['f'], add_uv_lock=False).read_text()

    assert TOKEN not in written
    assert re.search(r'\+<git\.current\.diff: line \d+ removed, it looks like a credential>', written)
    assert '+x = 1' in written


def test_deferred_run_metadata_refuses_a_credential_before_it_starts_its_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['pos', f'--policy.api_key={TOKEN}'])
    started = []
    monkeypatch.setattr(threading.Thread, 'start', lambda self: started.append(self))

    with pytest.raises(CredentialInRunMetadata):
        utils.save_run_metadata_deferred(tmp_path / 'out')

    assert started == []
