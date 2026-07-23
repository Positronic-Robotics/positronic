import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guard_main_merge as gmm  # noqa: E402

POSITRONIC_URL = 'git@github.com:Positronic-Robotics/positronic.git'
INFRA_URL = 'git@gh-infra:someone/agent_infra.git'
GUARDED = 'Positronic-Robotics/positronic'

CLONE = '/w/positronic'  # positronic clone, on main
WORKTREE = '/w/positronic-wt'  # positronic worktree, feature branch
INFRA = '/w/agent_infra'  # different repo, on main
INFRA_CASED = '/w/agent_infra-cased'  # the guarded repo, origin differs only in casing, on main
POSITRONIC_URL_LC = 'git@github.com:positronic-robotics/positronic.git'


EXISTING_DIRS = {CLONE, WORKTREE, INFRA, INFRA_CASED, '/w', '/tmp', '/somewhere'}


def fake_path_exists(p):
    d = os.path.normpath(p)
    while True:
        if d in EXISTING_DIRS:
            return True
        parent = os.path.dirname(d)
        if parent == d:
            return False
        d = parent


class FakeGit(gmm.GitInfo):
    """In-memory repo map; resolves a dir to its enclosing repo like git's upward discovery."""

    def __init__(self, repos):
        self.repos = repos

    def _repo(self, dirpath):
        d = os.path.normpath(dirpath)
        while True:
            if d in self.repos:
                return self.repos[d]
            parent = os.path.dirname(d)
            if parent == d:
                return None
            d = parent

    def branch(self, dirpath):
        r = self._repo(dirpath)
        return r['branch'] if r else ''

    def remote_url(self, dirpath, name):
        r = self._repo(dirpath)
        return r['remotes'].get(name, '') if r else ''

    def config(self, dirpath, key):
        r = self._repo(dirpath)
        return r.get('config', {}).get(key, '') if r else ''


@pytest.fixture
def git():
    return FakeGit({
        CLONE: {'branch': 'main', 'remotes': {'origin': POSITRONIC_URL}},
        WORKTREE: {'branch': 'feature-x', 'remotes': {'origin': POSITRONIC_URL}},
        INFRA: {'branch': 'main', 'remotes': {'origin': INFRA_URL}},
        INFRA_CASED: {'branch': 'main', 'remotes': {'origin': POSITRONIC_URL_LC}},
    })


def verdict(git, cmd, cwd=CLONE):
    return gmm.analyze(cmd, cwd, GUARDED, git, path_exists=fake_path_exists)


BLOCKED = [
    'git push',
    'git push origin main',
    'git push origin HEAD:main',
    'git push origin HEAD',
    f'git -C {CLONE} push',
    'git commit --amend --no-edit',
    'git checkout main && git merge feature',
    'git merge --continue',
    'git pull',
    'git pull --rebase',
    'git push --all origin',
    'git push origin HEAD:$BR',
    'git -C $DIR push',
    'gh pr merge 5',
    'gh pr -R o/r merge 5',
    # second invocation still guarded after an exempt one
    f'cd {INFRA} && git push && git -C {CLONE} push',
    # grouped cd does not persist outside the subshell — dir tracking poisons, stays guarded
    f'(true && cd {INFRA} ) && git push',
    f'cd {INFRA} && (cd {CLONE} && git push origin main)',
    '(git push)',
    # exempt repo aiming AT the guarded repo
    f'git -C {INFRA} push {POSITRONIC_URL} HEAD:main',
    f'git -C {INFRA} push --repo={POSITRONIC_URL} HEAD:main',
    f'git -C {INFRA} push https://github.com/Positronic-Robotics/positronic HEAD:main',
    f'git -C {INFRA} push {CLONE} HEAD:main',  # local-path destination, resolved via its origin
    f'git -C {INFRA} push just-added-remote HEAD:main',  # unresolvable remote name
    # a failed cd leaves the shell in the guarded repo, so the later push runs there
    'cd /missing; git push',
    'cd /missing || git push',
    # --git-dir / --work-tree redirect git onto the guarded repo from an exempt cwd
    f'git -C {INFRA} --git-dir={CLONE}/.git --work-tree={CLONE} push origin main',
    'git --git-dir=/some/.git push origin main',
    # backtick command substitution in a push arg can expand to `main`
    'git push origin `echo main`',
    'git push `echo origin` main',
    # $(...) substitution keeps a `$` token, caught as an unverifiable refspec
    'git push origin $(echo main)',
    # substituted git subcommand could be push/merge
    'git `echo push` origin main',
    # a clone whose origin differs only in owner/repo casing is still the guarded repo
    f'git -C {INFRA}-cased push',
    # tool invoked by absolute path / via sudo still guarded
    '/usr/bin/git push origin main',
    '/usr/bin/git push',
    '/usr/local/bin/gh pr merge 5',
    'sudo git push origin main',
    # a git command hidden inside a command substitution runs before the outer command
    'msg=`git push origin main`',
    'git status `git commit --amend`',
    'x=$(git push origin main)',
    'echo $(git commit --amend --no-edit)',
    # nested substitution
    'a=`b=$(git push origin main)`',
]

ALLOWED = [
    'ls -la',
    'git status',
    'git log --oneline',
    'git pull --ff-only',
    'git push origin HEAD:feature',
    'git merge --abort',
    # a different-origin repo runs its own push contract
    f'git -C {INFRA} push',
    f'git -C {INFRA} push origin main',
    f'git -C {INFRA} commit --amend --no-edit',
    f'cd {INFRA} && git add x && git commit -m "msg" && git push',
    f'git -C {INFRA} push origin HEAD:main',
    # quoted text is data, not an invocation
    'git commit -m "see git push docs"',
    'echo "git push origin main"',
    # a subshell cd does not persist, but a later plain absolute cd re-establishes certainty
    f'(cd /tmp) ; cd {INFRA} ; git push',
    # inside one subshell the cd DOES apply to the push that follows it
    f'cd /somewhere && (cd {INFRA} && git push origin main)',
    # backtick substitution in a non-push arg is fine
    'git log --format=`echo oneline`',
    f'git -C {INFRA} commit -m `date +%s`',
    # a substitution whose own command is harmless
    'msg=`git log --oneline` ; echo done',
    # single quotes suppress substitution — the backtick text is literal data
    "echo 'git push origin main'",
    "git commit -m 'run `git commit --amend` later'",
]


@pytest.mark.parametrize('cmd', BLOCKED)
def test_blocked(git, cmd):
    assert verdict(git, cmd) is not None, cmd


@pytest.mark.parametrize('cmd', ALLOWED)
def test_allowed(git, cmd):
    assert verdict(git, cmd) is None, cmd


def test_amend_blocked_from_worktree_cwd(git):
    assert verdict(git, 'git commit --amend', cwd=WORKTREE) is not None


def test_feature_branch_bare_push_allowed(git):
    assert verdict(git, 'git push', cwd=WORKTREE) is None
    assert verdict(git, 'git push -u origin feature-x', cwd=WORKTREE) is None


def test_worktree_cwd_targeting_clone_blocked(git):
    assert verdict(git, f'git -C {CLONE} push', cwd=WORKTREE) is not None


def test_cwd_outside_any_repo(git):
    # nothing to push from a repo-less cwd; git itself errors at runtime
    assert verdict(git, 'git push', cwd='/tmp') is None
    assert verdict(git, f'git -C {CLONE} push', cwd='/tmp') is not None


def test_bare_push_default_remote_can_target_guarded_repo(git):
    git.repos[INFRA]['config'] = {'branch.main.pushRemote': 'pos'}
    git.repos[INFRA]['remotes']['pos'] = POSITRONIC_URL
    assert verdict(git, f'git -C {INFRA} push') is not None


def test_multiline_command(git):
    assert verdict(git, 'git checkout main\ngit merge feature') is not None
    assert verdict(git, 'git status\ngit log') is None


def test_quoted_multiline_arg_does_not_split_invocation(git):
    assert verdict(git, 'git commit -m "line one\nline two" --amend') is not None


def test_unbalanced_quoting_falls_back_guarded(git):
    # a heredoc body with a stray apostrophe is unparseable; git-shaped text stays guarded
    assert verdict(git, "cat <<EOF\nit's got git push inside\nEOF") is not None
    assert verdict(git, "echo don't panic") is None


def test_deny_messages_name_the_operation(git):
    assert 'amend' in verdict(git, 'git commit --amend')
    assert 'gh pr merge' in verdict(git, 'gh pr merge 5')
    assert 'push to main' in verdict(git, 'git push origin main')


def _init_repo(path, url, branch='main'):
    subprocess.run(['git', 'init', '-q', '-b', branch, str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', url], check=True)


def test_end_to_end_subprocess(tmp_path):
    """The script speaks the hook stdin/exit-code protocol against real git repos."""
    clone = tmp_path / 'positronic'
    infra = tmp_path / 'infra'
    _init_repo(clone, POSITRONIC_URL)
    _init_repo(infra, INFRA_URL)
    script = Path(__file__).resolve().parents[1] / 'guard_main_merge.py'
    env = {**os.environ, 'CLAUDE_PROJECT_DIR': str(clone)}

    def run(cmd, cwd):
        payload = json.dumps({'cwd': str(cwd), 'tool_input': {'command': cmd}})
        return subprocess.run([sys.executable, str(script)], input=payload, env=env, capture_output=True, text=True)

    r = run('git push', clone)
    assert r.returncode == 2 and 'BLOCKED' in r.stderr
    assert run(f'cd {infra} && git push', clone).returncode == 0
    assert run('ls', clone).returncode == 0
