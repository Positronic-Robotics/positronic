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


def no_allow(number, guarded_slug):
    return False


def verdict(git, cmd, cwd=CLONE, allow_merge=no_allow):
    return gmm.analyze(cmd, cwd, GUARDED, git, path_exists=fake_path_exists, allow_merge=allow_merge)


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


def test_quoted_heredoc_body_does_not_guard_another_repo(git):
    """A commit message is data, and an apostrophe in it is not a parse failure.

    `<<'EOF'` bodies reach the command as literal stdin, so the apostrophe in "don't"
    used to leave the whole command unparseable — which poisoned the `cd`, and a push
    on a feature branch of an entirely different repo was refused as a push to main.
    """
    cmd = (
        "cd /w/agent_infra && git commit -F- -- x.py <<'EOF' && git push origin HEAD\n"
        "subject line\n\ndon't reverse this decision\nEOF"
    )
    assert verdict(git, cmd) is None


def test_quoted_heredoc_body_is_text_not_commands(git):
    """Nothing in a quoted heredoc body executes, so git-shaped prose in one is prose.

    A commit message explaining a main-branch rule is the realistic case, and blocking
    the commit that carries it is a false positive.
    """
    cmd = "cd /w/positronic && git commit -F- -- x.py <<'EOF'\nWhy we never git push origin main\nEOF"
    assert verdict(git, cmd) is None


def test_dash_quoted_heredoc_body_is_also_text(git):
    cmd = "cd /w/agent_infra && git commit -F- -- x.py <<-'EOF' && git push origin HEAD\n\tdon't\n\tEOF"
    assert verdict(git, cmd) is None


def test_commands_after_a_heredoc_are_still_analyzed(git):
    """Stripping the body must not swallow what follows the terminator."""
    cmd = "cd /w/positronic && git commit -F- -- x.py <<'EOF'\nmessage\nEOF\ngit push origin main"
    assert verdict(git, cmd) is not None


def test_deny_messages_name_the_operation(git):
    assert 'amend' in verdict(git, 'git commit --amend')
    assert 'gh pr merge' in verdict(git, 'gh pr merge 5')
    assert 'push to main' in verdict(git, 'git push origin main')


def test_a_lossily_parsed_merge_is_denied_rather_than_crashing(git):
    """An unparseable command falls back to a coarse scan, whose words carry shell punctuation."""
    assert verdict(git, "echo don't && gh pr merge)") is not None


def test_a_merge_naming_no_pull_request_says_to_name_one(git):
    assert 'name the pull request' in verdict(git, 'gh pr merge --squash', allow_merge=lambda *_: True)


def test_an_authorized_merge_goes_through(git):
    assert verdict(git, 'gh pr merge 566 --squash', allow_merge=lambda *_: True) is None


def test_the_authorization_is_asked_for_the_pull_request_being_merged(git):
    asked = []
    verdict(git, 'gh pr -R o/r merge 566 --delete-branch', allow_merge=lambda n, slug: asked.append((n, slug)) or False)
    assert asked == [(566, GUARDED.casefold())]


def test_a_commit_message_quoting_a_blocked_command_is_not_itself_blocked(git):
    """A backtick inside a quoted heredoc is prose: the delimiter suppresses every expansion."""
    cmd = "cd /w/positronic && git commit -F- -- x.py <<'EOF'\nthe guard blocks `gh pr merge` outright\nEOF"
    assert verdict(git, cmd) is None


def test_an_unquoted_heredoc_still_fails_closed(git):
    """Its body DOES expand, so a substitution in it is a command, not prose."""
    cmd = 'cd /w/positronic && git commit -F- -- x.py <<EOF\n`git push origin main`\nEOF'
    assert verdict(git, cmd) is not None


def write_allow(directory, number, repo='', issued_at=1_000_000, ttl_s=1800):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / gmm.RECEIPT_NAME.format(number=number)
    path.write_text(
        json.dumps({
            gmm.RECEIPT_REPO: repo,
            gmm.RECEIPT_NUMBER: number,
            gmm.RECEIPT_ISSUED_AT: issued_at,
            gmm.RECEIPT_TTL_S: ttl_s,
            'by': 'U0',
        })
    )
    return path


def test_an_authorization_is_spent_by_the_merge_it_permits(tmp_path):
    path = write_allow(tmp_path, 566)
    assert gmm.consume_merge_allow(566, GUARDED, tmp_path, now=1_000_060)
    assert not path.exists()
    assert not gmm.consume_merge_allow(566, GUARDED, tmp_path, now=1_000_060)


def test_an_expired_authorization_is_refused_and_cleared(tmp_path):
    path = write_allow(tmp_path, 566)
    assert not gmm.consume_merge_allow(566, GUARDED, tmp_path, now=1_002_000)
    assert not path.exists()


def test_an_authorization_for_another_pull_request_does_not_carry(tmp_path):
    write_allow(tmp_path, 566)
    assert not gmm.consume_merge_allow(565, GUARDED, tmp_path, now=1_000_060)


@pytest.mark.parametrize('repo', ['', 'positronic', 'Positronic-Robotics/positronic'])
def test_a_reference_naming_this_repository_is_honoured(tmp_path, repo):
    write_allow(tmp_path, 566, repo=repo)
    assert gmm.consume_merge_allow(566, GUARDED, tmp_path, now=1_000_060)


def test_an_authorization_for_another_repository_is_refused(tmp_path):
    write_allow(tmp_path, 566, repo='someone/other')
    assert not gmm.consume_merge_allow(566, GUARDED, tmp_path, now=1_000_060)


def test_a_missing_or_unreadable_receipt_is_simply_no_authorization(tmp_path):
    assert not gmm.consume_merge_allow(566, GUARDED, tmp_path, now=1_000_060)
    (tmp_path / gmm.RECEIPT_NAME.format(number=566)).write_text('{not json')
    assert not gmm.consume_merge_allow(566, GUARDED, tmp_path, now=1_000_060)


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
