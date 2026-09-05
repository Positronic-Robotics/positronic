import io
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


def verdict(git, cmd, cwd=CLONE, allow_merge=no_allow, gh_repo_env=''):
    return gmm.analyze(
        cmd, cwd, GUARDED, git, path_exists=fake_path_exists, allow_merge=allow_merge, gh_repo_env=gh_repo_env
    )


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
    verdict(git, 'gh pr merge 566 --delete-branch', allow_merge=lambda n, slug: asked.append((n, slug)) or False)
    assert asked == [(566, GUARDED.casefold())]


def test_a_value_taking_flag_does_not_donate_its_value_as_the_pull_request(git):
    """`gh pr merge --subject 566 999` merges 999; authorizing 566 must not clear it."""
    asked = []
    verdict(git, 'gh pr merge --subject 566 999 --squash', allow_merge=lambda n, slug: asked.append(n) or False)
    assert asked == [999]


def test_a_merge_in_another_repository_is_refused_without_consulting_any_authorization(git):
    """`-R` selects a repository this guard holds no receipts for."""
    asked = []
    assert (
        verdict(git, 'gh pr -R someone/other merge 566', allow_merge=lambda n, slug: asked.append(n) or True)
        is not None
    )
    assert asked == []


def test_a_repository_shorthand_selects_it_however_pflag_writes_it(git):
    """pflag takes a shorthand's value attached, and clusters shorthands ahead of it."""
    asked = []
    for cmd in (
        'gh pr -Rsomeone/other merge 566',
        'gh pr merge 566 -R=someone/other',
        'gh pr merge 566 -dRsomeone/other',
        'gh pr merge 566 -dR someone/other',
    ):
        assert verdict(git, cmd, allow_merge=lambda n, slug: asked.append(n) or True) is not None, cmd
    assert asked == []
    assert verdict(git, f'gh pr merge 566 -dR{GUARDED}', allow_merge=lambda *_: True) is None


def test_a_merge_word_the_shell_builds_is_refused(git):
    """A brace list, a glob and an expansion each hand gh an argument the text does not spell out."""
    asked = []
    for cmd in (
        'gh pr merge 566 $EXTRA',
        'gh pr merge 566 $(echo -R someone/other)',
        'gh pr merge 566 {-R,someone/other}',
        f'gh pr merge 566 --repo {GUARDED} --body-file /tmp/pr*.md',
        # `NAME=` is an assignment only ahead of the command word; written as an argument the shell
        # splits it like any other, so `x=junk -R someone/other` reaches gh.
        f'gh pr merge 566 --repo {GUARDED} --body x=$EXTRA',
        # double quotes suppress a glob but not a parameter expansion
        'gh pr merge 566 "$EXTRA" someone/other',
    ):
        assert verdict(git, cmd, allow_merge=lambda n, slug: asked.append(n) or True) is not None, cmd
    assert asked == []


def test_a_quoted_dollar_is_a_literal_the_shell_never_builds(git):
    """Quoting decides whether a `$` expands, wherever in the word the quote opens."""
    for cmd in (
        f"gh pr merge 566 --repo {GUARDED} --subject 'Read $EXTRA as text'",
        f"gh pr merge 566 --repo {GUARDED} --subject='Read $EXTRA as text'",
    ):
        assert verdict(git, cmd, allow_merge=lambda *_: True) is None, cmd


def test_an_expansion_beside_the_merge_cannot_reach_its_arguments(git):
    """A separator ends the command, so a later word is never one of gh's."""
    cmd = f'gh pr merge 566 --repo {GUARDED} --squash && echo $HOME'
    assert verdict(git, cmd, allow_merge=lambda *_: True) is None


def test_a_merge_reading_its_token_from_a_substitution_still_passes(git):
    """The assignment prefix expands into the environment, not into gh's arguments."""
    cmd = f'GH_TOKEN=$(cat ~/tokens/gh) gh pr merge 566 --repo {GUARDED} --squash'
    assert verdict(git, cmd, allow_merge=lambda *_: True) is None


def test_a_merge_run_from_another_repo_is_not_this_repository_s(git):
    """gh resolves the repository from the working directory when nothing names one."""
    asked = []
    assert (
        verdict(git, f'cd {INFRA} && gh pr merge 566', allow_merge=lambda n, slug: asked.append(n) or True) is not None
    )
    assert asked == []


def test_a_merge_whose_repository_cannot_be_established_is_refused(git):
    asked = []
    for cmd in ('cd /missing; gh pr merge 566', 'GH_REPO=someone/other gh pr merge 566'):
        assert verdict(git, cmd, allow_merge=lambda n, slug: asked.append(n) or True) is not None
    assert asked == []


def test_an_authorization_is_not_spent_by_a_command_the_guard_blocks_anyway(git):
    """The merge never runs, so the human's one authorization must survive the refusal."""
    asked = []
    assert (
        verdict(
            git, 'gh pr merge 566 --squash && git push origin main', allow_merge=lambda n, slug: asked.append(n) or True
        )
        is not None
    )
    assert asked == []


def test_a_merge_inside_a_command_substitution_is_refused_and_spends_nothing(git):
    """The body runs before the outer command, which can then be refused with the receipt gone."""
    asked = []
    cmd = 'echo ' + chr(96) + 'gh pr merge 566' + chr(96) + ' && git push origin main'
    assert verdict(git, cmd, allow_merge=lambda n, slug: asked.append(n) or True) is not None
    assert asked == []


def test_a_merge_beside_a_command_substitution_is_refused(git):
    """A substituted assignment name reaches the tokens as a sentinel, not as `GH_REPO=`."""
    asked = []
    cmd = 'env G' + chr(96) + 'printf H_REPO' + chr(96) + '=someone/other gh pr merge 566'
    denial = verdict(git, cmd, allow_merge=lambda n, slug: asked.append(n) or True)
    assert denial is not None and 'command substitution' in denial
    assert asked == []


def test_a_process_substitution_hides_a_second_command_in_the_same_segment(git):
    """`<(` keeps the command it feeds and the one it runs together, where only the first is read."""
    asked = []
    cmd = 'gh pr merge 566 --body-file <' + '(gh pr merge 567)'
    assert verdict(git, cmd, allow_merge=lambda n, slug: asked.append(n) or True) is not None
    assert asked == []


def test_a_spend_mark_cannot_be_taken_twice(tmp_path, as_root):
    """Exclusive create, so two hooks racing one receipt cannot both find it unspent."""
    write_allow(tmp_path, 566)
    assert consume(566, tmp_path)
    assert not consume(566, tmp_path)


def test_a_command_merging_two_pull_requests_is_refused(git):
    """One authorization names one merge, and the first would be spent before the rest is known."""
    asked = []
    verdict = gmm.analyze(
        'gh pr merge 566 && gh pr merge 567',
        CLONE,
        GUARDED,
        git,
        path_exists=fake_path_exists,
        allow_merge=lambda n, slug: asked.append(n) or True,
    )
    assert verdict is not None and 'one pull request per command' in verdict
    assert asked == []


def test_a_quoted_gh_repo_assignment_is_read_as_the_shell_reads_it(git):
    """Quote removal happens before the assignment, so the raw text never spells the name."""
    asked = []
    assert (
        verdict(git, "env G''H_REPO=someone/other gh pr merge 566", allow_merge=lambda n, slug: asked.append(n) or True)
        is not None
    )
    assert asked == []


def test_gh_repo_in_the_environment_is_enough_to_refuse(git):
    """It selects the repository the way `-R` does, and a command cannot be read for it."""
    asked = []
    assert (
        verdict(
            git, 'gh pr merge 566', gh_repo_env='someone/other', allow_merge=lambda n, slug: asked.append(n) or True
        )
        is not None
    )
    assert asked == []


def test_reading_the_help_is_not_a_merge(git):
    assert verdict(git, 'gh pr merge --help') is None
    assert verdict(git, 'gh help pr merge') is None


def test_a_commit_message_quoting_a_blocked_command_is_not_itself_blocked(git):
    """A backtick inside a quoted heredoc is prose: the delimiter suppresses every expansion."""
    cmd = "cd /w/positronic && git commit -F- -- x.py <<'EOF'\nthe guard blocks `gh pr merge` outright\nEOF"
    assert verdict(git, cmd) is None


def test_an_unquoted_heredoc_still_fails_closed(git):
    """Its body DOES expand, so a substitution in it is a command, not prose."""
    cmd = 'cd /w/positronic && git commit -F- -- x.py <<EOF\n`git push origin main`\nEOF'
    assert verdict(git, cmd) is not None


def write_allow(directory, number, repo='', issued_at=1_000_000, ttl_s=1800, receipt_id='abcd1234'):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / gmm.RECEIPT_NAME.format(number=number)
    path.write_text(
        json.dumps({
            gmm.RECEIPT_REPO: repo,
            gmm.RECEIPT_NUMBER: number,
            gmm.RECEIPT_ISSUED_AT: issued_at,
            gmm.RECEIPT_TTL_S: ttl_s,
            gmm.RECEIPT_ID: receipt_id,
            'by': 'U0',
        })
    )
    return path


@pytest.fixture
def as_root(monkeypatch):
    """Stand in for the ownership check: a test cannot create a root-owned file."""
    monkeypatch.setattr(gmm, '_written_by_root', lambda path, directory: path.exists())


def consume(number, tmp_path, now=1_000_060, repo=GUARDED):
    return gmm.consume_merge_allow(number, repo, tmp_path, tmp_path / 'spent', now=now)


def test_an_authorization_is_spent_by_the_merge_it_permits(tmp_path, as_root):
    write_allow(tmp_path, 566)
    assert consume(566, tmp_path)
    assert not consume(566, tmp_path)


def test_a_second_authorization_is_honoured_however_close_it_follows_the_first(tmp_path, as_root):
    """The spend mark names the authorization, so two of them never collide."""
    write_allow(tmp_path, 566, receipt_id='aaaa1111')
    assert consume(566, tmp_path)
    write_allow(tmp_path, 566, receipt_id='bbbb2222')
    assert consume(566, tmp_path)


@pytest.mark.parametrize('receipt_id', ['', None, 'no', 'has space', 'x' * 65])
def test_a_receipt_that_names_no_usable_authorization_is_refused(tmp_path, as_root, receipt_id):
    write_allow(tmp_path, 566, receipt_id=receipt_id)
    assert not consume(566, tmp_path)


def test_a_receipt_that_root_did_not_write_is_no_authorization(tmp_path):
    """The real check, unpatched: this test's own file is not root's."""
    write_allow(tmp_path, 566)
    assert not consume(566, tmp_path)


def test_an_expired_authorization_is_refused(tmp_path, as_root):
    write_allow(tmp_path, 566)
    assert not consume(566, tmp_path, now=1_002_000)


def test_an_authorization_for_another_pull_request_does_not_carry(tmp_path, as_root):
    write_allow(tmp_path, 566)
    assert not consume(565, tmp_path)


@pytest.mark.parametrize('repo', ['', 'positronic', 'Positronic-Robotics/positronic'])
def test_a_reference_naming_this_repository_is_honoured(tmp_path, as_root, repo):
    write_allow(tmp_path, 566, repo=repo)
    assert consume(566, tmp_path)


def test_an_authorization_for_another_repository_is_refused(tmp_path, as_root):
    write_allow(tmp_path, 566, repo='someone/other')
    assert not consume(566, tmp_path)


def test_a_missing_or_unreadable_receipt_is_simply_no_authorization(tmp_path, as_root):
    assert not consume(566, tmp_path)
    (tmp_path / gmm.RECEIPT_NAME.format(number=566)).write_text('{not json')
    assert not consume(566, tmp_path)


def test_a_world_writable_receipt_directory_is_not_root_only(tmp_path):
    path = write_allow(tmp_path, 566)
    tmp_path.chmod(0o777)
    assert not gmm._written_by_root(path, tmp_path)


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


MERGE_TOOL = 'mcp__github__merge_pull_request'
# The tools that reach a branch of this repository today. The guard reads a SHAPE rather than this
# list — `test_a_later_tool_of_the_same_shape_is_caught_without_being_named` is what keeps the
# enumeration exhaustive — but every one of them is pinned here as well, so a shape that stopped
# covering one of them fails rather than going quiet.
BRANCH_WRITE_TOOLS = ['mcp__github__create_or_update_file', 'mcp__github__push_files', 'mcp__github__delete_file']
GUARDED_REPO_ARGS = {'owner': 'Positronic-Robotics', 'repo': 'positronic'}
OTHER_REPO_ARGS = {'owner': 'someone', 'repo': 'agent_infra'}


def mcp_verdict(tool, arguments, guarded=GUARDED, allow_merge=no_allow):
    return gmm.analyze_mcp(tool, arguments, guarded, allow_merge=allow_merge)


def merge_args(number=566, **overrides):
    return {**GUARDED_REPO_ARGS, 'pullNumber': number, 'merge_method': 'squash', **overrides}


def test_an_mcp_merge_without_an_authorization_is_refused():
    assert mcp_verdict(MERGE_TOOL, merge_args()) is not None


def test_an_authorized_mcp_merge_goes_through():
    assert mcp_verdict(MERGE_TOOL, merge_args(), allow_merge=lambda number, slug: True) is None


def test_the_mcp_asks_about_the_pull_request_being_merged():
    asked = []
    mcp_verdict(MERGE_TOOL, merge_args(571), allow_merge=lambda number, slug: asked.append((number, slug)) or True)
    assert asked == [(571, GUARDED.casefold())]


def test_both_halves_spend_one_authorization(tmp_path, as_root, git):
    """The two paths do not each carry their own idea of what is authorized."""
    write_allow(tmp_path, 566)

    def allow(number, slug):
        return gmm.consume_merge_allow(number, slug, tmp_path, tmp_path / 'spent', now=1_000_060)

    assert mcp_verdict(MERGE_TOOL, merge_args(), allow_merge=allow) is None
    assert verdict(git, 'gh pr merge 566', allow_merge=allow) is not None


def test_an_mcp_merge_of_another_repository_is_refused_without_consulting_any_authorization():
    def allow(number, slug):
        raise AssertionError('the authorization must not be spent on a merge refused anyway')

    assert mcp_verdict(MERGE_TOOL, {**OTHER_REPO_ARGS, 'pullNumber': 566}, allow_merge=allow) is not None


@pytest.mark.parametrize('number', [None, 'abc', '', True, 1.5, {'n': 1}])
def test_an_mcp_merge_naming_no_pull_request_says_to_name_one(number):
    assert 'names no pull request' in mcp_verdict(MERGE_TOOL, merge_args(number))


def test_a_pull_request_number_json_wrote_as_a_float_is_the_same_number():
    assert mcp_verdict(MERGE_TOOL, merge_args(566.0), allow_merge=lambda number, slug: number == 566) is None


def test_an_mcp_merge_whose_repository_cannot_be_established_is_refused():
    assert mcp_verdict(MERGE_TOOL, {'pullNumber': 566}) is not None
    assert mcp_verdict(MERGE_TOOL, merge_args(), guarded='') is not None


@pytest.mark.parametrize('tool', BRANCH_WRITE_TOOLS)
def test_an_mcp_commit_onto_main_is_refused(tool):
    assert mcp_verdict(tool, {**GUARDED_REPO_ARGS, 'branch': 'main', 'message': 'm'}) is not None


@pytest.mark.parametrize('tool', BRANCH_WRITE_TOOLS)
def test_an_mcp_commit_onto_another_branch_is_untouched(tool):
    assert mcp_verdict(tool, {**GUARDED_REPO_ARGS, 'branch': 'feature-x', 'message': 'm'}) is None


@pytest.mark.parametrize('tool', BRANCH_WRITE_TOOLS)
def test_another_repository_runs_its_own_branch_contract(tool):
    assert mcp_verdict(tool, {**OTHER_REPO_ARGS, 'branch': 'main', 'message': 'm'}) is None


def test_a_later_tool_of_the_same_shape_is_caught_without_being_named():
    """The enumeration is a shape, so a tool the MCP grows tomorrow is guarded the day it appears."""
    assert mcp_verdict('mcp__github__replace_branch_contents', {**GUARDED_REPO_ARGS, 'branch': 'main'}) is not None
    assert mcp_verdict('mcp__github__merge_branch', {**GUARDED_REPO_ARGS, 'base': 'main'}) is not None


@pytest.mark.parametrize(
    'tool,arguments',
    [
        ('mcp__github__get_file_contents', {**GUARDED_REPO_ARGS, 'path': 'x'}),
        ('mcp__github__list_pull_requests', GUARDED_REPO_ARGS),
        ('mcp__github__create_pull_request', {**GUARDED_REPO_ARGS, 'base': 'main', 'head': 'feature-x'}),
        ('mcp__github__update_pull_request_branch', {**GUARDED_REPO_ARGS, 'pullNumber': 566}),
        ('mcp__tracker__create_ticket', {'title': 'merge the branch onto main'}),
        ('Bash', {'command': 'gh pr merge 566'}),
    ],
)
def test_a_call_that_cannot_land_a_commit_on_main_is_untouched(tool, arguments):
    assert mcp_verdict(tool, arguments) is None


def run_hook(payload, cwd, project_dir, script=None):
    script = script or Path(__file__).resolve().parents[1] / 'guard_main_merge.py'
    env = {**os.environ, 'CLAUDE_PROJECT_DIR': str(project_dir)}
    body = payload if isinstance(payload, str) else json.dumps({**payload, 'cwd': str(cwd)})
    return subprocess.run([sys.executable, str(script)], input=body, env=env, capture_output=True, text=True)


@pytest.fixture
def guarded_clone(tmp_path):
    clone = tmp_path / 'positronic'
    _init_repo(clone, POSITRONIC_URL)
    return clone


def test_the_mcp_merge_is_refused_end_to_end(guarded_clone):
    """The hole this closes: the same merge the shell path refuses, arriving as a tool call."""
    r = run_hook({'tool_name': MERGE_TOOL, 'tool_input': merge_args()}, guarded_clone, guarded_clone)
    assert r.returncode == 2 and 'BLOCKED' in r.stderr and '!allow_merge' in r.stderr


def test_an_mcp_commit_onto_main_is_refused_end_to_end(guarded_clone):
    arguments = {**GUARDED_REPO_ARGS, 'branch': 'main', 'files': [{'path': 'x', 'content': 'y'}], 'message': 'm'}
    r = run_hook({'tool_name': 'mcp__github__push_files', 'tool_input': arguments}, guarded_clone, guarded_clone)
    assert r.returncode == 2 and 'BLOCKED' in r.stderr


def test_an_mcp_read_is_allowed_end_to_end(guarded_clone):
    arguments = {**GUARDED_REPO_ARGS, 'path': 'README.md'}
    payload = {'tool_name': 'mcp__github__get_file_contents', 'tool_input': arguments}
    assert run_hook(payload, guarded_clone, guarded_clone).returncode == 0


@pytest.mark.parametrize('arguments,expected', [({'command': 'git push'}, 2), ({'command': 'ls'}, 0), ({}, 0)])
def test_the_command_half_is_unchanged_end_to_end(arguments, expected, guarded_clone):
    """A Bash payload decides exactly what it decided before the MCP half existed."""
    payload = {'tool_name': 'Bash', 'tool_input': arguments}
    assert run_hook(payload, guarded_clone, guarded_clone).returncode == expected


@pytest.mark.parametrize(
    'body,expected',
    [
        ('{not json', 0),
        ('[]', 0),
        ('{"tool_name": "mcp__github__merge_pull_request", "tool_input": {', 2),
        ('null mcp__github__push_files', 2),
    ],
)
def test_a_payload_the_guard_cannot_read_refuses_only_where_a_merge_could_hide(body, expected, guarded_clone):
    assert run_hook(body, guarded_clone, guarded_clone).returncode == expected


def test_a_gate_that_cannot_answer_refuses(monkeypatch, capsys, guarded_clone):
    """An authorization gate fails CLOSED: a crash in it must not become an allow."""
    monkeypatch.setattr(gmm, 'analyze_mcp', lambda *a, **k: 1 / 0)
    monkeypatch.setenv('CLAUDE_PROJECT_DIR', str(guarded_clone))
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps({'tool_name': MERGE_TOOL, 'tool_input': merge_args()})))
    assert gmm.main() == 2
    assert 'BLOCKED' in capsys.readouterr().err
