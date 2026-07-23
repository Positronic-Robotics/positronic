#!/usr/bin/env python3
"""Claude Code PreToolUse hook guarding this repo's `main` from the agent's Bash tool.

Blocks history-rewriting `git commit --amend`, merges / integrating pulls / direct pushes to
`main`, and `gh pr merge`. Amend rewrites history — create a new commit instead. Integrating
into main requires an explicit human/operator command run outside the agent's Bash tool.

Scope: the git-command guards apply only to invocations that operate on THIS repo — same
`origin` as the session's project repo, which covers clones and worktrees. A `git -C <dir> …`
or `cd <dir> && git …` targeting a different repo (a deploy clone with its own push contract)
is exempt, unless the push destination itself names the guarded repo. Anything unresolvable —
an unexpanded `$dir`, `cd -`, a `cd` inside `( … )` / `{ … }` / a substitution, a dir with no
origin — stays guarded: fail toward blocking.

Wired in `.claude/settings.json` (PreToolUse, matcher Bash): reads the hook payload on stdin,
exits 2 with a message on stderr to block, 0 to allow. Stdlib-only so it runs without the
project venv.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass

DENY_TAIL = (
    ' Merging to main requires a PR and an explicit human/operator command — a named human must run the merge'
    ' themselves (e.g. via the `!` prefix or their own shell).'
)
AMEND_MSG = 'BLOCKED: Never amend commits, create new ones instead.'

# git global options that consume the following argument in their space-separated form
GIT_ARG_OPTS = {'-C', '-c', '--namespace', '--git-dir', '--work-tree', '--super-prefix', '--exec-path'}
# git-push options that consume the following argument in their space-separated form
PUSH_ARG_OPTS = {'--receive-pack', '--exec', '--push-option', '-o'}
BOUNDARY_TOKENS = {';', '&', '|', '&&', '||'}
GROUP_TOKENS = {'(', ')', '{', '}'}
# git global options that redirect it onto ANOTHER repo than cwd/-C imply; resolving them is more
# trouble than it's worth, so their presence poisons the invocation's dir (fail closed).
GIT_DIR_REDIRECT_OPTS = ('--git-dir', '--work-tree', '--namespace')
# Sentinel for a command-substitution word (`` `…` ``): its runtime value is unknown, so it
# poisons whatever it appears in — a cd's target, a push's refspec.
SUBST = '\x00subst'

# A word that names `main` as a push target or checkout target: `main`, `+main`, `HEAD:main`,
# `origin/main`, `main:other` — but not `mainline` or `feature/main2`.
MAIN_REF_RE = re.compile(r'(^|[:/+])main(?![\w/\-])')


def repo_slug(url: str) -> str:
    """owner/repo from a remote URL, ssh or https form; empty when there is none.

    GitHub owner/repo are case-insensitive, so the slug is casefolded — two clones of the same
    repo whose remotes differ only in casing must compare equal.
    """
    url = re.sub(r'\.git/?$', '', url or '')
    m = re.search(r'[:/]([^/:]+/[^/:]+)$', url)
    return m.group(1).casefold() if m else ''


class GitInfo:
    """Read-only git state queries, faked in tests."""

    def _run(self, dirpath: str, *args: str) -> str:
        try:
            r = subprocess.run(['git', '-C', dirpath, *args], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return ''
        return r.stdout.strip() if r.returncode == 0 else ''

    def branch(self, dirpath: str) -> str:
        return self._run(dirpath, 'branch', '--show-current')

    def remote_url(self, dirpath: str, name: str) -> str:
        return self._run(dirpath, 'remote', 'get-url', name)

    def origin_url(self, dirpath: str) -> str:
        return self.remote_url(dirpath, 'origin')

    def config(self, dirpath: str, key: str) -> str:
        return self._run(dirpath, 'config', '--get', key)


@dataclass
class Invocation:
    kind: str  # 'git' | 'gh'
    words: list[str]  # tokens after the git/gh word
    dir: str | None  # effective working dir; None = unresolvable -> guarded


def _expand(path: str) -> str:
    return os.path.expanduser(path) if path.startswith('~') else path


def _cd_target(args: list[str], cur: str | None, path_exists) -> str | None:
    """Dir after a `cd …` segment. `cd -`, flags, unexpanded variables, and targets that do not
    exist all -> None (guarded): a failed `cd` leaves the shell in the ORIGINAL dir, so a later
    command runs there, not at the missing target."""
    if not args:
        return os.path.expanduser('~')
    target = args[0]
    if target.startswith('-') or '$' in target or '`' in target:
        return None
    target = _expand(target)
    if not os.path.isabs(target):
        if cur is None:
            return None
        target = os.path.join(cur, target)
    return target if path_exists(target) else None


def _apply_dash_c(words: list[str], base: str | None) -> str | None:
    """Apply `-C` values the way git composes them (a relative -C chains onto the previous).

    `--git-dir` / `--work-tree` / `--namespace` redirect git onto a different repo than cwd
    implies; rather than resolve them, their presence poisons the dir so the invocation is
    guarded regardless of where it was launched.
    """
    i = 0
    while i < len(words):
        w = words[i]
        if w in GIT_DIR_REDIRECT_OPTS or w.startswith(tuple(o + '=' for o in GIT_DIR_REDIRECT_OPTS)):
            return None
        if w == '-C':
            if i + 1 >= len(words) or '$' in words[i + 1] or words[i + 1] == SUBST:
                return None
            val = _expand(words[i + 1])
            if os.path.isabs(val):
                base = val
            elif base is not None:
                base = os.path.join(base, val)
            i += 2
            continue
        i += 1
    return base


def _segments(cmd: str) -> list[list[str]]:
    """Quote-aware token lists for each simple-command segment, with `<group>` marker segments.

    Splits on ; & | && || and on newlines between tokens. Grouping tokens ( ) { } become
    `['<group>']` markers — a `cd` inside a subshell may not apply to later segments, so the
    walker poisons its dir tracking there. A word carrying a backtick command substitution is
    replaced in place by the SUBST sentinel (its runtime value is unknown). Raises ValueError
    on unbalanced quoting.
    """
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    segments: list[list[str]] = [[]]
    prev_lineno = lex.lineno
    while True:
        tok = lex.get_token()
        if tok is None:
            break
        if lex.lineno != prev_lineno:
            # A lineno bump larger than the newlines inside the token itself means a newline
            # separated this token from the previous one — a segment boundary.
            if lex.lineno - prev_lineno > tok.count('\n'):
                segments.append([])
            prev_lineno = lex.lineno
        if tok in BOUNDARY_TOKENS:
            segments.append([])
        elif tok in GROUP_TOKENS:
            segments.append(['<group>'])
            segments.append([])
        else:
            segments[-1].append(SUBST if '`' in tok else tok)
    return [s for s in segments if s]


def _substitution_bodies(cmd: str) -> list[str]:
    """Bodies of the command substitutions in `cmd` — backtick `` `…` `` and `$(…)`.

    Bash runs these in a subshell (same cwd) BEFORE the outer command, so an embedded
    `` `git commit --amend` `` must be analyzed in its own right. Single-quoted regions
    suppress substitution and are skipped; double quotes do not. Best-effort scanner; an
    unbalanced construct simply yields nothing for that span (the outer SUBST poisoning still
    fails closed).
    """
    bodies: list[str] = []
    i, n, quote = 0, len(cmd), None
    while i < n:
        c = cmd[i]
        if quote == "'":
            quote = None if c == "'" else quote
            i += 1
            continue
        if c == '\\':
            i += 2
            continue
        if c == "'":
            quote = "'"
        elif c == '"':
            quote = None if quote == '"' else '"'
        elif c == '`':
            j = cmd.find('`', i + 1)
            if j == -1:
                break
            bodies.append(cmd[i + 1 : j])
            i = j + 1
            continue
        elif c == '$' and i + 1 < n and cmd[i + 1] == '(':
            depth, k = 1, i + 2
            while k < n and depth:
                depth += {'(': 1, ')': -1}.get(cmd[k], 0)
                k += 1
            if depth == 0:
                bodies.append(cmd[i + 2 : k - 1])
                i = k
                continue
        i += 1
    return bodies


def _find_tool(seg: list[str]) -> tuple[int, str | None]:
    """First token in `seg` that invokes git/gh, matched by BASENAME so a path-qualified form
    (`/usr/bin/git`, `sudo … git`) is recognized. Returns (index, 'git'|'gh') or (-1, None)."""
    for i, tok in enumerate(seg):
        if tok in ('<group>', SUBST):
            continue
        if os.path.basename(tok) in ('git', 'gh'):
            return i, os.path.basename(tok)
    return -1, None


def _parse_lossy(cmd: str) -> list[Invocation]:
    """Fallback for unparseable quoting (e.g. a heredoc body with a stray apostrophe).

    Coarse operator-split scan; every invocation gets a None dir, so it stays guarded.
    """
    invs = []
    for chunk in re.split(r'[;&|\n]+', cmd):
        m = re.search(r'(?:^|[^\w-])(?:\S*/)?(git|gh)\s+(\S.*)', chunk)
        if m:
            invs.append(Invocation(m.group(1), m.group(2).split(), None))
    return invs


def parse_invocations(cmd: str, cwd: str, path_exists=os.path.isdir) -> list[Invocation]:
    try:
        segments = _segments(cmd)
    except ValueError:
        return _parse_lossy(cmd)
    invs = []
    cur: str | None = cwd
    for seg in segments:
        if seg == ['<group>']:
            # A subshell / brace group breaks the linear cd model: `cd /a && (cd /b && git push)`
            # runs the push from /b, not /a. Poison the tracked dir; an absolute -C or a later
            # plain absolute cd re-establishes certainty.
            cur = None
            continue
        has_subst = SUBST in seg
        if os.path.basename(seg[0]) == 'cd':
            # A cd whose target is a command substitution goes to an unknown dir.
            cur = None if has_subst else _cd_target(seg[1:], cur, path_exists)
            continue
        idx, kind = _find_tool(seg)
        if kind:
            words = seg[idx + 1 :]
            invs.append(Invocation(kind, words, _apply_dash_c(words, cur) if kind == 'git' else cur))
        if has_subst:
            # A command substitution anywhere in this segment may have changed the dir for the
            # segments that follow.
            cur = None
    return invs


def _subcmd(words: list[str]) -> tuple[str, list[str]]:
    """The invocation's real subcommand (skipping git's global options) and the words after it."""
    i = 0
    while i < len(words):
        w = words[i]
        if w in GIT_ARG_OPTS:
            i += 2
            continue
        if w.startswith('-'):
            i += 1
            continue
        w = re.sub(r'[)}].*$', '', w)  # lossy parses can leave a closing delimiter glued on
        if w:
            return w, words[i + 1 :]
        i += 1
    return '', []


def _is_gh_pr_merge(words: list[str]) -> bool:
    """gh (cobra) accepts persistent flags between `pr` and the subcommand — `gh pr -R o/r merge 1`."""
    seen_pr = skip = False
    for w in words:
        if skip:
            skip = False
            continue
        if w.startswith('-'):
            if w in ('-R', '--repo'):
                skip = True
            continue
        w = re.sub(r'[)}].*$', '', w)
        if not w:
            continue
        if not seen_pr:
            if w == 'pr':
                seen_pr = True
            continue
        return w == 'merge'
    return False


def _push_dest_slug(inv_dir: str | None, rest: list[str], git: GitInfo) -> str:
    """Destination slug of one push invocation, resolved from the repo it runs in.

    The `<repository>` positional (or `--repo=`) may be a URL (slugged directly), a local path
    (identified by THAT repo's own origin), or a named remote (URL read from the invocation's
    repo config). A bare push resolves the branch's configured push remote
    (pushRemote -> remote.pushDefault -> branch.remote -> origin). Empty when unresolvable —
    callers treat that as guarded.
    """
    remote = None
    want_repo = skip = False
    for w in rest:
        if want_repo:
            remote = w
            break
        if skip:
            skip = False
            continue
        if w.startswith('-'):
            if w.startswith('--repo='):
                remote = w[len('--repo=') :]
                break
            if w == '--repo':
                want_repo = True
            elif w in PUSH_ARG_OPTS:
                skip = True
            continue
        remote = w
        break
    if remote is None:
        if inv_dir is None:
            return ''
        br = git.branch(inv_dir)
        remote = (
            git.config(inv_dir, f'branch.{br}.pushRemote')
            or git.config(inv_dir, 'remote.pushDefault')
            or git.config(inv_dir, f'branch.{br}.remote')
            or 'origin'
        )
    if '$' in remote:
        return ''
    if remote.startswith('file://'):
        remote = remote[len('file://') :]
    if remote.startswith(('/', './', '../', '~')):
        return repo_slug(git.origin_url(_expand(remote)))
    if '://' in remote or re.match(r'^[^/\s]+@[^/\s]+:', remote):
        return repo_slug(remote)
    if inv_dir is None:
        return ''
    return repo_slug(git.remote_url(inv_dir, remote))


def analyze(cmd: str, cwd: str, guarded_slug: str, git: GitInfo, path_exists=os.path.isdir, _depth=0) -> str | None:
    """The deny message for `cmd`, or None to allow it."""
    guarded_slug = guarded_slug.casefold()
    # A command substitution runs its own command (same cwd) before the outer command, so a git
    # invocation hidden inside `` `…` `` / `$(…)` must be analyzed too. Bounded recursion depth
    # guards against pathological nesting.
    if _depth < 8:
        for body in _substitution_bodies(cmd):
            deny = analyze(body, cwd, guarded_slug, git, path_exists, _depth + 1)
            if deny:
                return deny
    invs = parse_invocations(cmd, cwd, path_exists)
    git_invs = [(inv, *_subcmd(inv.words)) for inv in invs if inv.kind == 'git']

    def exempt(inv_dir: str | None) -> bool:
        if not guarded_slug or inv_dir is None:
            return False
        slug = repo_slug(git.origin_url(inv_dir))
        return bool(slug) and slug != guarded_slug

    def on_main(inv_dir: str | None) -> bool:
        return True if inv_dir is None else git.branch(inv_dir) == 'main'

    def switches_to_main() -> bool:
        return any(
            sub in ('checkout', 'switch') and not exempt(inv.dir) and any(re.fullmatch(r'\+?main', w) for w in rest)
            for inv, sub, rest in git_invs
        )

    def targets_main(inv_dir: str | None) -> bool:
        return on_main(inv_dir) or switches_to_main()

    for inv in invs:
        if inv.kind == 'gh' and _is_gh_pr_merge(inv.words):
            return 'BLOCKED: gh pr merge is not allowed.' + DENY_TAIL

    for inv, sub, rest in git_invs:
        if exempt(inv.dir):
            if sub != 'push':
                continue
            # An exempt repo can still aim AT the guarded repo — `git push <url-of-this-repo>
            # main` names the destination explicitly — so the exemption additionally requires a
            # resolvable destination (not a substitution) that is NOT the guarded repo.
            dest = _push_dest_slug(inv.dir, rest, git)
            if SUBST not in rest and dest and dest != guarded_slug:
                continue
        if sub == SUBST and targets_main(inv.dir):
            # git's subcommand is itself a command substitution — could be `push`/`merge`.
            return 'BLOCKED: a git subcommand built from a command substitution cannot be verified.' + DENY_TAIL
        if sub == 'commit' and any(w == '--amend' or w.startswith('--amend=') for w in rest):
            return AMEND_MSG
        if sub == 'merge' and targets_main(inv.dir):
            if not any(w in ('--abort', '--quit') for w in rest):
                return 'BLOCKED: git merge onto main is not allowed.' + DENY_TAIL
        if sub == 'pull' and targets_main(inv.dir):
            if not any(w == '--ff-only' or w.startswith('--ff-only=') for w in rest):
                return 'BLOCKED: git pull onto main is not allowed (only --ff-only).' + DENY_TAIL
        if sub == 'push':
            if SUBST in rest:
                # A backtick-substituted remote/refspec can expand to `origin main` at runtime.
                return (
                    'BLOCKED: a push argument built from a command substitution cannot be verified against main.'
                    + DENY_TAIL
                )
            if any(MAIN_REF_RE.search(w) for w in rest):
                return 'BLOCKED: direct push to main is not allowed.' + DENY_TAIL
            if any(w in ('--all', '--mirror', '--branches') or w.startswith(('--branches=',)) for w in rest):
                return 'BLOCKED: pushing all refs (which includes main) is not allowed.' + DENY_TAIL
            deny = _check_push_refspecs(rest, targets_main(inv.dir))
            if deny:
                return deny
    return None


def _check_push_refspecs(rest: list[str], to_main: bool) -> str | None:
    """Unexpanded-variable destinations, and — while on main — bare/HEAD pushes."""
    seen_remote = main_target = has_refspec = skip = False
    for w in rest:
        if skip:
            skip = False
            continue
        if w.startswith('-'):
            if w in PUSH_ARG_OPTS or w == '--repo':
                skip = True
            continue
        if not seen_remote:
            seen_remote = True  # first positional is the remote
            continue
        # A refspec is `[+]<src>[:<dst>]`; the leading force `+` is not part of the branch name.
        ref = w.removeprefix('+')
        dst = ref.split(':', 1)[1] if ':' in ref else ref
        if '$' in dst:
            return 'BLOCKED: a push refspec built from a shell variable cannot be verified against main.' + DENY_TAIL
        has_refspec = True
        # It targets main only when its destination is main: a bare HEAD/@ (dst defaults to the
        # current branch) or an empty dst. A src:dst with a real dst goes elsewhere.
        if ':' in ref:
            main_target = main_target or not ref.split(':', 1)[1]
        else:
            main_target = main_target or ref in ('HEAD', '@')
    if to_main and (main_target or not has_refspec):
        return 'BLOCKED: pushing the current branch (main) is not allowed.' + DENY_TAIL
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    cmd = (payload.get('tool_input') or {}).get('command') or ''
    if not cmd:
        return 0
    cwd = payload.get('cwd') or os.getcwd()
    git = GitInfo()
    guarded_slug = repo_slug(git.origin_url(os.environ.get('CLAUDE_PROJECT_DIR') or os.getcwd()))
    deny = analyze(cmd, cwd, guarded_slug, git)
    if deny:
        print(deny, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
