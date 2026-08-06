#!/usr/bin/env python3
"""Claude Code PreToolUse hook guarding this repo's `main` from the agent's Bash tool.

Blocks history-rewriting `git commit --amend`, merges / integrating pulls / direct pushes to
`main`, and `gh pr merge`. Amend rewrites history — create a new commit instead. Integrating
into main requires an explicit human/operator command run outside the agent's Bash tool, or a
receipt a human wrote from chat authorizing one named pull request (see `consume_merge_allow`).

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
import time
from dataclasses import dataclass
from pathlib import Path

DENY_TAIL = (
    ' Merging to main requires a PR and an explicit human/operator command — a named human must run the merge'
    ' themselves (e.g. via the `!` prefix or their own shell).'
)
MERGE_ESCAPE = (
    ' A named human can authorize THIS ONE merge from chat with `!allow_merge <pr>`, which the merge then spends.'
)
UNNUMBERED_MSG = (
    'BLOCKED: name the pull request to merge (`gh pr merge <number>`) — an authorization names one pull'
    ' request, so a merge that names none can never match it.'
)
MULTIPLE_MERGES_MSG = (
    'BLOCKED: merge one pull request per command — an authorization names one, and a command'
    ' merging several would have to spend the first before knowing the rest are allowed.'
)
MERGE_SUBSTITUTION_MSG = (
    'BLOCKED: a merge in a command carrying a command substitution cannot be verified — the'
    ' substitution decides at runtime what is merged, and where.'
)
MERGE_EXPANSION_MSG = (
    'BLOCKED: a merge whose words the shell expands cannot be verified — the expansion is split'
    ' into further arguments, which can select a repository the authorization never named.'
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

# gh's own name for the variable that selects a repository, read from the command and from the
# environment — two places that have to agree with gh and with each other.
GH_REPO_ENV = 'GH_REPO'

# A heredoc opener whose delimiter is QUOTED — `<<'EOF'`, `<<"EOF"`, `<<-'EOF'`. The quoting is
# what makes the body inert: it suppresses every expansion, so the text cannot execute.
HEREDOC_QUOTED_RE = re.compile(r"""<<(-?)\s*(['"])(\w+)\2""")


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


def _strip_heredoc_bodies(cmd: str) -> str:
    """`cmd` with the bodies of quoted-delimiter heredocs removed.

    A heredoc body is stdin DATA, never command words — but the tokenizer reads it as words
    anyway, so a single apostrophe in a commit message ("don't") leaves the whole command
    unparseable and every invocation in it falls back to guarded. That refused pushes to
    entirely different repos.

    Dropping the body loses nothing analyzable, because a QUOTED delimiter suppresses every
    expansion: the text reaches the command literally and cannot execute, so a backtick in such
    a body is prose rather than a substitution. Unquoted `<<EOF` is deliberately left in place —
    its body DOES expand `$(…)`, so it keeps failing closed.
    """
    lines = cmd.split('\n')
    kept: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        kept.append(line)
        i += 1
        for dash, _quote, delim in HEREDOC_QUOTED_RE.findall(line):
            while i < len(lines):
                # `<<-` lets the terminator be indented with tabs.
                terminator = lines[i].lstrip('\t') if dash else lines[i]
                i += 1
                if terminator == delim:
                    break
    return '\n'.join(kept)


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


def _substitution_bodies(cmd: str) -> list[str]:  # noqa: C901
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
    cmd = _strip_heredoc_bodies(cmd)
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


MERGE_ALLOW_DIR = Path('/var/lib/relay/merge_allows')
# Where an honoured receipt is recorded so it cannot serve a second merge. Separate from the
# receipts because that directory is root-only-writable and this hook does not run as root.
MERGE_SPENT_DIR = Path.home() / '.local' / 'state' / 'relay' / 'merge_allows_spent'
# The receipt is written by `listen_chat_receiver/relay/merge_allow.py` in the `os` repository,
# which cannot import from here and is imported by nothing here — so the file layout is the whole
# of the contract, and it is named once on each side rather than spelled out at every use.
RECEIPT_NAME = 'pr{number}.json'
RECEIPT_NUMBER = 'number'
RECEIPT_REPO = 'repo'
RECEIPT_ISSUED_AT = 'issued_at'
RECEIPT_TTL_S = 'ttl_s'
# Distinct per authorization, which is what a spend mark names. A receipt without one is refused:
# without it the mark would have to key on something that is not an identity, and a second
# authorization sharing that value would be read as already spent.
RECEIPT_ID = 'id'


def _written_by_root(path: Path, directory: Path) -> bool:
    """Whether only root could have put `path` where it is.

    Ownership is the provenance check, so it has to cover the directory too: a file root owns
    inside a directory anyone may write is a file anyone may replace.
    """
    try:
        file_stat, dir_stat = path.stat(), directory.stat()
    except OSError:
        return False
    return file_stat.st_uid == 0 and dir_stat.st_uid == 0 and not dir_stat.st_mode & 0o022


def consume_merge_allow(
    number: int,
    guarded_slug: str,
    directory: Path = MERGE_ALLOW_DIR,
    spent_dir: Path = MERGE_SPENT_DIR,
    now: float | None = None,
) -> bool:
    """Whether a human has authorized merging this pull request, spending the authorization.

    The relay writes the receipt as root when a whitelisted human types `!allow_merge` in chat,
    and refuses to write one for an injected message. A receipt cannot be forged without root,
    since only root may write into the receipt directory; and it serves one merge, since
    honouring it records a spend mark this process owns. Neither property holds against root,
    which any process on this host can reach through the account's sudo.

    An empty repo in the receipt names only a number, which resolves against the guarded
    repository.
    """
    path = directory / RECEIPT_NAME.format(number=number)
    if not _written_by_root(path, directory):
        return False
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if receipt.get(RECEIPT_NUMBER) != number:
        return False
    repo = str(receipt.get(RECEIPT_REPO) or '').casefold()
    slug = guarded_slug.casefold()
    if repo and repo != slug and repo != slug.rpartition('/')[2]:
        return False
    issued_at, ttl = receipt.get(RECEIPT_ISSUED_AT, 0), receipt.get(RECEIPT_TTL_S, 0)
    if (now if now is not None else time.time()) - issued_at >= ttl:
        return False
    identifier = str(receipt.get(RECEIPT_ID) or '')
    if not re.fullmatch(r'\w{4,64}', identifier):
        return False
    spent = spent_dir / f'pr{number}-{identifier}'
    spent_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive create, so two hooks racing the same receipt cannot both find it unspent.
        os.close(os.open(spent, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except FileExistsError:
        return False
    return True


def _carries_substitution(cmd: str) -> bool:
    """Whether `cmd` carries a substitution, or quoting that cannot be read.

    A substitution's value exists only at runtime, so it can name the pull request, the
    repository, or the variable that selects either. None of those can be weighed beforehand.
    A PROCESS substitution runs a command of its own besides — and its `<(` keeps that command
    in the same segment as the one it feeds, where only the first tool word is ever read.
    """
    try:
        segments = _segments(_strip_heredoc_bodies(cmd))
    except ValueError:
        return True
    return any(word == SUBST or '<(' in word or '>(' in word for segment in segments for word in segment)


# Shell syntax that turns one written word into arguments the text does not spell out: a parameter
# expansion, whose value is word-split; a brace list; a glob, which matches as many names as it finds.
EXPANDING_CHARS = '$*?[{'
# `NAME=value` before the command word. Its value reaches the environment whole — never word-split
# into arguments — so an expansion there says nothing about what the command is asked to do.
ASSIGNMENT_RE = re.compile(r'\w+=')


def _expands(word: str) -> bool:
    """Whether the shell builds `word` into something other than its own text.

    Quoting is what decides, and it is per REGION rather than per word: `--subject='$x'` is a
    literal while `--subject=$x` is not. A backslash escapes the character after it.
    """
    quote = ''
    i = 0
    while i < len(word):
        c = word[i]
        if quote:
            quote = '' if c == quote else quote
        elif c == '\\':
            i += 1
        elif c in '\'"':
            quote = c
        elif c in EXPANDING_CHARS:
            return True
        i += 1
    return False


def _carries_expansion(cmd: str) -> bool:
    """Whether `cmd` hands the shell an argument its own text does not spell out.

    `EXTRA='-R other/repo'; gh pr merge 566 $EXTRA` reaches gh as a repository selector no
    authorization named, and `{-R,other/repo}` and a glob do the same with no `$` in sight. The
    quoting that would make each one literal is removed before a word reaches `_segments`, so this
    reads the command's own text.
    """
    try:
        words = shlex.split(_strip_heredoc_bodies(cmd), posix=False)
    except ValueError:
        return True
    return any(_expands(w) for w in words if not ASSIGNMENT_RE.match(w))


def _sets_gh_repo(cmd: str) -> bool:
    """Whether `cmd` sets `GH_REPO` for anything it runs.

    The shell removes quotes before it reads an assignment, so `G''H_REPO=x` sets the variable
    while the raw text never spells its name. Quoting the tokenizer cannot read is itself an
    answer of yes, since what it hides could be the assignment.
    """
    try:
        segments = _segments(_strip_heredoc_bodies(cmd))
    except ValueError:
        return True
    return any(word.startswith(GH_REPO_ENV + '=') for segment in segments for word in segment)


def _gh_repo(explicit: str, inv_dir: str | None, git: GitInfo, cmd: str, gh_repo_env: str) -> str:
    """The repository a `gh` invocation acts on, or '' when that cannot be established.

    `-R` names it outright, and `GH_REPO` does the same from the environment — which is not
    something a command can be read for, so either source of it gives up. Otherwise gh resolves
    the repository from the working directory, which is why an unresolvable directory gives up
    too: a merge is not this repository's just because nothing said otherwise.
    """
    if gh_repo_env or _sets_gh_repo(cmd):
        return ''
    if explicit:
        return explicit.casefold()
    return repo_slug(git.origin_url(inv_dir)) if inv_dir is not None else ''


# gh's own spellings of the flag that selects a repository.
GH_REPO_FLAGS = ('-R', '--repo')
# Asking for help makes gh print it and merge nothing.
GH_HELP_FLAGS = ('--help', '-h')
# `gh pr merge` flags that consume the following argument, so its value is never the pull
# request being merged: `gh pr merge --subject 566 999` merges 999.
GH_MERGE_VALUE_FLAGS = frozenset({
    '-A',
    '--author-email',
    '-b',
    '--body',
    '-F',
    '--body-file',
    '--match-head-commit',
    '-t',
    '--subject',
    *GH_REPO_FLAGS,
})


def _gh_shorthand(word: str) -> tuple[str, str]:
    """The flag a `-abc` cluster names, and the value written onto it.

    pflag clusters shorthands and takes a value attached to the last of them, so `-dRowner/repo`
    is `--delete-branch --repo owner/repo`.
    """
    for i, letter in enumerate(word[1:], 1):
        short = '-' + letter
        if short in GH_HELP_FLAGS or short in GH_MERGE_VALUE_FLAGS:
            return short, word[i + 1 :].removeprefix('=')
    return '', ''


def _gh_flag(word: str) -> tuple[str, str]:
    """The flag one word names, and the value written onto it.

    An empty value means the next word carries it; ('', '') is a word naming no flag read here.
    """
    flag, eq, value = word.partition('=')
    if flag in GH_HELP_FLAGS or flag in GH_MERGE_VALUE_FLAGS:
        return flag, value if eq else ''
    return ('', '') if word.startswith('--') else _gh_shorthand(word)


def _gh_pr_merge(words: list[str]) -> tuple[bool, int | None, str]:
    """Whether this is `gh pr merge`, the pull request it names, and the repository it targets.

    The number is None unless the merge target is written as one, and the repository is '' unless
    `-R` names another — gh takes a URL or a branch there too, which no authorization can name.
    gh (cobra) accepts persistent flags between `pr` and the subcommand — `gh pr -R o/r merge 1`.
    """
    positional: list[str] = []
    repo = pending = ''
    for w in words:
        if pending:
            repo, pending = (w if pending in GH_REPO_FLAGS else repo), ''
        elif w.startswith('-'):
            flag, value = _gh_flag(w)
            if flag in GH_HELP_FLAGS:
                return False, None, ''
            pending = flag if flag and not value else ''
            if value and flag in GH_REPO_FLAGS:
                repo = value
        elif bare := re.sub(r'[)}].*$', '', w):  # a lossy parse leaves a closing delimiter glued on
            positional.append(bare)
    if not positional or positional[0] == 'help' or 'pr' not in positional:
        return False, None, ''
    after = positional[positional.index('pr') + 1 :]
    if not after or after[0] != 'merge':
        return False, None, ''
    target = after[1] if len(after) > 1 else ''
    return True, int(target) if target.isdigit() else None, repo


def _push_dest_slug(inv_dir: str | None, rest: list[str], git: GitInfo) -> str:  # noqa: C901
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


def analyze(  # noqa: C901
    cmd: str,
    cwd: str,
    guarded_slug: str,
    git: GitInfo,
    path_exists=os.path.isdir,
    _depth=0,
    in_substitution=False,
    allow_merge=consume_merge_allow,
    gh_repo_env='',
) -> str | None:
    """The deny message for `cmd`, or None to allow it."""
    guarded_slug = guarded_slug.casefold()
    # A command substitution runs its own command (same cwd) before the outer command, so a git
    # invocation hidden inside `` `…` `` / `$(…)` must be analyzed too. Bounded recursion depth
    # guards against pathological nesting.
    if _depth < 8:
        for body in _substitution_bodies(_strip_heredoc_bodies(cmd)):
            deny = analyze(body, cwd, guarded_slug, git, path_exists, _depth + 1, True, allow_merge, gh_repo_env)
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

    pending_merge = None
    for inv in invs:
        if inv.kind != 'gh':
            continue
        is_merge, number, target = _gh_pr_merge(inv.words)
        if is_merge:
            if in_substitution or _carries_substitution(cmd):
                return MERGE_SUBSTITUTION_MSG
            if _carries_expansion(cmd):
                return MERGE_EXPANSION_MSG
            if number is None:
                return UNNUMBERED_MSG
            if pending_merge is not None:
                return MULTIPLE_MERGES_MSG
            if _gh_repo(target, inv.dir, git, cmd, gh_repo_env) != guarded_slug:
                return 'BLOCKED: gh pr merge is not allowed.' + DENY_TAIL + MERGE_ESCAPE
            # Spending it here would pay for a merge a later invocation in the same command can
            # still block, so the authorization is consulted once everything else has passed.
            pending_merge = number

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

    if pending_merge is not None and not allow_merge(pending_merge, guarded_slug):
        return 'BLOCKED: gh pr merge is not allowed.' + DENY_TAIL + MERGE_ESCAPE
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
    deny = analyze(cmd, cwd, guarded_slug, git, gh_repo_env=os.environ.get(GH_REPO_ENV, ''))
    if deny:
        print(deny, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
