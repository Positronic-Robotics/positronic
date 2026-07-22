# Operator workflow style (NOT loaded by default)

These are **operator / personal workflow preferences** — how an agent should *work*
(commit cadence), as opposed to what the code should *look like*. The latter — design
discipline, code style, commit-message format, history integrity (no amends), signing,
tests — is the repo's shared engineering contract and lives in the root `CLAUDE.md`,
which binds every contributor.

This file is intentionally **not referenced from `CLAUDE.md`**, so nothing here is
auto-loaded. Different operators drive their agents differently and instruct them through
their own agent config. If you want these preferences, copy them into a file your agent
actually loads (for Claude Code: `~/.claude/CLAUDE.md`, or a rules file it imports); if
your global rules already say something different, those govern your agent. Keeping them here — out of the shared contract — is what lets two
operators coexist in one repo without one operator's workflow silently overriding the
other's.

## Rules

- Don't make commits or git changes unprompted; an invoked PR workflow skill (e.g.
  `/polish`, `/push-pr`) authorizes its commits and pushes
