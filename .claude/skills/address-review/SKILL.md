---
name: address-review
description: Respond to GitHub PR review comments in one pass — fetch, triage (agree or disagree), fix the valid ones, commit, push, reply to all, and resolve only the threads you fixed (declines, defers, and discussion questions stay open for the human). Use when a PR has reviewer or bot (e.g. Codex) comments to address.
allowed-tools: Bash(git add:*), Bash(git commit:*), Bash(git push:*), Bash(git diff:*), Bash(git rev-parse:*), Bash(git remote:*), Bash(awk:*), Bash(gh api:*), Bash(gh pr:*), Bash(gh repo:*), Bash(gh run:*), Bash(uv run:*), Bash(bash .claude/skills/address-review/watch.sh:*)
---

# Address Review Comments

One full cycle of responding to review feedback on the current branch's PR: read every
unresolved comment, decide on the merits, fix what is worth fixing, then commit, push,
reply, and resolve **only the threads you fixed**. Declines, defers, and discussion /
question comments get a reasoned reply and stay **open** — closing them is the human's call.

The `push-pr` skill delegates here when review comments arrive. After each push this skill
watches **both CI (GitHub Actions on the pushed commit) and the reviewer's asynchronous
re-review** in the background, and loops itself through another pass for each new comment
round or CI failure (Step 7) — so the user can walk away instead of babysitting the cycle,
and gets pinged only when it truly converges (CI green **and** the reviewer signs off) or
needs their call. A red build is never "done", no matter what the reviewer says.

## Principles

- **Judge every comment on merit.** Agree or disagree; never auto-apply. A reviewer —
  especially a bot — can be wrong, stale, or missing project context. Check intent before
  deciding: `git show <sha>`, the surrounding code, and any design docs. The code may be
  intentional.
- **Fix → push → reply → resolve, in that order.** Reply text references the commit that
  fixed it, so the fix must land first.
- **Every thread you fixed MUST be resolved — and only those.** Once a bot comment's fix is
  pushed and you have replied referencing the commit, resolve its thread: a fixed Codex
  thread left open reads as still-broken to anyone scanning the PR and leaves the conversation
  cluttered with items already handled. The two directions are symmetric and both mandatory:
  **fixed → resolve**, and **declines,
  defers, and discussion / question comments stay OPEN** — those get a reasoned reply, but
  resolving them closes a conversation that is the human's to close (especially a reviewer's
  "why…?" or "should we…?", which is an invitation to discuss, not a change request). Never
  resolve a thread just to clear the queue, and never leave a thread you fixed unresolved.
  When a comment is genuinely on the fence, leave it open.
- **Stay scoped** to the feedback. Don't sprawl into unrelated refactors.
- **Follow the repo's commit conventions** (see the `push-pr` skill). CRITICAL: never add
  `Co-Authored-By` or any AI / Claude / assistant attribution to commits, replies, or PRs.

## Step 1: Find the PR and its comments

```bash
gh pr view --json number,url,headRefName,baseRefName \
  -q '"PR #\(.number)  \(.url)  (\(.headRefName) -> \(.baseRefName))"' || { echo "No PR for this branch"; exit 1; }
PR=$(gh pr view --json number -q .number)
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)        # BASE repo — the PR and its comments live here
OWNER=${REPO%/*}; NAME=${REPO#*/}
# The PR's head branch + the repo that hosts it. In a cross-repo (fork) PR the head repo is NOT
# $REPO: the PR lives on the base, its commits belong on the fork. Derive the push target from
# the PR — never hard-code a remote.
HEAD_REF=$(gh pr view --json headRefName -q .headRefName)
HEAD_REPO=$(gh pr view --json headRepositoryOwner,headRepository -q '.headRepositoryOwner.login + "/" + .headRepository.name')
# the local remote whose URL hosts $HEAD_REPO — origin in the standard fork setup; empty only if
# no remote points at the head repo, in which case the Step 4 push fails loudly (add the remote)
HEAD_REMOTE=$(git remote -v | awk -v r="$HEAD_REPO" '$2 ~ "[:/]" r "(\\.git)?$" {print $1; exit}')
```

`REPO` (base) is where every `gh api repos/$REPO/...` read below goes; `HEAD_REMOTE` + `HEAD_REF`
(head) is where Step 4 pushes. For a same-repo PR they coincide; for a fork PR they differ —
keep them distinct.

Fetch all three comment surfaces — any can carry feedback:

```bash
# Inline (line-level) review comments — the usual source
gh api repos/$REPO/pulls/$PR/comments --paginate \
  -q '.[] | {id, user:.user.login, path, line:(.line // .original_line), in_reply_to:.in_reply_to_id, body}'

# Review summaries (top-level review bodies)
gh api repos/$REPO/pulls/$PR/reviews --paginate -q '.[] | {user:.user.login, state, body}'

# Conversation comments (issue-level)
gh api repos/$REPO/issues/$PR/comments --paginate -q '.[] | {user:.user.login, body}'
```

Ignore comments that are already your own replies, and comments on already-resolved threads
(the GraphQL query in Step 5 reports `isResolved` per thread).

## Step 2: Triage (agree or disagree)

For each open comment, decide and note severity if the bot tagged one (e.g. Codex P1/P2):

- **Fix** — valid and in scope → change the code, then resolve.
- **Partial / alternative** — valid concern, but a different fix than suggested → explain;
  resolve only if you land a concrete change, else leave open.
- **Decline** — wrong, not applicable, or contradicts a deliberate decision → reasoned
  reply, **leave open**.
- **Defer** — valid but out of scope for this PR → reply (note where it's tracked),
  **leave open**.
- **Discuss** — the reviewer is asking a question or opening a design discussion, not
  requesting a change → answer it, **leave open** for them to respond.

Present the triage as a short numbered list: comment → verdict → planned fix.

**Autonomy:** invoking this skill authorizes the cycle. Act on clear-cut fixes and clear
declines without prompting. Pause and confirm only when a fix is risky, large, or whether
to agree is genuinely unclear. The user may drop or override any item.

## Step 3: Fix

Make the code changes for the Fix / Partial items. Group related changes into a coherent
set.

**Then judge the resulting whole, not the patch.** Review fixes are local edits into a
structure that was reviewed as a whole, so they frequently spawn emergent smells: a
unification that leaves two classes structurally identical, a moved value whose old
parameter now has no reason to exist, a comment now describing pre-fix code. After the
fixes, re-read each touched file in its entirety and run the design-integrity and
comments checks from the `polish` skill (Steps 2 and 4 there) over it — explicitly
including the structural-twins check. Fix what emerges before committing; fold those
changes into the same commit and mention them in the relevant replies.

## Step 4: Verify, commit, push

Re-run the Step 1 fetch right before committing and triage anything new. A comment that lands
while you were fixing — after your first fetch but before this commit — is in a blind spot:
the Step 7 watcher only counts activity created after the commit it anchors on, so without this
re-fetch that feedback is in neither the triage list nor the watcher and gets silently skipped.
Fold late arrivals into this same pass.

Run the repo's checks before committing so pre-commit / CI find nothing new:

```bash
# only existing files — ruff fails on deleted paths (E902)
uv run --locked ruff check <existing files> && uv run --locked ruff format <existing files>   # or the repo's own lint / test / format
```

Commit (concise, imperative, no AI attribution — see `push-pr` commit style) and push to the
PR branch — but only if this pass actually changed files. A decline/defer/discuss-only round
or a flaky-CI rerun stages nothing, and an empty `git commit` exits non-zero and would abort
the pass before you reply; in that case skip the commit/push and reply against the current
`HEAD`:

```bash
git add <files>
if git diff --cached --quiet; then
  SHA=$(git rev-parse --short HEAD)        # nothing staged — keep HEAD, go reply
else
  git commit -m "<imperative summary of the review fixes>"
  git push "$HEAD_REMOTE" HEAD:"$HEAD_REF"   # head repo's remote (Step 1) — not necessarily origin; explicit refspec
  SHA=$(git rev-parse --short HEAD)
fi
```

## Step 5: Reply, then resolve

**Reply** to each comment: state what changed and the commit (`Fixed in $SHA: ...`), or the
reasoning for a decline/defer. Write the body to a file to dodge shell-quoting pitfalls
(apostrophes, backticks). The endpoint depends on which surface the comment came from:

```bash
# Inline (line-level) review comment — reply into its thread. The id must be the thread's
# TOP-LEVEL comment; GitHub rejects a reply addressed to a reply. If the comment you're
# answering is itself a follow-up, Step 1 shows its `in_reply_to` (the thread root) — use that
# id here instead of the follow-up's own id.
gh api -X POST repos/$REPO/pulls/$PR/comments/<root_comment_id>/replies -F body=@reply.txt

# Review summary (top-level review body) or issue-level conversation comment — these have no
# inline thread, so post a normal PR comment that quotes/links what you are answering:
gh api -X POST repos/$REPO/issues/$PR/comments -F body=@reply.txt
```

A combined review (e.g. one Codex body carrying several findings) arrives as a single
conversation comment, not per-finding threads — answer it with one issue-level reply that
addresses each finding; there is nothing to resolve in Step 5 for that surface.

**Resolve** every thread you *fixed* (a concrete change landed) — this is mandatory, not
optional: a fixed Codex thread you leave open keeps the PR looking unaddressed to human
reviewers. Resolution requires GraphQL (the REST API can't do it). Identify each fixed thread by
its **root** comment id — the thread's first comment; for a reply you answered that's its
`in_reply_to` (Step 1 shows it), otherwise the comment's own id. Match that root id to the thread,
then resolve:

```bash
# Each thread's node id + isResolved + its ROOT comment databaseId. Match a fixed comment to its
# thread by that root id (a review-thread reply carries in_reply_to = the root), so thread length
# never hides the match: `comments(first:1)` is the root regardless of how many replies follow.
# The first output line is the page cursor: if it shows hasNextPage=true, re-run with
# `-F after=<endCursor>` and resolve the later pages too, or a fixed thread there stays unresolved.
gh api graphql -f query='
query($owner:String!,$repo:String!,$num:Int!,$after:String){
  repository(owner:$owner,name:$repo){ pullRequest(number:$num){
    reviewThreads(first:100,after:$after){ pageInfo { hasNextPage endCursor } nodes { id isResolved comments(first:1){ nodes { databaseId } } } } } }
}' -F owner=$OWNER -F repo=$NAME -F num=$PR \
  -q '.data.repository.pullRequest.reviewThreads as $t
        | "hasNextPage=\($t.pageInfo.hasNextPage) endCursor=\($t.pageInfo.endCursor)",
          ($t.nodes[] | [.id, (.isResolved|tostring), (.comments.nodes[0].databaseId|tostring)] | @tsv)'

# Resolve a fixed thread by its node id
gh api graphql -f query='mutation($id:ID!){ resolveReviewThread(input:{threadId:$id}){ thread { isResolved } } }' -f id=<thread_node_id>
```

**Leave declined, deferred, and discussion threads OPEN** — each carries a reasoned reply,
but closing the conversation is the human's call (resolving a "why…?"/"should we…?" thread
preempts a discussion they opened on purpose). The reply records your reasoning either way,
and the human resolves when satisfied.

## Step 6: Report

Summarize:
- a table of comment → verdict → fix (with commit SHA),
- what was pushed,
- which threads you resolved (fixes only) vs left open (declines / defers / discussion),
- any follow-ups the user should track.

A bot will re-review on push and may add comments. **Don't hand the watch back to the
user** — go to Step 7, which watches for that re-review in the background and loops you
through another pass automatically until the reviewer converges.

## Step 7: Watch for convergence in the background (so the user doesn't have to)

Two signals decide whether a push actually lands the PR, and **both are asynchronous**: CI
(GitHub Actions, minutes) and the reviewer's re-review (Codex within minutes, a human whenever
they get to it). The user should poll neither. After Step 5, launch one background watcher that
blocks until something actionable happens, then loop on how it exits:

- exit **10** → a new round of reviewer comments landed (Codex or a human) → run another full
  pass (Steps 1–6), which ends right back here and relaunches the watcher;
- exit **20** → truly converged: **CI green on the pushed commit AND** a reviewer sign-off
  (Codex 👍 newer than the push, or a human approval) → give the final report, **notify the
  user**, and **stop**;
- exit **30** → a quiet interval elapsed → **relaunch the watcher and keep waiting** (don't
  stop); only after several consecutive quiet cycles ping the user that it's still watching;
- exit **40** → **CI failed on the pushed commit** → run a CI-fix pass (below), which ends
  back here and relaunches the watcher;
- exit **50** → **CI couldn't be read** (the check-runs API kept failing) → stop and tell the
  user; it's a token/permission or outage problem, not something to loop on.

CI failure is the **highest-priority** exit — a red build is never "converged" regardless of
what the reviewer says, so the watcher checks it first and gates exit 20 on CI being green. It judges
CI **only for the commit you pushed** (`HEAD` sha), so stale runs from earlier commits can't
confuse it, and acts only once a check has *completed* with a failing conclusion.

The watcher just polls `gh`, so it costs no tokens while it waits. It is **harness-tracked**:
when it exits you are re-invoked automatically with its output — that notification is the
primary wake signal, so launch it with `run_in_background` and **don't** add a short poll on top
of it. If your runtime exposes a scheduled-wake primitive (e.g. `ScheduleWakeup` under `/loop`),
also set one **long fallback heartbeat** (~20–30 min) as a safety net against a missed or dropped
exit-notification — not a poll; when it fires, re-check for unhandled comments and, if none,
reschedule it. Otherwise the primary notification plus the watcher relaunching on every exit
(see exit 30) are the wake mechanism. Its echoed `WATCH <code>` line names the next
action, so the loop survives even if this skill text has fallen out of context (re-read this
file if unsure).

Launch the watcher (shipped alongside this skill) right after Step 5, in the background:

```bash
bash .claude/skills/address-review/watch.sh
```

`watch.sh` is self-contained — it reads the PR, repo, pushed SHA, and reviewer set from the
current checkout, polls `gh` every 90s for up to 25 minutes, and echoes a `WATCH <code>` line
then exits with that code (each handled below). It lives in its own file rather than inline so a
single allow rule, `Bash(bash .claude/skills/address-review/watch.sh:*)`, pre-approves the whole
loop: Claude Code does not re-parse a script file's internal commands for permission, whereas an
inline compound command would have every subcommand (`jq`, `date`, `sleep`, `[`, …) checked
independently and could stop for a prompt mid-run while the user has walked away. The decision
logic and its rationale live in the script's comments — read `watch.sh` before changing it.

When the watcher exits and you are re-invoked:

- **exit 10** — run Steps 1–6 on the new comments, then relaunch the watcher.
  **Stop-guard:** if the round only re-flags comments you already declined (Codex re-posts
  declined items as fresh threads), that is **not** convergence — reply once more pointing at
  your prior reasoning, then **stop and surface it for the user**; never loop into forcing a
  fix you disagree with.
- **exit 20** — converged: CI is green on the pushed commit, a reviewer signed off (Codex 👍
  newer than your push, or a human approval), and every comment carries your reply (open
  declines / defers / discussions are fine). Give the final report and **notify the user** (a
  push notification if available) that the review loop is done — they walked away expecting to
  be pinged.
- **exit 30** — a quiet interval elapsed with no CI failure, new round, or sign-off. The PR
  isn't done, so **don't stop**: relaunch the watcher to keep waiting (reviewers and slow CI can
  take far longer than one interval). Only after several consecutive quiet cycles — i.e. a long
  genuine silence — ping the user that it's still watching, then keep watching unless they say to
  stop. The point is to survive idle periods, not to hand the wait back at the first timeout.
- **exit 40** — CI failed on the pushed commit. Re-enter the cycle treating the failing jobs
  as feedback, exactly like a review comment:
  1. **Identify** what failed: `gh pr checks $PR` for the overview, then read the failing
     job's log — `gh run view --job <job-id> --log-failed` (the job id is in the check URL) —
     for the actual error, not just the red X.
  2. **Triage on merit**, the same Fix / flaky / decline split as a comment:
     - *Real defect in this PR's changes* → fix it, verify locally if you can, push (the
       watcher relaunches on the new commit).
     - *Flaky / infra* (network blip, runner hiccup, transient) → `gh run rerun <run-id>
       --failed` and relaunch the watcher; change no code.
     - *Consequence of a deliberate design choice in the PR* (e.g. an interpreter version,
       platform, or dependency the change knowingly can't satisfy at the pinned date) → this
       is **not** a mechanical fix. **Stop and surface it to the user with options** (narrow
       scope, pin/exclude, accept the gap); don't silently commit-thrash to force it green.
  3. **Stop-guard:** never retry the same fix more than once or twice. If a CI failure
     persists after your fix, stop and surface it rather than churning the PR with commits.
- **exit 50** — the watcher couldn't read check runs for the pushed commit even after retries.
  This is an environment problem (most often the token lacks the checks-read scope, or a GitHub
  outage), not reviewer feedback. **Stop and tell the user** the SHA and the likely cause —
  don't relaunch, which would just hit the same wall.
