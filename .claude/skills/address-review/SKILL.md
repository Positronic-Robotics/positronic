---
name: address-review
description: Respond to GitHub PR review comments in one pass — fetch, triage (agree or disagree), fix the valid ones, commit, push, reply to all, and resolve only the threads you fixed (declines, defers, and discussion questions stay open for the human). Use when a PR has reviewer or bot (e.g. Codex) comments to address.
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
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
OWNER=${REPO%/*}; NAME=${REPO#*/}
```

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

Run the repo's checks before committing so pre-commit / CI find nothing new:

```bash
# only existing files — ruff fails on deleted paths (E902)
uv run --locked ruff check <existing files> && uv run --locked ruff format <existing files>   # or the repo's own lint / test / format
```

Commit (concise, imperative, no AI attribution — see `push-pr` commit style) and push to the
PR branch:

```bash
git add <files>
git commit -m "<imperative summary of the review fixes>"
git push origin HEAD
SHA=$(git rev-parse --short HEAD)
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
reviewers. Resolution requires GraphQL (the REST API can't do it). Map each comment's
`databaseId` to its thread node id, then resolve:

```bash
# Thread node id + isResolved + the comment databaseIds it contains. `first:100` covers all
# but the largest PRs; if `hasNextPage` is true, page with `after: <endCursor>` rather than
# silently dropping the overflow threads (a fixed thread that isn't returned stays unresolved).
# `comments(first:10)` is enough to match the original review comment, which is always first.
gh api graphql -f query='
query($owner:String!,$repo:String!,$num:Int!){
  repository(owner:$owner,name:$repo){ pullRequest(number:$num){
    reviewThreads(first:100){ pageInfo { hasNextPage endCursor } nodes { id isResolved comments(first:10){ nodes { databaseId } } } } } }
}' -F owner=$OWNER -F repo=$NAME -F num=$PR \
  -q '.data.repository.pullRequest.reviewThreads.nodes[] | [.id, (.isResolved|tostring), ([.comments.nodes[].databaseId]|map(tostring)|join(","))] | @tsv'

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
- exit **30** → nothing came back within the cap → tell the user it went quiet;
- exit **40** → **CI failed on the pushed commit** → run a CI-fix pass (below), which ends
  back here and relaunches the watcher.

CI failure is the **highest-priority** exit — a red build is never "converged" regardless of
what the reviewer says, so the watcher checks it first and gates exit 20 on CI being green. It judges
CI **only for the commit you pushed** (`HEAD` sha), so stale runs from earlier commits can't
confuse it, and acts only once a check has *completed* with a failing conclusion.

The watcher just polls `gh`, so it costs no tokens while it waits. It is **harness-tracked**:
when it exits you are re-invoked automatically with its output — so launch it with
`run_in_background` and **don't** also schedule a wakeup to poll for it. Its echoed
`WATCH <code>` line names the next action, so the loop survives even if this skill text has
fallen out of context (re-read this file if unsure).

Launch this right after Step 5, in the background:

```bash
set -e
PR=$(gh pr view --json number -q .number)
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
SHA=$(git rev-parse HEAD)                              # judge CI only for the commit we pushed
# Anchor to the pushed commit's own timestamp, NOT to when this watcher launches: the
# reply/resolve work in Step 5 takes a while and a reviewer can respond in that window, so a
# launch-time anchor would fold those responses into the baseline and miss them. Counting
# reviewer activity created after the commit time catches them no matter when the loop starts.
# In UTC with a Z suffix so it sorts lexically against GitHub's UTC timestamps (a local-offset
# %cI like +03:00 would mis-compare against their Z form).
SINCE=$(TZ=UTC0 git show -s --date=format-local:'%Y-%m-%dT%H:%M:%SZ' --format=%cd HEAD)
ME=$(gh api user -q .login)                            # the PR author — exclude your own replies
# A tracked reviewer is the Codex bot OR any human; other bots (e.g. the repo's Claude review
# workflow) are intentionally NOT gated on. So: login matches "codex", or is a non-bot that
# isn't you.
reviewer="select((.user.login|test(\"codex\")) or ((.user.login|endswith(\"[bot]\")|not) and (.user.login!=\"$ME\")))"
deadline=$(( $(date +%s) + 1500 ))                    # 25-minute cap
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 90
  # --- CI on the pushed commit (highest priority; a failing check is actionable at once) ---
  cr=$(gh api "repos/$REPO/commits/$SHA/check-runs?per_page=100" 2>/dev/null || echo '{}')
  # Any completed check whose conclusion is outside the passing set counts as failed — this
  # covers failure/timed_out plus cancelled/action_required/stale/startup_failure, so a
  # non-passing-but-not-"failure" check can never be mistaken for green.
  fail=$(echo "$cr"    | jq '[.check_runs[]? | select(.status=="completed") | select((.conclusion // "") as $c | (["success","neutral","skipped"] | index($c)) == null)] | length')
  pending=$(echo "$cr" | jq '[.check_runs[]? | select(.status!="completed")] | length')
  total=$(echo "$cr"   | jq '[.check_runs[]?] | length')
  if [ "$fail" -gt 0 ]; then
    echo "WATCH 40: CI failed on $SHA ($fail failing check(s)) — run a CI-fix pass"; exit 40
  fi
  # --- new reviewer activity since the push, any of the three surfaces (Codex or a human) ---
  # Count with `--paginate --slurp | jq`, NOT `--paginate -q`: with -q, gh runs the filter once
  # per page and prints one count per page, so the arithmetic below breaks on a multi-page PR.
  # --slurp wraps all pages into a single array (`.[][]` flattens it). Only COMMENTED /
  # CHANGES_REQUESTED reviews are actionable; an APPROVED (or DISMISSED) review is a sign-off,
  # left to the convergence block — counting it here would exit 10 instead of 20 and, with no
  # new commit, re-count the same approval forever.
  new=$(gh api repos/$REPO/pulls/$PR/reviews   --paginate --slurp | jq "[.[][] | $reviewer | select(.submitted_at > \"$SINCE\") | select(.state==\"COMMENTED\" or .state==\"CHANGES_REQUESTED\")] | length")
  new=$(( new + $(gh api repos/$REPO/pulls/$PR/comments --paginate --slurp | jq "[.[][] | $reviewer | select(.created_at > \"$SINCE\")] | length") ))
  new=$(( new + $(gh api repos/$REPO/issues/$PR/comments --paginate --slurp | jq "[.[][] | $reviewer | select(.created_at > \"$SINCE\")] | length") ))
  if [ "$new" -gt 0 ]; then
    echo "WATCH 10: new reviewer round — run another address-review pass (Steps 1-6)"; exit 10
  fi
  # --- convergence: CI green (all checks done, none failing) AND a reviewer sign-off that is
  # newer than the push. Codex signs off with a 👍 (+1) reaction; a human signs off with an
  # APPROVED review. Both are gated on `> $SINCE` — a stale approval predating the push does not
  # certify the pushed changes. 👀 (eyes) means "reviewing now" — ignore it.
  ci_green=false
  [ "$total" -gt 0 ] && [ "$pending" -eq 0 ] && [ "$fail" -eq 0 ] && ci_green=true
  plus1=$(gh api repos/$REPO/issues/$PR/reactions \
    -H "Accept: application/vnd.github.squirrel-girl-preview+json" --paginate --slurp \
    | jq "[.[][] | select(.user.login|test(\"codex\")) | select(.content==\"+1\") | select(.created_at > \"$SINCE\")] | length")
  approved=$(gh api repos/$REPO/pulls/$PR/reviews --paginate --slurp | jq "[.[][] | $reviewer | select(.state==\"APPROVED\") | select(.submitted_at > \"$SINCE\")] | length")
  if [ "$ci_green" = true ] && { [ "$plus1" -gt 0 ] || [ "$approved" -gt 0 ]; }; then
    echo "WATCH 20: converged (CI green + reviewer sign-off) — done"; exit 20
  fi
done
echo "WATCH 30: no verdict after 25m — tell the user"; exit 30
```

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
- **exit 30** — went quiet. Don't loop blindly: tell the user, and relaunch the watcher only
  if they want to keep waiting.
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
