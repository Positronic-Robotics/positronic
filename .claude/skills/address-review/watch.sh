#!/usr/bin/env bash
# Background convergence watcher for the address-review skill. Launched in the background after a
# pass; blocks until something actionable happens, then echoes a `WATCH <code>` line and exits
# with that code (10 new round, 20 converged, 30 quiet, 40 CI failed). Self-contained — it reads
# the PR/repo from the current checkout. Lives in its own file so a single allow rule
# (`Bash(bash .claude/skills/address-review/watch.sh:*)`) pre-approves the whole loop: Claude
# Code does not re-parse a script file's internal commands for permission, whereas an inline
# compound command would have every subcommand (jq, date, sleep, `[`, …) checked independently.
set -e
PR=$(gh pr view --json number -q .number)
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
OWNER=${REPO%/*}; NAME=${REPO#*/}                      # for the GraphQL thread query below
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
# Re-entry can't key off SINCE alone: a decline-only pass makes no commit (SINCE wouldn't
# advance, so the handled comment would re-trigger forever), and your own reply can post-date a
# reviewer comment that arrived mid-pass (burying it under a later anchor). Inline feedback is
# therefore detected structurally below — an unresolved thread whose last comment is a
# reviewer's, no timestamps. The review-summary and issue-comment surfaces have no per-item
# resolved state, so they fall back to a timestamp: SINCE_DONE = the later of the commit and
# your latest reply on ANY surface. Include inline replies, not just issue-level ones — when you
# answer a Codex round inline (e.g. a decline, no commit), its COMMENTED review row is still
# `> SINCE` and would re-trigger forever unless your inline reply also advances the floor.
me_issue=$(gh api repos/$REPO/issues/$PR/comments --paginate --slurp | jq -r "[.[][] | select(.user.login==\"$ME\") | .created_at] | max // \"\"")
me_inline=$(gh api repos/$REPO/pulls/$PR/comments --paginate --slurp | jq -r "[.[][] | select(.user.login==\"$ME\") | .created_at] | max // \"\"")
SINCE_DONE=$(printf '%s\n%s\n%s\n' "$SINCE" "$me_issue" "$me_inline" | sort | tail -1)
deadline=$(( $(date +%s) + 1500 ))                    # 25-minute cap
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 90
  # --- CI on the pushed commit (highest priority; a failing check is actionable at once) ---
  # --paginate --slurp so a commit with >100 check runs isn't judged on its first page alone —
  # a failing or pending job on a later page must not be missed. Slurp yields an array of page
  # objects, so the runs are at `.[].check_runs[]`. Do NOT fabricate an empty list on API failure:
  # that would hide a token/permission or outage problem behind a fake "no checks" reading
  # (total=0 blocks both exit 40 and ci_green, so the persisting loop would spin forever). Retry a
  # few times for a transient blip; if it stays unreadable, surface it and stop (exit 50).
  cr=""
  for try in 1 2 3; do
    cr=$(gh api "repos/$REPO/commits/$SHA/check-runs?per_page=100" --paginate --slurp 2>/dev/null) && break
    cr=""; sleep 5
  done
  if [ -z "$cr" ]; then
    echo "WATCH 50: cannot read check runs for $SHA after 3 tries — token checks-read scope or a GitHub outage; surface to the user"; exit 50
  fi
  # Any completed check whose conclusion is outside the passing set counts as failed — this
  # covers failure/timed_out plus cancelled/action_required/stale/startup_failure, so a
  # non-passing-but-not-"failure" check can never be mistaken for green.
  fail=$(echo "$cr"    | jq '[.[].check_runs[]? | select(.status=="completed") | select((.conclusion // "") as $c | (["success","neutral","skipped"] | index($c)) == null)] | length')
  pending=$(echo "$cr" | jq '[.[].check_runs[]? | select(.status!="completed")] | length')
  total=$(echo "$cr"   | jq '[.[].check_runs[]?] | length')
  if [ "$fail" -gt 0 ]; then
    echo "WATCH 40: CI failed on $SHA ($fail failing check(s)) — run a CI-fix pass"; exit 40
  fi
  # --- new reviewer feedback to re-enter on ---
  # Inline (the surface Codex/humans use most): race-free and timestamp-free — count UNRESOLVED
  # threads whose latest comment is a reviewer's (not you). A thread you handled has your reply
  # as its last comment (fixed ones are also resolved), so neither handled nor declined threads
  # re-trigger; only an unanswered reviewer comment does. Paginate by cursor: threads come back
  # oldest-first, so on a PR with >100 threads a fresh unanswered round sits on the LAST page.
  new=0; after=""; more=true
  while [ "$more" = "true" ]; do
    targs=(-F o=$OWNER -F r=$NAME -F n=$PR); [ -n "$after" ] && targs+=(-F after="$after")
    page=$(gh api graphql -f query='query($o:String!,$r:String!,$n:Int!,$after:String){repository(owner:$o,name:$r){pullRequest(number:$n){reviewThreads(first:100,after:$after){pageInfo{hasNextPage endCursor} nodes{isResolved comments(last:1){nodes{author{login}}}}}}}}' "${targs[@]}")
    new=$(( new + $(echo "$page" | jq "[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved==false) | (.comments.nodes[-1].author.login // \"\") | select(test(\"codex\") or ((endswith(\"[bot]\")|not) and (. != \"$ME\")))] | length") ))
    more=$(echo "$page" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
    after=$(echo "$page" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.endCursor')
  done
  # Review summaries + issue comments have no per-item state, so they're time-anchored on
  # SINCE_DONE with `--paginate --slurp | jq` (NOT `--paginate -q`, which prints one count per
  # page and breaks the arithmetic). COMMENTED/CHANGES_REQUESTED only — an APPROVED/DISMISSED
  # review is a sign-off for the convergence block. This timestamp path is best-effort, but a
  # round that also carries inline comments is caught race-free above, so the gap is only a
  # body-only review/comment landing in your reply window — rare, and it surfaces on next trigger.
  new=$(( new + $(gh api repos/$REPO/pulls/$PR/reviews --paginate --slurp | jq "[.[][] | $reviewer | select(.submitted_at > \"$SINCE_DONE\") | select(.state==\"COMMENTED\" or .state==\"CHANGES_REQUESTED\")] | length") ))
  new=$(( new + $(gh api repos/$REPO/issues/$PR/comments --paginate --slurp | jq "[.[][] | $reviewer | select(.created_at > \"$SINCE_DONE\")] | length") ))
  if [ "$new" -gt 0 ]; then
    echo "WATCH 10: new reviewer round — run another address-review pass (Steps 1-6)"; exit 10
  fi
  # --- convergence: CI green (all checks done, none failing) AND a reviewer sign-off. A human
  # signs off via an APPROVED review pinned to the exact SHA (commit_id == $SHA), so an approval of
  # an earlier state can't certify this one. Codex signs off with a 👍 (+1) reaction, which carries
  # no commit id and so can't be SHA-pinned the way a review can. Anchor it as tightly as the API
  # allows: created_at > $SINCE_DONE (the later of the pushed commit and your latest reply ≈ push
  # time), which drops a 👍 left before this round's reply. A residual race is unavoidable — a
  # delayed Codex 👍 for an earlier clean state landing after this reply would still count, since a
  # reaction is not tied to a SHA. 👀 (eyes) = "reviewing now" — ignore.
  ci_green=false
  [ "$total" -gt 0 ] && [ "$pending" -eq 0 ] && [ "$fail" -eq 0 ] && ci_green=true
  plus1=$(gh api repos/$REPO/issues/$PR/reactions \
    -H "Accept: application/vnd.github.squirrel-girl-preview+json" --paginate --slurp \
    | jq "[.[][] | select(.user.login|test(\"codex\")) | select(.content==\"+1\") | select(.created_at > \"$SINCE_DONE\")] | length")
  approved=$(gh api repos/$REPO/pulls/$PR/reviews --paginate --slurp | jq "[.[][] | $reviewer | select(.state==\"APPROVED\") | select(.commit_id == \"$SHA\")] | length")
  if [ "$ci_green" = true ] && { [ "$plus1" -gt 0 ] || [ "$approved" -gt 0 ]; }; then
    echo "WATCH 20: converged (CI green + reviewer sign-off) — done"; exit 20
  fi
done
echo "WATCH 30: quiet interval elapsed — relaunch and keep waiting (ping the user only after several quiet cycles)"; exit 30
