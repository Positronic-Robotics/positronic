#!/usr/bin/env bash
# Block history-rewriting `--amend`, and merges / integrating pulls / direct pushes to `main`,
# from the agent's Bash tool. Amend rewrites history — create a new commit instead. Integrating
# into main requires an explicit human/operator command run outside this tool.
set -euo pipefail

cmd="$(jq -r '.tool_input.command // empty')"
[ -z "$cmd" ] && exit 0

deny() {
  echo "BLOCKED: $1 Merging to main requires a PR and an explicit human/operator command — a named human must run the merge themselves (e.g. via the \`!\` prefix or their own shell)." >&2
  exit 2
}

deny_amend() {
  echo "BLOCKED: Never amend commits, create new ones instead." >&2
  exit 2
}

# Each `git …` / `gh …` invocation in the command, split on shell operators (& | ;). A subcommand
# word that only appears inside a quoted argument (`git commit -m "…push…"`) is filtered out below.
mapfile -t git_cmds < <(printf '%s' "$cmd" | grep -oE '(^|[^[:alnum:]_-])git[[:space:]][^&|;]*' | sed -E 's/^[^g]*//')
mapfile -t gh_cmds < <(printf '%s' "$cmd" | grep -oE '(^|[^[:alnum:]_-])gh[[:space:]][^&|;]*' | sed -E 's/^[^g]*//')

# Split a string into shell words, honoring '…', "…" and backslash escapes — no expansion, no
# execution. `read -ra` word-splits on whitespace blind to quotes, so a global option value with a
# space (`-c user.name='Jane Doe'`) would leak its tail as a fake subcommand; this keeps it one word.
tokenize() {
  local s="$1" c q='' tok='' started=false i=0 n=${#1}
  local -a out=()
  while [ "$i" -lt "$n" ]; do
    c="${s:i:1}"; i=$((i + 1))
    if [ -n "$q" ]; then
      if [ "$c" = "$q" ]; then q=''; else tok+="$c"; fi
    elif [ "$c" = "'" ] || [ "$c" = '"' ]; then q="$c"; started=true
    elif [ "$c" = '\' ] && [ "$i" -lt "$n" ]; then tok+="${s:i:1}"; i=$((i + 1)); started=true
    elif [[ "$c" == [[:space:]] ]]; then
      $started && { out+=("$tok"); tok=''; started=false; }
    else tok+="$c"; started=true; fi
  done
  $started && out+=("$tok")
  [ "${#out[@]}" -gt 0 ] && printf '%s\n' "${out[@]}"
}

# The subcommand of one `git …` invocation: the first bareword after git's own global options
# (some of which consume a following argument). Empty if none.
git_subcmd() {
  local skip=false w toks
  mapfile -t toks < <(tokenize "${1#git}")
  for w in "${toks[@]}"; do
    if $skip; then skip=false; continue; fi
    if [ "${w#-}" != "$w" ]; then
      case "$w" in -C | -c | --namespace | --git-dir | --work-tree | --super-prefix | --exec-path) skip=true ;; esac
      continue
    fi
    printf '%s' "$w"
    return 0
  done
  return 1
}

# invocations whose real subcommand matches $1
invocations_of() {
  local sub g
  for g in "${git_cmds[@]}"; do
    sub="$(git_subcmd "$g" || true)"
    [ "$sub" = "$1" ] && printf '%s\n' "$g"
  done
  return 0
}

# True if any `gh …` invocation is a `gh pr merge`. gh (cobra) accepts persistent flags between `pr`
# and the subcommand — `gh pr -R owner/repo merge 123` — so match the real subcommand: the first
# bareword after `pr`, skipping option words and the value of the arg-taking repo flag (-R/--repo).
gh_pr_merges() {
  local g w toks seen_pr skip
  for g in "${gh_cmds[@]}"; do
    mapfile -t toks < <(tokenize "${g#gh}")
    seen_pr=false skip=false
    for w in "${toks[@]}"; do
      if $skip; then skip=false; continue; fi
      if [ "${w#-}" != "$w" ]; then
        case "$w" in -R | --repo) skip=true ;; esac
        continue
      fi
      if ! $seen_pr; then
        [ "$w" = "pr" ] && seen_pr=true
        continue
      fi
      [ "$w" = "merge" ] && return 0
      break
    done
  done
  return 1
}

on_main() { [ "$(git branch --show-current 2>/dev/null || true)" = "main" ]; }

# The command may switch onto main before a later merge/push runs (`git checkout main && git merge x`),
# so the pre-execution branch is not the whole story. Treat a same-command checkout/switch whose
# target is main as landing on main.
switches_to_main() {
  local g
  while IFS= read -r g; do
    printf '%s' "$g" | grep -qE '(^|[[:space:]])\+?main([^[:alnum:]_/-]|$)' && return 0
  done < <(invocations_of checkout; invocations_of switch)
  return 1
}
targets_main() { on_main || switches_to_main; }

# gh pr merge
if gh_pr_merges; then
  deny "gh pr merge is not allowed."
fi

# git commit --amend rewrites history. Matched on the real `commit` subcommand (so unrelated text
# like `grep -- --amend` or `echo --amend` is not blocked) and independent of the checked-out branch.
while IFS= read -r commit_cmd; do
  printf '%s' "$commit_cmd" | grep -qE -- '(^|[[:space:]])--amend([[:space:]]|=|$)' && deny_amend
done < <(invocations_of commit)

# git merge into main. Only non-integrating cleanup (--abort/--quit) is exempt, matched as a
# standalone token so a branch/tag named `feature--abort` is not mistaken for cleanup; --continue
# COMPLETES an in-progress merge and creates the merge commit, so it stays denied. Scoped per
# invocation: a real merge chained with a cleanup — `git merge feature || git merge --abort` — is
# still denied on the merge that touches main.
if targets_main; then
  while IFS= read -r merge_cmd; do
    cleanup=false
    while IFS= read -r mt; do case "$mt" in --abort | --quit) cleanup=true ;; esac; done < <(tokenize "$merge_cmd")
    $cleanup && continue
    deny "git merge onto main is not allowed."
  done < <(invocations_of merge)
fi

# git pull integrates the fetched ref into the current branch — a merge, or a rebase that replays
# divergent history. On main that is the same human-only integration the merge guard covers; only an
# explicit fast-forward-only pull cannot create a merge commit or land divergent work.
if targets_main; then
  while IFS= read -r pull_cmd; do
    printf '%s' "$pull_cmd" | grep -qE -- '(^|[[:space:]])--ff-only([[:space:]]|=|$)' && continue
    deny "git pull onto main is not allowed (only --ff-only)."
  done < <(invocations_of pull)
fi

# git push that updates main: an explicit main refspec (from any branch), an all-refs push
# (--all/--mirror/--branches push local main from any branch), or — while on main — a push that
# does not explicitly and exclusively target a non-main branch.
mapfile -t push_cmds < <(invocations_of push)
for push_cmd in "${push_cmds[@]}"; do
  # explicit main refspec: `main`, `:main`, `HEAD:main`, `origin main`, `+main`, `/main` …
  printf '%s' "$push_cmd" | grep -qE '(:|/|[[:space:]]\+?)main([^[:alnum:]_-]|$)' \
    && deny "direct push to main is not allowed."
  # all-refs pushes carry local main regardless of the checked-out branch
  printf '%s' "$push_cmd" | grep -qE -- '(^|[[:space:]])(--all|--mirror|--branches)([[:space:]]|=|$)' \
    && deny "pushing all refs (which includes main) is not allowed."
done

# A push refspec built from a shell variable (`git push origin HEAD:$TARGET`, `git push origin $REF`)
# can't be resolved here — PreToolUse runs before expansion — and could expand to main, so refuse any
# push whose destination carries an unexpanded `$`, regardless of the checked-out branch.
for push_cmd in "${push_cmds[@]}"; do
  seen_remote=false skip_next=false
  while IFS= read -r word; do
    if $skip_next; then skip_next=false; continue; fi
    if [ "${word#-}" != "$word" ]; then
      case "$word" in --receive-pack | --exec | --push-option | --repo | -o) skip_next=true ;; esac
      continue
    fi
    if ! $seen_remote; then seen_remote=true; continue; fi
    case "${word#*:}" in *'$'*) deny "a push refspec built from a shell variable can't be verified against main." ;; esac
  done < <(tokenize "${push_cmd#*push}")
done
if targets_main; then
  for push_cmd in "${push_cmds[@]}"; do
    mapfile -t words < <(tokenize "${push_cmd#*push}")
    seen_remote=false main_target=false has_refspec=false skip_next=false
    for word in "${words[@]}"; do
      if $skip_next; then skip_next=false; continue; fi
      if [ "${word#-}" != "$word" ]; then
        # options that consume a following argument in their space-separated form
        case "$word" in --receive-pack | --exec | --push-option | --repo | -o) skip_next=true ;; esac
        continue
      fi
      if ! $seen_remote; then
        seen_remote=true # first positional is the remote
        continue
      fi
      has_refspec=true
      # A refspec is `[+]<src>[:<dst>]`; the leading force `+` is not part of the branch name. It
      # targets main only when its destination is main: a bare HEAD/@ (dst defaults to the current
      # branch) or an empty dst. A src:dst with a real dst goes elsewhere.
      ref="${word#+}"
      if [ "${ref%%:*}" != "$ref" ]; then
        [ -z "${ref#*:}" ] && main_target=true
      else
        case "$ref" in HEAD | @) main_target=true ;; esac
      fi
    done
    # A push with the `push` subcommand but no parseable refspec targets main (bare push).
    { $main_target || ! $has_refspec; } && deny "pushing the current branch (main) is not allowed."
  done
fi

exit 0
