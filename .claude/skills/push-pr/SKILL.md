---
name: push-pr
description: Push current branch to origin and create a PR to upstream. Checks for uncommitted changes first.
---

# Push and Create PR

This skill pushes the current branch to origin (fork) and creates a PR to upstream (main repo).

## Workflow

1. **Check for uncommitted changes**
   ```bash
   git status --porcelain
   ```
   If there are changes, ask the user if they want to commit them first.

2. **If committing**, follow the commit style (see below), then continue.

3. **Push to origin**
   ```bash
   git push -u origin HEAD
   ```

4. **Create PR to upstream**
   ```bash
   gh pr create --repo Positronic-Robotics/positronic --base main --head vertix:BRANCH_NAME \
     --title "Title" --body "$(cat <<'EOF'
   ## Summary
   <bullet points describing the change>

   ## Test plan
   <how to test, or "Tested locally" if applicable>
   EOF
   )"
   ```

## Commit Message Style

Follow the project's commit message conventions:
- Short, imperative sentences (e.g., "Fix wrong type", "Add feature X")
- Use backticks for code references (e.g., "Unify `GrootActionDecoder` and `GrootObservationEncoder`")
- No trailing period for short messages
- Examples from this repo:
  - `Avoid loading object dtype in SimpleSignal`
  - `Fix groot metadata so that lerobot dataset can work with multiple keys`
  - `6D rotation representation`
  - `Unify GR00T action decoding and observation encoding`

## Analyzing Changes for PR

**Important**: Before writing the PR title and description:

1. **Look at ALL commits on the branch** (not just the latest):
   ```bash
   git log main..HEAD --oneline
   ```

2. **Review the full diff from main**:
   ```bash
   git diff main...HEAD --stat
   ```

3. **Identify the MAJOR change** - what is the primary purpose of this branch?
   - Multiple small fixes supporting one feature = describe the feature
   - Refactoring + new capability = focus on the new capability
   - Don't list every small change; summarize the intent

4. **Title should capture the major change**, not enumerate commits

## Remotes

| Remote | Repository | Purpose |
|--------|------------|---------|
| `origin` | vertix/positronic-open | Push branches here |
| `upstream` | Positronic-Robotics/positronic | Create PRs here |

## After PR Creation

Show the PR URL so user can review it.
