# PR #479 rework — implementation plan (temp doc, do NOT commit)

Handoff plan for a fresh agent. Covers: Sergey's 2026-07-24 review on PR #479, the documentation
work Vladimir asked for, and the review-loop infrastructure changes. Read this whole doc before
touching code. Keep this file untracked — it is lane-local planning, not PR content.

## Context and current state

- **PR:** [#479 — Opt-in wall-clock telemetry for `positronic eval run` + `eval timing-report`](https://github.com/Positronic-Robotics/positronic/pull/479),
  branch `eval-timing`, lane worktree `/home/posi/lanes/eval-timing`.
- **State (2026-07-24):** rebased onto `origin/main` at `2658ca1` (includes the docs framework:
  `docs/architecture.md` from PRs #503/#506, `docs/evaluation.md` from #502). Full suite green
  (683 passed, 7 skipped). Force-pushed; review tree `pr479-eval-timing` refreshed.
- **Sergey's five review comments** (all accepted):
  1. [docs/eval-timing.md:15](https://github.com/Positronic-Robotics/positronic/pull/479#discussion_r3644237251) — fresh-`output_dir` rule is a smell; falls away with (3).
  2. [eval_timing.py:232](https://github.com/Positronic-Robotics/positronic/pull/479#discussion_r3644545555) — `record_env_phases` needs a general signature (mapping/`**kwargs`).
  3. [eval_timing.py:253](https://github.com/Positronic-Robotics/positronic/pull/479#discussion_r3644573385) — GPU sampler becomes a pimm `ControlSystem` feeding `DsWriterAgent`; kills `gpu_dmon.log` + parsing + timestamp join + fresh-dir rule. Policy-endpoint GPU stays outside (different box).
  4. [ds_writer_agent.py:144](https://github.com/Positronic-Robotics/positronic/pull/479#discussion_r3644640836) — writer must not know inference exists; fold drains into one opaque `(name, value)` iterator.
  5. [episode.py:18](https://github.com/Positronic-Robotics/positronic/pull/479#discussion_r3644884509) — revert `TELEMETRY_PREFIX` + `content_signals` from dataset core; the proper fix is [#508](https://github.com/Positronic-Robotics/positronic/issues/508) (signal defaults; bounds intersect only non-defaulted signals).
- **Open Codex thread to carry into the batch:** [P2, comment id 3645226176](https://github.com/Positronic-Robotics/positronic/pull/479#discussion_r3645226176)
  — `timing_report.py:247`: a `--gpu_policy_log` collected with plain `nvidia-smi dmon` has no `fb`
  column, so the parser skips every row and silently reports policy GPU util/VRAM as 0. Valid; fix
  in Phase 1 step 5.

## DECIDED (Vladimir, 2026-07-24): revert the core, keep the signals, ACCEPT the bounds shrink

On Sergey's episode.py comment, Vladimir's call: revert `episode.py` fully and keep writing the
per-tick `timing.*` signals **as-is** — and simply **accept the shrunken episode bounds** (the
tick lost at the window edge, a visualization/replay artifact) **until #508 or a variation of it
lands**. No statics-only interim, no gating this PR on #508.

What that means concretely:
- Episode bounds go back to the plain intersection over ALL signals. A `timing.*` stream's first
  sample lands a beat after episode start (and the 1 Hz GPU stream up to ~1 s in), so `start_ts`
  moves right by that much. Known, accepted, temporary.
- The PR description and the episode.py thread reply state this explicitly and **reference #508
  as the follow-up** that removes the distortion (timing signals then declare `default=0.0`, their
  true pre-first-sample value; bounds stop seeing them).
- Because the shrink is accepted, the `GpuMonitor` ControlSystem lands **in this PR**, exactly as
  Sergey's comment proposes.

## Phase 0 — preflight

1. Work in this lane worktree. Verify `git config user.email` == `mc.vertix@gmail.com` before any
   commit (signing key requirement).
2. `git fetch origin main` and rebase if main moved; suite must be green before starting
   (`uv run --locked pytest --no-cov`).
3. Review-loop discipline for every phase: batch ALL fixes → one push → one
   `mcp__publish_pr__request_codex_review` → wait via codex_watch. Never per-finding pushes. Fix
   P0 always; fix P1 with a concrete failure scenario; decline the rest with a one-line reason.
   Max ~2 automated re-review rounds, then hand residue to the human operator driving the work.

## Phase 1 — PR #479 rework (one batched push, commit-per-step)

### Step 1 — revert the episode-core changes (comment 5)
- Remove `TELEMETRY_PREFIX`, `content_signals`, and the content/telemetry split in
  `_EpisodeTimeIndexer` from `positronic/dataset/episode.py`. Bounds and `time[...]` go back to
  plain `signals`, over every signal including `timing.*`.
- Re-route the consumers that were pointed at `content_signals`
  (`positronic/vendors/lance/convert.py`, `positronic/dataset/tests/test_episode.py`) back to
  `signals`.
- Move the `'timing.'` name prefix constant into `positronic/eval_timing.py` — a naming convention
  of the producer, not dataset dispatch. The dataset core must not reference it.
- Update `positronic/dataset/CLAUDE.md`: delete the TELEMETRY_PREFIX bounds paragraph (it
  documents the reverted design).
- Fix up any test that asserted telemetry-excluded bounds: the expected bounds now include the
  timing streams (the accepted shrink).

### Step 2 — writer agnosticism (comment 4)
- In `positronic/eval_timing.py`: fold `drain_step()` + `step_signal_items()` + `drain_infers()`
  into one `drain_signal_items() -> Iterator[tuple[str, float]]` (a step yields ≤1 pair per phase;
  infers yield N pairs sharing the `timing.infer_ms` name). `INFER_MS_SIGNAL` knowledge stays in
  `eval_timing`.
- In `positronic/dataset/ds_writer_agent.py`: `_append_timing` collapses to
  `for name, value in <drain>(): _append(ep_writer, name, value, ts_ns)`.
- **Structural completion (proposed to Vladimir, not redirected):** inject the timing hooks into
  `DsWriterAgent.__init__` as a small callable bundle — per-tick drain, `finish_episode` statics,
  `discard` — defaulting to a no-op, instead of importing `eval_timing`. Both construction sites
  are in `positronic/wire.py` (~34, ~92). The `eval_timing.timed(Phase.RECORD_IO)` spans inside
  the writer (~247/302) move behind the same bundle so the dataset package ends with **zero**
  `eval_timing` imports (enables the boundary test in step 7).

### Step 3 — GPU sampler as a ControlSystem (comments 1 + 3)
- New `GpuMonitor(pimm.ControlSystem)` — place next to `positronic/drivers/sound.py` (the existing
  non-robot CS precedent). One constructor param: `sampling_hz` (default 1.0).
- Each sample reports the box's global GPU stats (util %, memory used — everything the report
  needs; no per-process attribution, per Sergey). Prefer one
  `nvidia-smi --query-gpu=... --format=csv,noheader` invocation per sample over `dmon` stream
  parsing — the column-position pitfall dies with the log file. Pin to the first CUDA-visible
  device else 0 (keep the existing pinning logic + comment from `_start_gpu_sampler`). No
  `nvidia-smi` on PATH → emit nothing, log once.
- Emit one bundled sample per tick; `DsWriterAgent`'s existing `expand_suffixed` mechanism fans it
  out to suffixed signals — target names `timing.gpu_*`.
- Wire in `positronic/wire.py` as a **background** system (`World.start(..., background=[...])`,
  see `pimm/world.py:688`) with its output connected as a `DsWriterAgent` input, gated on the same
  `--timing` path that calls `eval_timing.bind` (`positronic/cli/eval/run.py:202`). Samples land
  per-episode through the normal writer path; the writer already stamps both clocks
  (`primary_ts` world clock + `extra_ts['system']` real epoch), so virtual time is a non-issue.
- Delete from `positronic/eval_timing.py`: `_start_gpu_sampler`, `GPU_LOG_FILENAME`, the sampler
  handling in `bind()` (bind keeps only the collector token; drop the `out_dir` param if nothing
  else uses it).
- Delete the fresh-`output_dir` rejection in `positronic/cli/eval/run.py` — it existed only to
  protect the log file (resolves comment 1).
- Between-episodes: the harness attributes inter-episode wall into the next episode (existing
  statics design); GPU samples between episodes are not recorded — fine, sim GPU is already
  scoped to timed spans (`4c35955`).

### Step 4 — generalize `record_env_phases` (comment 2)
- `positronic/eval_timing.py:232` → `record_env_phases(phases: Mapping[str, float])` (or
  `**phases: float`); each reported phase becomes signal `timing.env_<phase>`. Nothing in our code
  enumerates phase names — the env owns its decomposition.
- `positronic/simulator/env_server/proxy.py:138` passes the env's timing dict through unchanged.
- `StepTiming` (`eval_timing.py:57`): keep client-measured fields (`env_step_s`, `record_io_s`,
  `env_client_s`) as dataclass fields; server-reported phases become a `dict[str, float]`
  (genuinely unbounded key space — a real dict per repo style) merged into `drain_signal_items()`
  output.
- `timing_report.py`'s `env_step_split` reduce iterates whatever `timing.env_*` columns exist
  instead of the fixed three.

### Step 5 — timing_report updates + fix the open Codex P2
- Sim-box GPU numbers reduce from the episodes' `timing.gpu_*` signals; delete the sim-side
  `gpu_dmon.log` discovery/parse (`timing_report.py:403` + the sim half of `_parse_dmon`).
- `--gpu_policy_log` (user-supplied dmon log, different box) stays. Fix
  [Codex P2 3645226176](https://github.com/Positronic-Robotics/positronic/pull/479#discussion_r3645226176):
  a log without the `fb` column (plain `dmon`, not `-s um`) must fail loudly naming the missing
  column and the `-s um` requirement — never silently report 0. State `-s um` in the doc (step 6).
  Reply + resolve that thread in the batch round.

### Step 6 — documentation (same PR, same commits as the code they describe)
- **`docs/eval-timing.md`:** collect section — `timing.gpu_*` signals replace the side-file
  paragraph; drop the fresh-`output_dir` requirement and its rationale; add one caveat line:
  `timing.*` streams start at their first activity, so episode bounds (a visualization/replay
  concern) shrink by up to one sampling interval until
  [#508](https://github.com/Positronic-Robotics/positronic/issues/508) lands. Policy-GPU section
  states the `nvidia-smi dmon -s um` requirement.
- **`positronic/dataset/CLAUDE.md`** — add a **"The dataset holds no opinions"** section:
  - The core (Signal/Episode/Dataset/writer) is agnostic to the nature, source, and structure of
    every signal: no reserved names, no name-based dispatch, no signal classes. Recording writes
    what happened at original resolution/fidelity; interpretation belongs to consumers.
  - **Consumers:** two classes. *Replay/debug* — code we control; decides sparse/dense
    reconciliation and derives any boundary notion it needs from raw signals. *Training-set
    conversion* — codec-driven; consumes only codec-declared keys, so an unknown signal is a
    no-op by construction. This is current fact, verified 2026-07-24: `apply_codec` wraps the
    dataset in `TransformedDataset(codec.training_encoder)`, every vendor `training_encoder` is a
    pure `Derive` projection (yields only its declared keys; base passthrough is opt-in via
    `Group(Identity(), ...)` and unused in training encoders), `convert_to_lerobot_dataset`
    asserts codec-set `lerobot_features` meta, and LeRobot `add_frame` validates frames against
    declared features. State the contract: a training encoder is a pure projection — never add
    `Identity()` passthrough to one.
  - Bounds: a convenience projection over signals, not semantics; a consumer needing a different
    episode extent derives it from raw signals. (After #508: intersection over non-defaulted
    signals.)
- **`docs/architecture.md`** — one new derived decision, ~1 paragraph, in the doc's voice
  ("stated with what forces it"): **"Telemetry is a producer like any other."** Anything sampling
  the world over time is a ControlSystem feeding the standard writer path; per-episode facts are
  statics; aggregates are offline reduces over the recording; no side files or ambient
  subprocesses. Forced by "components are functions over flowing data" + "every run produces a
  Positronic dataset". (Sergey wrote this doc — keep the entry tight and expect his edit.)

### Step 7 — invariant tests (add to existing test packages, no new suites without need)
- **Boundary:** `positronic/dataset/*` modules import nothing from `positronic.eval_timing`
  (import-graph assertion — possible once step 2's injection lands) and contain no `'timing.'`
  literals.
- **Writer agnosticism:** the writer records arbitrary `(name, value)` pairs unchanged — no name
  has behavior.
- **Converter no-op golden** (in scope — it passes today and pins the contract): add an unknown
  signal to a fixture episode; the codec-transformed sample (`apply_codec` path) is unchanged.
  This is the structural-whitelist guarantee from step 6's consumers section.

### Batch round
One push of steps 1–7 → reply to all five of Sergey's threads (per-thread: what landed, commit
link; the episode.py reply names the accepted bounds shrink and references #508 as the follow-up)
and the Codex P2 thread → one `request_codex_review` → stopping rule from Phase 0. Update the PR
description likewise (accepted-shrink caveat + #508 reference). Then hand off: lead with the
review branch name (`pr479-eval-timing`, refresh via
`uv run ~/bin/pr_review_worktrees.py --repo Positronic-Robotics/positronic --clone ~/positronic`)
+ PR URL.

## Phase 2 — review-loop infra — VLADIMIR'S VERDICT 2026-07-24: mostly dropped, DO NOT implement here

- **2.1 Repo `AGENTS.md` Code Review Rules — DROPPED.** "codex is generally already following" the
  design-discipline bullets; no server-side rules to add.
- **2.2 Escalation rubric — DONE, relocated OUT of the repo.** Landed as the global cross-project
  rule `~/.claude/rules/design_decision_escalation.md` (promoted to claude-shared), role-neutral
  wording (human/operator/engineer, not "founder"). Nothing to add to `docs/agent_workflow_style.md`
  (that repo file already exists and says operator-workflow prefs belong in the operator's own agent
  config — consistent with this move).
- **2.3 address-review stopping rule — REJECTED as proposed.** "should not be mechanical"; a P1-only
  gate is wrong, and "a fix breaking elsewhere is a real scenario" (fix it, don't decline). Review
  judgment per finding, not a severity gate. Not encoding a mechanical stopping rule.
- Batch-strictly (one push → one `@codex review`) is **already the case** — no change.

## Follow-ups — VLADIMIR: "leave it to future us" — DO NOT do now

1. **#508 — signal defaults** (already filed, Sergey's spec). After it merges, `timing.*` declares
   `default=0.0` and the accepted bounds shrink from this PR disappears (remove the caveat line).
2. **Pre-PR clean pass** in the publish flow — under advisement (Claude tokens vs free Codex
   wall-time); not built now.
3. **Converter no-op whitelist** hardening — future.

## Cleanup (post-implementation)
- [ ] All five Sergey threads + Codex P2 replied/resolved per outcome
- [ ] internal#55 STATUS updated; #508 + follow-up issues cross-linked
- [ ] Delete this file
