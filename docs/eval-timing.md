# Eval Timing

Opt-in wall-clock telemetry for sim evals. A sim eval runs on a **virtual clock**, so the recorded
episode timestamps say nothing about how much real compute a rollout cost. `--timing` captures the
wall-clock split a sizing or perf pass needs — policy inference, env reset/step, record IO, GPU
util/VRAM — that the recorded dataset cannot recover on its own.

It is a **producer → reducer** pipeline: `eval run --timing` records raw per-rollout timings into the
dataset itself *during* the eval; `eval timing-report` reduces them *offline* into a pass-level report.
They are separate so the reducer stores nothing beyond the recorded raw data, and can re-run any time
against a finished dataset without re-running the eval.

## Collect (producer)

Add `--timing` to any sim `eval run`:

```bash
uv run positronic eval run \
  --eval=@positronic.cfg.eval.sim.robolab.banana_in_bowl \
  --eval.trial_count=10 \
  --policy=@positronic.cfg.policy.remote --policy.host=<endpoint-ip> \
  --output_dir=s3://<bucket>/evals/robolab_banana/ \
  --timing
```

This records the telemetry into each episode of the dataset — no side files:

- `timing.*` signals — the per-tick wall costs (env step, record IO, and the env server's own
  disjoint step decomposition as `timing.env_*` phases) plus one `timing.infer_ms` sample per policy
  round-trip.
- `timing.gpu_*` signals — the eval box's GPU, sampled ~1 Hz by a foreground control system: `timing.gpu_util`
  (whole-box utilisation %), `timing.gpu_mem` (whole-box memory used, MiB), and `timing.gpu_mem_proc` (the
  memory attributed to **this eval's process tree** — the harness and its env-server/Isaac children, MiB).
  Absent on a box without `nvidia-smi`.
- `timing.wall_s` / `timing.finished_at` / `timing.reset_s` statics — the once-per-episode wall scalars.

**Concurrent-sim footgun:** `timing.gpu_util` and `timing.gpu_mem` are **whole-box** — they measure the whole
GPU, not this eval. With several sims co-located on one GPU, do **not** sum or average utilisation across
concurrent episodes: they all report the same shared-box number. Memory *is* attributable per eval
(`timing.gpu_mem_proc`); utilisation is not (per-process util is unreliable under MPS / co-location, so it is
deliberately not recorded).

`--timing` is sim-only and off by default — with the flag absent every hook is a no-op, so a normal
eval is unaffected.

> **Caveat (temporary):** a `timing.*` stream's first sample lands a beat after the episode opens (up to one
> ~1 Hz GPU sampling interval), so episode bounds shrink by that much — a visualisation/replay concern only.
> [#508](https://github.com/Positronic-Robotics/positronic/issues/508) removes the distortion: the timing
> signals declare `default=0.0` and bounds stop intersecting them.

## Report (reducer)

Point `timing-report` at the dataset dir (a local path, or the same `s3://` URI):

```bash
uv run positronic eval timing-report --dataset_dir=s3://<bucket>/evals/robolab_banana/
```

It reduces each episode's `timing.*` signals and statics — joined with the duration, on-disk size, and
the `eval.scored` success verdict the episode already records — and prints a pass report:

- `real_time_factor` — recorded episode seconds per wall second.
- `policy_busy_fraction`, `infer_p50_ms` / `infer_p95_ms` — how much of the pass the policy gated, and
  its per-call latency.
- `wall_split` — the fraction of W_pass in each phase: reset / env_step / policy_wait / record_io / overhead
  / between_episodes (episode writer close — parquet/video flush — plus teardown, homing, world rebuild);
  sums to 1.
- `env_step_split` — the env server's own reported phases (for robolab: physics / render / server_other)
  plus wire / materialize inside the env step, when the env reports a decomposition. The phase set is the
  env's own; the reducer sums whatever `timing.env_*` columns it recorded.
- `mean_bytes_per_rollout`, `success_rate`, and GPU mean-util / peak-VRAM for the sim box — reduced from the
  recorded `timing.gpu_*` signals (with this eval's peak process-tree VRAM alongside the whole-box peak). The
  policy endpoint (a different box) folds in the same numbers from an optional `--gpu_policy_log`; collect
  that log with `nvidia-smi dmon -s um`, which emits the `fb` framebuffer column the reducer needs (a plain
  `dmon` log, lacking `fb`, is rejected).

The report also lands as `timing_summary.json` next to the input (a sibling `<dataset_dir>.timing_summary.json`
for an `s3://` input).

## See Also

- [Inference Guide](inference.md) — running evals and recording datasets
- `positronic/eval_timing.py` — the collector; `positronic/cli/eval/timing_report.py` — the reducer
