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

Add `--timing` to any sim `eval run`; it needs an `--output_dir` to write into:

```bash
uv run positronic eval run \
  --eval=@positronic.cfg.eval.sim.robolab.banana_in_bowl \
  --eval.trial_count=10 \
  --policy=@positronic.cfg.policy.remote --policy.host=<endpoint-ip> \
  --output_dir=s3://<bucket>/evals/robolab_banana/ \
  --timing
```

This records the telemetry into each episode of the dataset:

- `timing.*` signals — the per-tick wall costs (env step, record IO, and the env server's own
  physics/render/server split) plus one `timing.infer_ms` sample per policy round-trip.
- `timing.wall_s` / `timing.finished_at` / `timing.reset_s` statics — the once-per-episode wall scalars.

One side file lands beside the dataset — `gpu_dmon.log`, `nvidia-smi dmon` samples for the eval's GPU
(skipped when no `nvidia-smi` is present): a background per-box time series that outlives any single
episode, so the dataset cannot carry it.

`--timing` is sim-only and off by default — with the flag absent every hook is a no-op, so a normal
eval is unaffected.

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
  / between_episodes (inter-episode teardown, homing, world rebuild); sums to 1.
- `env_step_split` — physics / render / server_other / wire / materialize inside the env step, when the env
  reports its decomposition.
- `mean_bytes_per_rollout`, `success_rate`, and GPU mean-util / peak-VRAM for the sim box (and the
  policy endpoint, if you pass its `nvidia-smi dmon` log via `--gpu_policy_log`).

The report also lands as `timing_summary.json` next to the input (a sibling `<dataset_dir>.timing_summary.json`
for an `s3://` input).

## See Also

- [Inference Guide](inference.md) — running evals and recording datasets
- `positronic/eval_timing.py` — the collector; `positronic/cli/eval/timing_report.py` — the reducer
