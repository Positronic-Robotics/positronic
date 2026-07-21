# Eval Timing

Opt-in wall-clock telemetry for sim evals. A sim eval runs on a **virtual clock**, so the recorded
episode timestamps say nothing about how much real compute a rollout cost. `--timing` captures the
wall-clock split a sizing or perf pass needs — policy inference, env reset/step, record IO, GPU
util/VRAM — that the recorded dataset cannot recover on its own.

It is a **producer → reducer** pipeline: `eval run --timing` writes raw per-rollout timings *during*
the eval; `eval timing-report` reduces them *offline* into a pass-level report. They are separate so
the reducer stores nothing the dataset already holds, and can re-run any time against a finished
dataset without re-running the eval.

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

This writes two files beside the recorded dataset:

- `timing.jsonl` — one JSON line per rollout: the wall costs (reset, env step, policy wait, record IO,
  overhead), the raw per-call inference latencies, and the env server's own physics/render/server split.
- `gpu_dmon.log` — `nvidia-smi dmon` samples for the eval's GPU (skipped when no `nvidia-smi` is present).

`--timing` is sim-only and off by default — with the flag absent every hook is a no-op, so a normal
eval is unaffected.

## Report (reducer)

Point `timing-report` at the dataset dir (a local path, or the same `s3://` URI):

```bash
uv run positronic eval timing-report --dataset_dir=s3://<bucket>/evals/robolab_banana/
```

It joins each timing record to its recorded episode by `episode_uid` — recovering duration, on-disk
size, and the `eval.scored` success verdict from the dataset — and prints a pass report:

- `real_time_factor` — recorded episode seconds per wall second.
- `policy_busy_fraction`, `infer_p50_ms` / `infer_p95_ms` — how much of the pass the policy gated, and
  its per-call latency.
- `wall_split` — reset / env_step / policy_wait / record_io / overhead seconds.
- `env_step_split` — physics / render / wire / server_other inside the env step, when the env reports it.
- `mean_bytes_per_rollout`, `success_rate`, and GPU mean-util / peak-VRAM for the sim box (and the
  policy endpoint, if you pass its `nvidia-smi dmon` log via `--gpu_policy_log`).

The report also lands as `timing_summary.json` next to the input (a sibling `<dataset_dir>.timing_summary.json`
for an `s3://` input).

## See Also

- [Inference Guide](inference.md) — running evals and recording datasets
- `positronic/eval_timing.py` — the collector; `positronic/cli/eval/timing_report.py` — the reducer
