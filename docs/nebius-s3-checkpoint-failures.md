# Nebius S3 connectivity failures during training

A record of an intermittent Nebius object-storage problem hit during the DreamZero wan2.2 5B
full-finetune on Nebius (June 2026), what it broke, how to recognize it, and how to reduce its blast
radius. The storage endpoint involved is `storage.eu-north1.nebius.cloud`.

## Symptom

During training, the checkpoint pipeline (HuggingFace `Trainer` save + the positronic background sync
that mirrors the output dir to `s3://checkpoints/...`) intermittently fails to reach S3:

```
Background sync iteration failed: Could not connect to the endpoint URL:
"https://storage.eu-north1.nebius.cloud/..."
```

Upload throughput collapses to <1 MB/s for a 14 GB checkpoint, or the connection drops outright. It is
transient and node-local: the same bucket reads/writes fine from a laptop and from other VMs at the same
time, and the endpoint recovers on its own after minutes to tens of minutes. It is **not** a disk,
CUDA, or OOM problem — none of those signatures appeared in the logs.

## Two distinct ways it bites

1. **Corrupt (partial) checkpoint upload.** The sync uploads `config.json` and other small files but
   drops out before the multi-GB `model-*.safetensors` shards land. The result on S3 is a checkpoint
   directory that *looks* present (`config.json`, `experiment_cfg`) but has **zero or partial model
   shards**. Serving such a checkpoint silently loads the base / untrained weights and produces garbage
   — e.g. the arm flailing / falling to the table that looked like a model failure but was a bad
   upload. In this incident `checkpoint-9000` shipped with no shards on the first attempt and
   `checkpoint-10000` landed only 3 of 6 shards.

2. **Training crash / wedge.** The HF checkpoint save is synchronous, so when the upload hangs the whole
   training loop stalls (no `training_step` for 30+ min). If the sync raises instead of hanging, the
   subprocess exits non-zero and `subprocess.run(command, check=True)` in `train.py` raises
   `CalledProcessError`, killing the job. Either way training does not advance past the failing save.

## Why it's expensive here

The DreamZero run used `save_only_model=true` (full FT, to keep checkpoints small enough to upload).
That means **no optimizer / trainer state is saved**, so a killed job **cannot resume cleanly** — a
restart is from scratch and loses all completed steps. Each checkpoint is ~14 GB, so every save is a
large, failure-prone upload over the flaky endpoint.

## How to detect it

- Grep the job log for `Could not connect to the endpoint URL` and `Background sync iteration failed`.
- After a run, verify each checkpoint on S3 has the **full shard set** before trusting it:
  `aws s3 ls .../checkpoint-N/` and confirm the `model-*.safetensors` count matches `model.safetensors.index.json`.
  A checkpoint with `config.json` but missing/short shards is corrupt — never serve it.
- A wedged job shows no new `training_step` lines and a stalled checkpoint upload at <1 MB/s.

## Mitigations

- **Always validate a checkpoint before serving.** Treat presence of `config.json` as meaningless;
  trust only a complete shard set. Keep the last *known-complete* checkpoint as the fallback model.
- **Minimize upload events.** The crash is upload-triggered, so fewer saves → fewer chances to hit a
  blip and a higher probability of reaching the target step. Raising `save_steps` (e.g. one save every
  5000 steps instead of every 1000) trades checkpoint granularity for completion odds; with a per-save
  failure probability `p`, completion probability is `(1 - p)^(number of saves)`.
- **Restart on a fresh VM.** The failures were node-local; a new job lands on a different node, which
  often has a healthy path to the endpoint.
- **Re-upload recovers when the endpoint heals.** A corrupt checkpoint that is still complete on the
  training node's local disk gets re-synced once connectivity returns (this is how `checkpoint-9000`
  later became valid). If the local copy is gone, the only fix is a re-run.

## Takeaway

The single most important habit: **a checkpoint on S3 is not trustworthy until its shard set is
complete.** Most of the "the model is broken" confusion in this incident traced back to serving a
checkpoint whose weights never finished uploading.
