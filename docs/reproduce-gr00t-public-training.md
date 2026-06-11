# Reproducing GR00T training on Positronic's public datasets

A record of reproducing GR00T (`nvidia/GR00T-N1.6-3B`) fine-tuning on two of
Positronic's **public** datasets — `sim_stack_cubes` and the PhAIL
`teleop_unified` teleoperation set — with **relative end-effector actions** and a
**6D (rot6d) rotation representation**, on the Nebius Serverless pipeline.

Run date: 2026-06-03.

## Datasets (public)

| Dataset | Config target | Public S3 source |
|---|---|---|
| sim_stack | `@positronic.cfg.ds.sim.sim_stack_cubes` | `s3://positronic-public/datasets/sim-stack-cubes/` |
| PhAIL teleop (unified task label) | `@positronic.cfg.phail.v1_0.teleop_unified` | `s3://positronic-public/phail/v1.0/dataset/teleoperation/` |

## Encoding

- Convert codec `ee_rot6d`: end-effector pose with a 6D (rot6d) rotation
  representation.
- Relative vs. absolute actions is a training-time choice via `--modality_config`.
  We use `ee_rot6d_rel` (relative) for both datasets.

## Step 1 — Convert (CPU Jobs)

Submitted from the repo root via the `remote-training` wrapper:

```bash
bash workflows/nebius/convert.sh gr00t \
  --dataset.dataset=@positronic.cfg.ds.sim.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.gr00t.codecs.ee_rot6d \
  --output_dir=s3://interim/testing/sim_stack/groot/ee_rot6d/

bash workflows/nebius/convert.sh gr00t \
  --dataset.dataset=@positronic.cfg.phail.v1_0.teleop_unified \
  --dataset.codec=@positronic.vendors.gr00t.codecs.ee_rot6d \
  --output_dir=s3://interim/testing/phail_unified/groot/ee_rot6d/
```

| Dataset | Nebius Job | Output |
|---|---|---|
| sim_stack | `aijob-e00f7m7p1qt2w84dts` | `s3://interim/testing/sim_stack/groot/ee_rot6d/` |
| PhAIL | `aijob-e00nb2tqvx5p8kqv3m` | `s3://interim/testing/phail_unified/groot/ee_rot6d/` |

> PhAIL convert note: the first attempt (`aijob-e00j0bh4ev601bd3s7`) hit the
> wrapper's default 4h job timeout at 81% (449 episodes, ~2 cameras, 1485–2070
> frames each, ~42 s/episode sequential → ~5.4h total). `convert.sh` was extended
> to expose `NEBIUS_CPU_PRESET` and `NEBIUS_JOB_TIMEOUT`; the re-run used
> `NEBIUS_CPU_PRESET=16vcpu-64gb NEBIUS_JOB_TIMEOUT=8h` and completed in ~5h48m.
> Note: the larger preset gave **no speedup** (conversion is sequential per-episode
> + S3-IO bound, not CPU-core bound) — the **timeout bump** was the actual fix.
> For future large converts, raise `NEBIUS_JOB_TIMEOUT` and keep the default
> `8vcpu-32gb`. (sim_stack, smaller, converted fine on `8vcpu-32gb` / 4h.)

## Step 2 — Train (H100 Jobs)

Relative actions (`ee_rot6d_rel`), rot6d. Checkpoint lands at
`<output_path>/<exp_name>/`.

```bash
# sim_stack — 20k steps
bash workflows/nebius/train.sh gr00t \
  --input_path=s3://interim/testing/sim_stack/groot/ee_rot6d/ \
  --output_path=s3://checkpoints/testing/sim_stack/groot/ \
  --exp_name=ee_rot6d_rel \
  --modality_config=ee_rot6d_rel \
  --num_train_steps=20000 --save_steps=2000 --num_workers=4
```

| Dataset | Nebius Job | Steps | Status | Checkpoint |
|---|---|---|---|---|
| sim_stack | `aijob-e00ct19xsfxxxdxrpq` | 20 000 | COMPLETED | `s3://checkpoints/testing/sim_stack/groot/ee_rot6d_rel/` (`checkpoint-2000` … `checkpoint-20000`) |
| PhAIL (run 1) | `aijob-e00zv2yss8b7ercp3t` | →90 000 | STOPPED (disk full) | reached `checkpoint-90000` |
| PhAIL (resume) | `aijob-e00a3p4z6a7yg3enje` | 90 000→150 000 | RUNNING | `s3://checkpoints/testing/phail_unified/groot/ee_rot6d_rel/` |

> PhAIL train note: run 1 stopped at **step 90000** (reported COMPLETED, not a
> timeout — it was only ~16h in, before the 24h cap). Cause: each GR00T checkpoint
> is **~22.9 GB** (incl. a 12.96 GB `optimizer.pt`) and the wrapper forces
> `--save_total_limit 9999` (keep all), so 9 checkpoints ≈ 206 GB exhausted the
> **default 250 GiB** disk; the write of `checkpoint-100000` failed (the trailing
> wandb `socket.send()` errors are the disk-full symptom).
>
> Fix: `train.sh` now takes `--disk-size` (env `NEBIUS_DISK_SIZE`, **default 750Gi**).
> Resumed from `checkpoint-90000` (valid: `optimizer.pt` present, `global_step=90000`)
> with a less-frequent save cadence (`--save_steps=30000` → saves at 120k, 150k).
> At ~1h45m/10k, 90k→150k is ~10.5h (well under the 24h cap); 750Gi holds the
> re-synced checkpoints + the rest.
>
> ```bash
> bash workflows/nebius/train.sh gr00t \
>   --input_path=s3://interim/testing/phail_unified/groot/ee_rot6d/ \
>   --output_path=s3://checkpoints/testing/phail_unified/groot/ \
>   --exp_name=ee_rot6d_rel --modality_config=ee_rot6d_rel \
>   --num_train_steps=150000 --save_steps=30000 --resume=true   # NEBIUS_DISK_SIZE defaults to 750Gi
> ```
