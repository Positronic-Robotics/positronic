# Nebius Serverless Workflow

Run Positronic training on [Nebius Serverless Jobs](https://docs.nebius.com/serverless/jobs/manage)
instead of provisioning a GPU VM. Same containers, same training scripts, no VM lifecycle to
manage and no idle compute cost.

This page mirrors **Steps 1 (Convert) and 2 (Train)** of
[docs/training-workflow.md](../../docs/training-workflow.md). Step 3 (inference serving) will
follow in an upcoming change.

## Prerequisites

- Nebius CLI v0.12.209 or newer, authenticated to your project
- Read access to your input dataset bucket and write access to your checkpoint output bucket
- AWS access key, AWS secret key, and a Weights & Biases API key

## One-time setup

Create four MysteryBox secrets that the jobs will reference by name. AWS keys are read from
your local `~/.aws/credentials`; the WandB key from `docker/.env.wandb`. The first three are
single-key payloads consumed via `--env-secret`. The fourth is a two-key payload consumed by
`--volume` for Mountpoint-S3 authentication (Nebius requires the keys to be named
`S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY`).

```bash
PARENT_ID=project-e00f38wexevrr52b8j  # adjust to your own project
AWS_PROFILE_FOR_S3=default            # adjust if your S3 profile isn't `default`

nebius mysterybox secret create \
  --parent-id "$PARENT_ID" \
  --name positronic-serverless-aws-access-key-id \
  --description "AWS access key for serverless training jobs" \
  --secret-version-payload "$(jq -nc \
    --arg v "$(aws configure get aws_access_key_id --profile "$AWS_PROFILE_FOR_S3")" \
    '[{key:"AWS_ACCESS_KEY_ID",string_value:$v}]')"

nebius mysterybox secret create \
  --parent-id "$PARENT_ID" \
  --name positronic-serverless-aws-secret-access-key \
  --description "AWS secret key for serverless training jobs" \
  --secret-version-payload "$(jq -nc \
    --arg v "$(aws configure get aws_secret_access_key --profile "$AWS_PROFILE_FOR_S3")" \
    '[{key:"AWS_SECRET_ACCESS_KEY",string_value:$v}]')"

nebius mysterybox secret create \
  --parent-id "$PARENT_ID" \
  --name positronic-serverless-wandb-api-key \
  --description "WandB API key for serverless training jobs" \
  --secret-version-payload "$(jq -nc \
    --arg v "$(grep -E '^WANDB_API_KEY=' docker/.env.wandb | cut -d= -f2-)" \
    '[{key:"WANDB_API_KEY",string_value:$v}]')"

nebius mysterybox secret create \
  --parent-id "$PARENT_ID" \
  --name positronic-serverless-s3-creds \
  --description "S3 credentials for serverless --volume Mountpoint-S3 mounts" \
  --secret-version-payload "$(jq -nc \
    --arg k "$(aws configure get aws_access_key_id --profile "$AWS_PROFILE_FOR_S3")" \
    --arg s "$(aws configure get aws_secret_access_key --profile "$AWS_PROFILE_FOR_S3")" \
    '[{key:"S3_ACCESS_KEY_ID",string_value:$k},{key:"S3_SECRET_ACCESS_KEY",string_value:$s}]')"
```

The names matter — `train.sh` references the secrets by name. If a secret with one of these
names already exists, the create call fails; skip it.

## Convert a Positronic dataset to LeRobot 0.3.3 format

ACT (and SmolVLA, GR00T, OpenPI) trains on the LeRobot 0.3.3 dataset format. `convert.sh` runs
`python -m positronic.vendors.lerobot_0_3_3.to_lerobot convert` inside a Nebius Job on CPU
(`cpu-e2`, `8vcpu-32gb`). Conversion is video-encoding heavy — CPU is the right resource; a GPU
would be wasted.

Example: convert the public [`sim_stack_cubes`](../../positronic/cfg/ds/phail.py) dataset (317
cube-stacking episodes, hosted on Positronic's public S3 bucket) into a LeRobot dataset on your
own bucket:

```bash
bash workflows/nebius/convert.sh \
  --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
  --output_dir=s3://<your-bucket>/sim_stack_cubes_lerobot/
```

The job reads from `s3://positronic-public/...` anonymously — the dataset's `PUBLIC` profile in
[`positronic/cfg/ds/__init__.py`](../../positronic/cfg/ds/__init__.py) opts into unauthenticated
S3 access, so no extra credentials are needed for the read side. The write side uses the AWS
keys you stored in MysteryBox. Wall-clock for `sim_stack_cubes` is ~25 min (~7 min cold start,
~13 min processing + video encoding, seconds for upload).

The output is a standard LeRobot 0.3.3 layout — `data/`, `videos/`, `meta/{info,episodes,
episodes_stats,tasks}.jsonl`, plus a `run_metadata_*.yaml` capturing the conversion code state.
The output path is what you pass to `train.sh --input_path=...` next.

## Train ACT

`train.sh` runs `python -m positronic.vendors.lerobot_0_3_3.train` inside a Nebius Job on H100
(`gpu-h100-sxm`, `1gpu-16vcpu-200gb`). The bucket from `--input_path=s3://...` is mounted with
[Mountpoint-S3](https://docs.nebius.com/object-storage/interfaces/mountpoint-s3) at `/mnt/input`
(read-only) so the dataset is streamed on demand instead of being downloaded into local cache.
`--output_dir` stays an `s3://` URL handled by `pos3` — LeRobot's checkpoint saver uses
symlinks, which Mountpoint-S3 does not support.

Example: train ACT on the converted `sim_stack_cubes` dataset from the previous step:

```bash
bash workflows/nebius/train.sh \
  --input_path=s3://<your-bucket>/sim_stack_cubes_lerobot/ \
  --exp_name=act_sim_stack_v1 \
  --output_dir=s3://<your-bucket>/checkpoints/lerobot/ \
  --num_train_steps=50000 \
  --save_freq=10000
```

The CLI prints the new job ID and useful follow-up commands:

```
resource_id: aijob-e00...
status: {}

Useful Commands
  • To stream job logs:  nebius ai job logs aijob-e00... --follow
  • To view job details: nebius ai job get aijob-e00...
  ...
```

The job stays in `PROVISIONING`/`STARTING` for ~10 min while the image pulls and the Python
environment resolves inside the container, then runs the actual training. Cost scales with
total wall clock — the cold-start fraction shrinks for longer runs.

## Verifying the run

When the job state reaches `COMPLETED`, the checkpoint structure mirrors a local run:

```bash
aws s3 ls s3://<your-bucket>/checkpoints/lerobot/<exp_name>/ --recursive
```

Expected layout: `checkpoints/<step>/pretrained_model/{config.json,model.safetensors,...}`,
`checkpoints/<step>/training_state/...`, a `run_metadata_*.yaml` capturing the code state, and
an empty `wandb/` placeholder. Live WandB metrics flow to your account directly via the API
key — they aren't synced to S3.

## What changed vs. running on a VM

| Concern | VM flow | Serverless Job |
|---|---|---|
| First-run setup | provision VM, install drivers, configure SSH/network, mount AWS credentials | three `nebius mysterybox secret create` calls |
| Per-run lifecycle | boot VM → SSH → `docker compose run` → stop VM | `bash train.sh ...` |
| Idle cost | accrues until the VM is stopped | none — job releases compute on completion |
| Credentials | live on operator's laptop, mounted into the container | stay in MysteryBox; never leave the cloud |
| Cold-start | seconds (VM kept warm) | ~10 min per job (image pull + venv resolve) |

Trade fast cold-starts for zero idle cost and zero VM administration.

## Configuration

`train.sh` reads two environment variables, both with defaults pointing at the Positronic
project:

| Variable | Default | Purpose |
|---|---|---|
| `NEBIUS_PARENT_ID` | `project-e00f38wexevrr52b8j` | Nebius project to create the job in |
| `NEBIUS_SUBNET_ID` | `vpcsubnet-e00pk1j1x6hjmr4m92` | VPC subnet for the job's compute instance |

Other operational settings (image, GPU platform/preset, MysteryBox secret names, S3 endpoint
URL, region) are hardcoded — change them by editing `train.sh` directly.

## Coming next

- **Inference serving** — currently a long-running `lerobot-0_3_3-server` container on a GPU VM.
  Next phase explores serving via a Nebius Serverless Endpoint.
