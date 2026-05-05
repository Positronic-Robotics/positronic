# Nebius Serverless Workflow

Run the full Positronic training and inference workflow on
[Nebius Serverless](https://docs.nebius.com/serverless) — Jobs for batch work (data conversion,
training) and Endpoints for HTTP inference servers. Same containers, same scripts, no VM
lifecycle to manage and no idle compute cost.

This page mirrors all three cloud-side steps of
[docs/training-workflow.md](../../docs/training-workflow.md): Convert, Train, Serve. Step 4
(running inference from your robot or simulator against the served policy) is unchanged — see
[docs/inference.md](../../docs/inference.md).

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

The names matter — `convert.sh`, `train.sh`, and `serve.sh` reference the secrets by name. If a
secret with one of these names already exists, the create call fails; skip it.

## Convert a Positronic dataset to LeRobot 0.3.3 format

ACT (and SmolVLA, GR00T, OpenPI) trains on the LeRobot 0.3.3 dataset format. `convert.sh` runs
`python -m positronic.vendors.<vendor>.to_lerobot convert` inside a Nebius Job on CPU
(`cpu-e2`, `8vcpu-32gb`). Conversion is video-encoding heavy — CPU is the right resource; a GPU
would be wasted. Supported vendors: `lerobot_0_3_3` (used by ACT) and `lerobot` (used by SmolVLA
and other lerobot 0.4.x policies). OpenPI and GR00T have no converter of their own — they read
the `lerobot_0_3_3` output directly.

Example: convert the public [`sim_stack_cubes`](../../positronic/cfg/ds/phail.py) dataset (317
cube-stacking episodes, hosted on Positronic's public S3 bucket) into a LeRobot dataset on your
own bucket:

```bash
bash workflows/nebius/convert.sh lerobot_0_3_3 \
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

## Train

`train.sh` runs `python -m positronic.vendors.<vendor>.train` inside a Nebius Job on H100
(`gpu-h100-sxm`, `1gpu-16vcpu-200gb`). Supported vendors: `lerobot_0_3_3` (ACT), `lerobot`
(SmolVLA), `openpi`, `gr00t`. The vendor selects the container image and `uv` extras — the rest
of the job spec (preset, secrets, S3 endpoint, mount) is identical.

The bucket from `--input_path=s3://...` is mounted with
[Mountpoint-S3](https://docs.nebius.com/object-storage/interfaces/mountpoint-s3) at `/mnt/input`
(read-only) so the dataset is streamed on demand instead of being downloaded into local cache.
`--output_dir` stays an `s3://` URL handled by `pos3` — vendor checkpoint savers tend to use
symlinks, which Mountpoint-S3 does not support.

Example: train ACT on the converted `sim_stack_cubes` dataset from the previous step:

```bash
bash workflows/nebius/train.sh lerobot_0_3_3 \
  --input_path=s3://<your-bucket>/sim_stack_cubes_lerobot/ \
  --exp_name=act_sim_stack_v1 \
  --output_dir=s3://<your-bucket>/checkpoints/lerobot/ \
  --num_train_steps=50000 \
  --save_freq=10000
```

Swap `lerobot_0_3_3` for `lerobot`, `openpi`, or `gr00t` to train other policies on the same
dataset; remaining flags forward to that vendor's `train` CLI.

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

Expected ACT layout: `checkpoints/<step>/pretrained_model/{config.json,model.safetensors,...}`,
`checkpoints/<step>/training_state/...`, a `run_metadata_*.yaml` capturing the code state, and
an empty `wandb/` placeholder. SmolVLA matches the same layout; OpenPI and GR00T use their own
checkpoint shapes (see each vendor's README under `positronic/vendors/`). Live WandB metrics
flow to your account directly via the API key — they aren't synced to S3.

## Serve a checkpoint as an HTTP endpoint

`serve.sh` creates a [Nebius Serverless Endpoint](https://docs.nebius.com/serverless/endpoints/manage)
running `python -m positronic.vendors.<vendor>.server` on H100, with a public static IP on
port 8000. Endpoints don't have managed DNS yet, so the IP is the contact address — it's stable
across endpoint stop/start, but new endpoints get new IPs. Supported vendors:
`lerobot_0_3_3`, `lerobot`, `openpi`, `gr00t`.

Take a vendor and a unique endpoint name as the first two arguments; remaining arguments forward
to the server CLI. Example using the public ACT demo checkpoint at
`s3://positronic-public/checkpoints/sim_stack_cubes/act/` (no S3 credentials needed inside the
container — the `demo` subcommand is `lerobot_0_3_3`-only and reads anonymously):

```bash
bash workflows/nebius/serve.sh lerobot_0_3_3 my-act-demo demo
```

Or against your own trained checkpoint:

```bash
bash workflows/nebius/serve.sh lerobot_0_3_3 act-server serve \
  --checkpoints_dir=s3://<your-bucket>/checkpoints/lerobot/<exp_name>/
```

Same shape for the other vendors — replace the vendor token and point `--checkpoints_dir` at the
matching checkpoint:

```bash
bash workflows/nebius/serve.sh lerobot smolvla-server serve \
  --checkpoints_dir=s3://<your-bucket>/checkpoints/smolvla/<exp_name>/

bash workflows/nebius/serve.sh openpi pi-server serve \
  --checkpoints_dir=s3://<your-bucket>/checkpoints/openpi/<exp_name>/

bash workflows/nebius/serve.sh gr00t groot-server ee_rot6d_rel \
  --checkpoints_dir=s3://<your-bucket>/checkpoints/groot/<exp_name>/
```

`serve.sh` blocks until the public IP is allocated (typically <1 min), then prints a banner with
the URL, endpoint ID, and the commands to follow logs and tear down. The container takes another
~10–15 min to finish `uv sync` and load the model into GPU memory; once `INFO Started server
process` appears in `nebius ai endpoint logs`, sanity-check with:

```bash
curl http://<endpoint-ip>:8000/api/v1/models
# → {"models": ["050000"]}
```

Run inference from your laptop or robot host using the existing `positronic-inference` CLI
([docs/inference.md](../../docs/inference.md)):

```bash
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=<endpoint-ip> \
  --policy.port=8000 \
  --output_dir=.data/inference/<run-name>/
```

When you're done, `stop.sh` deletes the endpoint and releases the public IP:

```bash
bash workflows/nebius/stop.sh my-act-demo
```

To pause an endpoint without releasing its static IP (useful if you want to reuse the same IP
later), use `nebius ai endpoint stop <id>` directly — `start` resumes it.

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

All scripts read two environment variables, both with defaults pointing at the Positronic
project:

| Variable | Default | Purpose |
|---|---|---|
| `NEBIUS_PARENT_ID` | `project-e00f38wexevrr52b8j` | Nebius project to create the job/endpoint in |
| `NEBIUS_SUBNET_ID` | `vpcsubnet-e00pk1j1x6hjmr4m92` | VPC subnet for the compute instance |

Other operational settings (platform/preset, MysteryBox secret names, S3 endpoint URL, region)
are hardcoded — change them by editing the script directly. The vendor positional arg selects
the container image and `uv` extras:

| Vendor | Image | `uv` extra |
|---|---|---|
| `lerobot_0_3_3` (ACT) | `positro/positronic` | `--extra lerobot_0_3_3` |
| `lerobot` (SmolVLA) | `positro/positronic` | `--extra lerobot` |
| `openpi` | `positro/openpi` | _(none — `/openpi` is co-installed)_ |
| `gr00t` | `positro/gr00t` | _(none — `/gr00t` is co-installed)_ |
