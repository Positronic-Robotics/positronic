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
- Write access to an S3 bucket where converted datasets and checkpoints will land
  (Public datasets like `sim_stack_cubes` are read anonymously — no credentials needed for
  the read side)
- AWS access key + secret for that bucket
- _Optional:_ a Weights & Biases API key for live training metrics. To skip wandb, omit the
  wandb secret below and run training jobs with `WANDB_SECRET= bash workflows/nebius/train.sh ...`.

## One-time setup

Create up to four MysteryBox secrets that the jobs will reference by name. AWS keys are read
from your local `~/.aws/credentials`; the WandB key from `docker/.env.wandb`. The first three
are single-key payloads consumed via `--env-secret`. The fourth is a two-key payload consumed
by `--volume` for Mountpoint-S3 authentication (Nebius requires the keys to be named
`S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY`). The wandb secret is optional — skip it if you
don't use Weights & Biases.

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

Also create one shared filesystem for the dependency caches. Every job and endpoint mounts it
read-write at `/cache`; `uv`, HuggingFace, and openpi asset downloads land there and persist
across cold starts, so only the first run after a dependency change pays the download cost:

```bash
nebius compute filesystem create \
  --parent-id "$PARENT_ID" \
  --name positronic-serverless-cache \
  --type network_ssd \
  --size-gibibytes 512

# --volume needs the filesystem ID, not its name. Grab it and export it:
nebius compute filesystem list --parent-id "$PARENT_ID" --format json \
  | jq -r '.items[] | select(.metadata.name=="positronic-serverless-cache") | .metadata.id'
# → computefilesystem-...   (pass via NEBIUS_CACHE_FS, or rely on the script default)
```

The filesystem is RWX — many jobs/endpoints attach it concurrently. pos3's own cache
(`~/.cache/positronic/s3/`) is deliberately *not* redirected here; it stays on each container's
local disk and re-fetches from S3 by design.

## Convert a Positronic dataset

Each model family expects a specific dataset format. `convert.sh` runs the right converter
with the right [codec](../../docs/codecs.md) for the model you choose, dispatched by the
vendor positional:

| Model | `<vendor>` arg | Converter | Codec namespace |
|---|---|---|---|
| ACT | `lerobot_0_3_3` | `positronic.vendors.lerobot_0_3_3.to_lerobot` | `@positronic.vendors.lerobot_0_3_3.codecs.*` |
| SmolVLA | `lerobot` | `positronic.vendors.lerobot.to_lerobot` | `@positronic.vendors.lerobot.codecs.*` |
| OpenPI | `openpi` | `positronic.vendors.lerobot_0_3_3.to_lerobot` (re-used) | `@positronic.vendors.openpi.codecs.*` |
| GR00T | `gr00t` | `positronic.vendors.lerobot_0_3_3.to_lerobot` (re-used) | `@positronic.vendors.gr00t.codecs.*` |

The job runs on CPU (`cpu-e2`, `8vcpu-32gb`) — conversion is video-encoding heavy; a GPU would
be wasted.

Example: convert the public [`sim_stack_cubes`](../../positronic/cfg/ds/phail.py) dataset (317
cube-stacking episodes, hosted on Positronic's public S3 bucket) into an ACT-ready LeRobot
dataset on your own bucket:

```bash
bash workflows/nebius/convert.sh lerobot_0_3_3 \
  --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
  --output_dir=s3://<your-bucket>/sim_stack_cubes_lerobot/
```

Same shape for the other vendors — swap the vendor token and the codec:

```bash
bash workflows/nebius/convert.sh openpi \
  --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.openpi.codecs.ee \
  --output_dir=s3://<your-bucket>/sim_stack_cubes_openpi/

bash workflows/nebius/convert.sh gr00t \
  --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.gr00t.codecs.ee_rot6d_joints \
  --output_dir=s3://<your-bucket>/sim_stack_cubes_gr00t/
```

`sim_stack_cubes` is publicly hosted on Nebius and read anonymously. The output path is what
you pass to `train.sh --input_path=...` next.

## Train

`train.sh` runs `python -m positronic.vendors.<vendor>.train` inside a Nebius Job on H100
(`gpu-h100-sxm`, `1gpu-16vcpu-200gb`). Supported vendors: `lerobot_0_3_3` (ACT), `lerobot`
(SmolVLA), `openpi`, `gr00t`. The vendor selects the container image and `uv` extras — the rest
of the job spec (preset, secrets, S3 endpoint, mount) is identical.

The bucket from `--input_path=s3://...` is mounted with
[Mountpoint-S3](https://docs.nebius.com/object-storage/interfaces/mountpoint-s3) at `/mnt/input`
(read-only) so the dataset is streamed on demand instead of being downloaded into local cache.
`--output_dir` stays an `s3://` URL handled by [`pos3`](https://github.com/Positronic-Robotics/pos3)
— vendor checkpoint savers tend to use symlinks, which Mountpoint-S3 does not support.

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

The job stays in `PROVISIONING`/`STARTING` while the image pulls and the Python environment
resolves inside the container, then runs the actual training. The first job after a dependency
change pays the full `uv`/HF download cost (~10 min); subsequent jobs reuse the shared `/cache`
filesystem and start substantially faster. Cost scales with total wall clock — the cold-start
fraction shrinks for longer runs.

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

bash workflows/nebius/serve.sh openpi my-openpi serve \
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

No VM to provision, SSH into, or remember to shut down. Credentials stay in MysteryBox instead
of on operator laptops. Compute is released the moment a job finishes — idle cost goes to zero.
The trade-off is a cold start per job (image pull + venv resolve) instead of a warm VM —
mitigated by the shared `/cache` filesystem, so only the first run after a dependency change
pays the full download.

## Configuration

The script defaults point at Positronic Robotics' own Nebius project — **external users must
override them** with their own project + subnet IDs:

| Variable | Default (Positronic-internal) | Purpose |
|---|---|---|
| `NEBIUS_PARENT_ID` | `project-e00f38wexevrr52b8j` | Nebius project to create the job/endpoint in |
| `NEBIUS_SUBNET_ID` | `vpcsubnet-e00pk1j1x6hjmr4m92` | VPC subnet for the compute instance |
| `WANDB_SECRET` | `positronic-serverless-wandb-api-key` | MysteryBox secret name for the WandB key. Set empty (`WANDB_SECRET=`) to skip wandb entirely. |
| `NEBIUS_CACHE_FS` | `computefilesystem-e00f6jyfr5wkawyrab` | Shared filesystem **ID** (not name — `--volume` rejects names) mounted RW at `/cache` for the `uv`/HF/openpi caches (`UV_CACHE_DIR`, `HF_HOME`, `OPENPI_DATA_HOME`). Not used by pos3. The default is Positronic-internal; external users must override with their own filesystem ID. |

Other operational settings (platform/preset, MysteryBox secret names, S3 endpoint URL, region)
are hardcoded — change them by editing the script directly. The vendor positional arg selects
the container image and `uv` extras:

| Vendor | Image | `uv` extra |
|---|---|---|
| `lerobot_0_3_3` (ACT) | `positro/positronic` | `--extra lerobot_0_3_3` |
| `lerobot` (SmolVLA) | `positro/positronic` | `--extra lerobot` |
| `openpi` | `positro/openpi` | `--extra openpi` (serve); none for train/stats |
| `gr00t` | `positro/gr00t` | _(none — `/gr00t` is co-installed)_ |
