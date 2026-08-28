# Nebius Serverless Workflow

Run the full Positronic training and inference workflow on
[Nebius Serverless](https://docs.nebius.com/serverless) — Jobs for batch work (data conversion,
training) and Endpoints for HTTP inference servers. Same containers, same scripts, no VM
lifecycle to manage and no idle compute cost.

This page mirrors all three cloud-side steps of
[docs/training-workflow.md](../../docs/training-workflow.md): Convert, Train, Serve. Step 4 —
running inference from your robot or simulator against the served policy — works as
[docs/inference.md](../../docs/inference.md) describes, except that an endpoint served here is
reached at a managed `https://` URL and gated on a bearer token: see
[Authenticated inference](#authenticated-inference).

## Prerequisites

- Nebius CLI v0.12.209 or newer, authenticated to your project
- Write access to an S3 bucket where converted datasets and checkpoints will land
  (Public datasets like `sim_stack_cubes` are read anonymously — no credentials needed for
  the read side)
- AWS access key + secret for that bucket
- _Optional:_ a Weights & Biases API key for live training metrics. To skip wandb, omit the
  wandb secret below and run training jobs with `WANDB_SECRET= bash workflows/nebius/train.sh ...`.

## One-time setup

Create up to five MysteryBox secrets that the jobs will reference by name. AWS keys are read
from your local `~/.aws/credentials`; the WandB key from `docker/.env.wandb`. The first three
are single-key payloads consumed via `--env-secret`. The fourth is a two-key payload consumed
by `--volume` for Mountpoint-S3 authentication (Nebius requires the keys to be named
`S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY`). The fifth is the bearer token every served
endpoint is gated on — `serve.sh` injects it as the container's `AUTH_TOKEN`, and the payload
key must be `AUTH_TOKEN` too. The wandb secret is optional — skip it if you don't use
Weights & Biases.

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

nebius mysterybox secret create \
  --parent-id "$PARENT_ID" \
  --name positronic-serverless-inference-token \
  --description "Bearer token gating served inference endpoints" \
  --secret-version-payload "$(jq -nc \
    --arg v "$(openssl rand -hex 32)" '[{key:"AUTH_TOKEN",string_value:$v}]')"
```

The names matter — `convert.sh`, `train.sh`, `serve.sh`, and `eval.sh` reference the secrets by
name. If a secret with one of these names already exists, the create call fails; skip it.

## Shared cache filesystem

Create one shared filesystem for the dependency caches. Every job and endpoint mounts it
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

To inspect or wipe this filesystem later, see
[Appendix: Cleaning the shared cache](#appendix-cleaning-the-shared-cache).

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

Example: convert the public [`sim_stack_cubes`](../../positronic/cfg/ds/sim.py) dataset (317
cube-stacking episodes, hosted on Positronic's public S3 bucket) into an ACT-ready LeRobot
dataset on your own bucket:

```bash
bash workflows/nebius/convert.sh lerobot_0_3_3 \
  --dataset.dataset=@positronic.cfg.ds.sim.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
  --output_dir=s3://<your-bucket>/sim_stack_cubes_lerobot/
```

Same shape for the other vendors — swap the vendor token and the codec:

```bash
bash workflows/nebius/convert.sh openpi \
  --dataset.dataset=@positronic.cfg.ds.sim.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.openpi.codecs.ee \
  --output_dir=s3://<your-bucket>/sim_stack_cubes_openpi/

bash workflows/nebius/convert.sh gr00t \
  --dataset.dataset=@positronic.cfg.ds.sim.sim_stack_cubes \
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
running `python -m positronic.vendors.<vendor>.server` on H100. The endpoint gets no public IP:
Nebius fronts the container's port 8000 with a managed `https://` URL, which terminates TLS and
is the contact address. That URL survives endpoint stop/start; deleting an endpoint retires it,
so a re-created one of the same name gets a new URL. Supported vendors: `lerobot_0_3_3`,
`lerobot`, `openpi`, `gr00t`.

Every endpoint is gated on a bearer token — see [Authenticated inference](#authenticated-inference)
below for loading it and for why the check lives in the server rather than at the Nebius ingress.

Take a vendor and a unique endpoint name as the first two arguments; remaining arguments forward
to the server CLI. Example using the public ACT demo checkpoint at
`s3://positronic-public/checkpoints/sim_stack_cubes/act/` (no S3 credentials needed inside the
container — the `demo` subcommand is `lerobot_0_3_3`-only and reads anonymously):

```bash
bash workflows/nebius/serve.sh lerobot_0_3_3 my-act-demo demo
```

Or against your own trained checkpoint:

```bash
bash workflows/nebius/serve.sh lerobot_0_3_3 act-server ee \
  --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/lerobot/<exp_name>/
```

Same shape for the other vendors — replace the vendor token and point `--pipeline.source.checkpoints_dir` at the
matching checkpoint:

```bash
bash workflows/nebius/serve.sh lerobot smolvla-server ee \
  --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/smolvla/<exp_name>/

# --pipeline.ee_frame states the EE frame the checkpoint speaks; None means the rig's `default`
bash workflows/nebius/serve.sh openpi my-openpi ee \
  --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/openpi/<exp_name>/ \
  --pipeline.ee_frame=None

bash workflows/nebius/serve.sh gr00t groot-server ee_rot6d_rel \
  --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/groot/<exp_name>/
```

`serve.sh` blocks until the managed URL appears (typically <1 min), then prints a banner with
that URL, the endpoint ID, and the commands to follow logs and tear down. The container takes
another ~10–15 min to finish `uv sync` and load the model into GPU memory; once `INFO Started
server process` appears in `nebius ai endpoint logs`, sanity-check with (`AUTH_TOKEN` loaded as
in [Authenticated inference](#authenticated-inference)):

```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" https://<endpoint-managed-url>/api/v1/models
# → {"models": ["050000"]}
```

Run inference from your laptop or robot host with `positronic eval run`
([docs/inference.md](../../docs/inference.md)); `.authed_remote` attaches the token:

```bash
uv run positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=.authed_remote \
  --policy.url=https://<endpoint-managed-url> \
  --output_dir=.data/inference/<run-name>/
```

When you're done, `stop.sh` deletes the endpoint:

```bash
bash workflows/nebius/stop.sh my-act-demo
```

Deleting retires the managed URL, and a re-created endpoint of the same name gets a new one — so anything
holding it, a robot config or an eval job, breaks on redeploy. To keep the URL, use `nebius ai endpoint
stop <id>` instead: it releases the compute too, and `start` resumes on the same URL.

### The managed URL is assigned, not chosen

It belongs to [a tunnel](https://docs.nebius.com/tunnels/overview) Nebius creates with the endpoint —
`https://port8000-<tunnel-id>.tunnel.applications.<region>.nebius.cloud`. No flag sets it and nothing
derives it, which is why `serve.sh` polls `status.public_endpoints` to learn it.

A URL that outlives the endpoint needs a tunnel of your own (`nebius tunnel create`) with its agent in the
container, which also names the host (`services.name`, up to 20 lowercase alphanumerics — `phail` rather
than `port8000`). Nebius offers no custom domain or uploaded certificate on either path.

## Authenticated inference

The server validates `Authorization: Bearer <token>` on `/api/v1/models` and on the inference
WebSocket, rejecting before the session opens. `serve.sh` injects the token from the
`positronic-serverless-inference-token` secret as the container's `AUTH_TOKEN`; export the same
value locally and `.authed_remote` sends it (it raises immediately if the variable is unset).

```bash
source workflows/nebius/common.sh   # the secret serve.sh injected, whatever NEBIUS_AUTH_TOKEN_SECRET selects
SECRET_ID=$(nebius mysterybox secret get-by-name --parent-id "$PARENT_ID" \
  --name "$AUTH_TOKEN_SECRET" --format json | jq -r '.metadata.id')
export AUTH_TOKEN=$(nebius mysterybox payload get-by-key \
  --secret-id "$SECRET_ID" --key "$AUTH_TOKEN_KEY" --format json | jq -r '.data.string_value')

uv run positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=.authed_remote \
  --policy.url=https://<endpoint-managed-url> \
  --output_dir=.data/inference/<run-name>/
```

Nebius' own `--auth token` would spare the in-server check, but its auth proxy does not preserve the
WebSocket upgrade upstream: the session route answers `200` instead of upgrading, with a valid token as
much as without. `--auth none` bypasses that proxy, which is why the same request upgrades there. Nebius
confirmed it on ticket U22281505; gRPC shares the proxy, so it is no escape hatch.

Two behaviours here are observed, not promised: Nebius documents no WebSocket or connection-lifetime
contract at all, and the ingress closes a connection it has read nothing from after ~90 s — shorter than a
cold checkpoint's first inference, so the client holds sessions open with pings (`ping_interval` in
`positronic/offboard/client.py`). `pytest -m endpoint` is what catches either changing.

### Letting the config read the secret

`.nebius_remote` fetches the token from MysteryBox itself, so nothing is exported and a
[rotated](#rotating-the-token) token is picked up on the next run:

```bash
uv run positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=.nebius_remote \
  --policy.url=https://<endpoint-managed-url> \
  --output_dir=.data/inference/<run-name>/
```

It needs a logged-in `nebius`, so it is an operator's convenience: a robot host and the eval job use
`.authed_remote` against `AUTH_TOKEN` and need no cloud credentials. Lookup in `positronic/utils/nebius.py`.

### Checking a deployment

`pytest -m endpoint` states what a served endpoint must do: both routes refuse a missing, wrong, or
`Bearer`-less token and serve with the right one, and a session survives an idle past the ~90 s close.
Unset, the tests serve their own server and prove the code; pointed at `POSITRONIC_ENDPOINT_URL` they run
the same assertions through the managed ingress, where `--auth token` and the idle close are invisible from
inside the container. `e2e.sh` runs them after its serve stage.

Against an endpoint of your own — the demo checkpoint needs no training and no S3 credentials:

```bash
bash workflows/nebius/serve.sh lerobot_0_3_3 auth-smoke demo
# wait for `INFO Started server process` in: nebius ai endpoint logs <id> --follow
# AUTH_TOKEN as exported above; the tests read it and fail on a KeyError without it
POSITRONIC_ENDPOINT_URL=https://<endpoint-managed-url> \
  uv run --locked pytest positronic/offboard/tests/test_server.py -m endpoint --no-cov
bash workflows/nebius/stop.sh auth-smoke
```

### Rotating the token

Add a new primary version:

```bash
source workflows/nebius/common.sh
SECRET_ID=$(nebius mysterybox secret get-by-name --parent-id "$PARENT_ID" \
  --name "$AUTH_TOKEN_SECRET" --format json | jq -r '.metadata.id')

nebius mysterybox secret-version create --parent-id "$SECRET_ID" --set-primary \
  --payload "$(jq -nc --arg k "$AUTH_TOKEN_KEY" --arg v "$(openssl rand -hex 32)" \
    '[{key:$k,string_value:$v}]')"
```

`.nebius_remote` picks the new value up next run; an exported `AUTH_TOKEN` holds the old one until
re-exported. Either way a running endpoint keeps the token its container read at start, so the old one
opens sessions until the endpoint is replaced — which matters when the rotation answers a disclosure.
`serve.sh` only creates and the name must be free, so replacing means deleting first:

```bash
bash workflows/nebius/stop.sh <endpoint-name>
bash workflows/nebius/serve.sh <vendor> <endpoint-name> <same args as before>
```

That issues a new managed URL. Whether `stop` then `start` picks the token up instead, keeping the URL, is
untested.

## Run a simulator eval (RoboLab)

> **Blocked on the platform today**: Nebius serverless jobs inject the compute driver stack only —
> no `libGLX_nvidia`/Vulkan userspace and no ICD manifest (verified empirically 2026-07-17 on
> `gpu-l40s-d`) — and Isaac's RTX renderer cannot create a GPU device without them
> (`omni.gpu_foundation_factory: Failed to create any GPU devices`). Everything below works up to
> that point (image pull, L40S scheduling, cache reuse) and is expected to work as-is once Nebius
> enables graphics injection for jobs; until then run the same eval on a VM via the
> `robolab-eval` compose service.
>
> Nebius **VM** images have the same gap on the host side: the preinstalled NVIDIA driver is
> compute-only. Provision the graphics userspace once per VM before the first run — extract
> `libnvidia-gl-<major>` (exact same version as the installed driver, deb from the CUDA apt repo)
> into `/usr/lib/x86_64-linux-gnu` + `/usr/share`, run `ldconfig`, then regenerate the CDI spec
> (`sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml`) so the container toolkit
> mounts it. The image's `NVIDIA_DRIVER_CAPABILITIES=all` handles the container side.

`eval.sh` submits `positronic eval run` as a Job on the `positro/robolab` image. The job boots
the benchmark's env server (Isaac Sim) inside the container and drives the policy over the
network — so it runs on an **L40S** platform (Isaac's RTX renderer needs RT cores; H100 has
none). `NEBIUS_PLATFORM=gpu-l40s-a` switches to the AMD-host L40S variant when the Intel one
is out of capacity.

Serve the policy first and point the eval at the endpoint's managed URL. `eval.sh` injects the
same `AUTH_TOKEN` into the job, which is where `authed_remote` reads it:

```bash
bash workflows/nebius/serve.sh openpi pi05-jointpos droid_jointpos

bash workflows/nebius/eval.sh \
  --eval=@positronic.cfg.eval.sim.robolab.banana_in_bowl \
  --eval.trial_count=10 \
  --policy=@positronic.cfg.policy.authed_remote \
  --policy.url=https://<endpoint-managed-url> \
  --output_dir=s3://<your-bucket>/evals/robolab_banana/
```

`--output_dir=s3://...` uploads the recorded eval dataset via `pos3.sync`; follow progress
with `nebius ai job logs <aijob-id> --follow`.

Unlike the other scripts, the shared cache filesystem mounts at `/root/.cache` (not `/cache`):
the env-server launcher keeps its pinned RoboLab checkout and venv at `$HOME`-relative paths,
and they dominate cold start (~25 GB of LFS assets plus the Isaac wheel stack). One mount
persists checkout, venv, and uv cache together — uv's default cache dir lands on the same
`<fs>/uv` subdir the `UV_CACHE_DIR=/cache/uv` jobs share. First run pays the full download;
warm runs skip straight to Isaac boot. Don't fan out eval jobs on a cold cache — seed it with
one run first (same rule as the shared uv cache).

## What changed vs. running on a VM

No VM to provision, SSH into, or remember to shut down. Credentials stay in MysteryBox instead
of on operator laptops. Compute is released the moment a job finishes — idle cost goes to zero.

## Configuration

Everything the scripts share — the project, subnet, cache filesystem, image tag, the auth token
secret, and the S3 credential flags — is defined once in
[`common.sh`](common.sh), which each script sources; a setting only one script means (a preset, a
timeout, a platform) stays in that script. The defaults point at Positronic Robotics' own Nebius
project — **external users must override them** with their own project + subnet IDs:

| Variable | Default (Positronic-internal) | Purpose |
|---|---|---|
| `NEBIUS_PARENT_ID` | `project-e00f38wexevrr52b8j` | Nebius project to create the job/endpoint in |
| `NEBIUS_SUBNET_ID` | `vpcsubnet-e00pk1j1x6hjmr4m92` | VPC subnet for the compute instance |
| `WANDB_SECRET` | `positronic-serverless-wandb-api-key` | MysteryBox secret name for the WandB key. Set empty (`WANDB_SECRET=`) to skip wandb entirely. |
| `NEBIUS_AUTH_TOKEN_SECRET` | `positronic-serverless-inference-token` | `serve.sh` and `eval.sh` only. MysteryBox secret name (payload key `AUTH_TOKEN`) injected as the container's `AUTH_TOKEN`. There is no open-endpoint mode. See [Authenticated inference](#authenticated-inference). |
| `NEBIUS_CACHE_FS` | `computefilesystem-e00f6jyfr5wkawyrab` | Shared filesystem **ID** (not name — `--volume` rejects names) mounted RW at `/cache` for the `uv`/HF/openpi caches (`UV_CACHE_DIR`, `HF_HOME`, `OPENPI_DATA_HOME`). Not used by pos3. The default is Positronic-internal; external users must override with their own filesystem ID. |
| `NEBIUS_IMAGE_REPO` | `positro/robolab` | *(`eval.sh` only)* Image repository the RoboLab eval job pulls, without the tag. Defaults to the Docker Hub `positro/robolab`; set it to an in-region Nebius Container Registry path (`cr.<region>.nebius.cloud/<registry-id>/robolab`) to skip the cross-cloud Docker Hub pull. `<registry-id>` is the Container Registry ID **without** the `registry-` prefix (from `nebius registry list`) — NOT the project ID. Combined with `NEBIUS_IMAGE_TAG` as `${NEBIUS_IMAGE_REPO}:${NEBIUS_IMAGE_TAG}`. |
| `NEBIUS_IMAGE_TAG` | `latest` | Docker image tag the job/endpoint pulls (`positro/<image>:<tag>`). `cd docker && make push-* IMAGE_TAG=<branch>` pushes that tag unconditionally; set `NEBIUS_IMAGE_TAG=<branch>` to run a branch build remotely without clobbering `:latest`. `make push-*` only updates `:latest` when run with `CI` set. Note `convert.sh openpi` chains a stats job on the `positro/openpi` image, so with `NEBIUS_IMAGE_TAG=<branch>` you must also have pushed `positro/openpi:<branch>` (not just `positro/positronic:<branch>`); otherwise leave `NEBIUS_IMAGE_TAG` unset so stats uses `:latest`. |

Settings with no environment override — platform, preset, the AWS MysteryBox secret names, the S3
endpoint URL and region — are changed by editing `common.sh` if they are shared, or the script
itself if they are its own. The vendor positional arg selects the container image and `uv` extras:

| Vendor | Image | `uv` extra |
|---|---|---|
| `lerobot_0_3_3` (ACT) | `positro/positronic` | `--extra lerobot_0_3_3` |
| `lerobot` (SmolVLA) | `positro/positronic` | `--extra lerobot` |
| `openpi` | `positro/openpi` | `--extra openpi` (serve); none for train/stats |
| `gr00t` | `positro/gr00t` | _(none — `/gr00t` is co-installed)_ |

## Appendix: Cleaning the shared cache

There is no file browser for the [shared cache filesystem](#shared-cache-filesystem) — to
inspect or wipe it you mount it in a throwaway job. Make sure no jobs/endpoints are using the
cache first (a wipe while a warm job reads it will break that job).

Inspect usage:

```bash
nebius ai job create --parent-id "$PARENT_ID" --subnet-id "$SUBNET_ID" \
  --name cache-du --image busybox:latest \
  --container-command du --args '-sh /cache /cache/uv /cache/hf /cache/openpi' \
  --platform cpu-e2 --preset 4vcpu-16gb --timeout 1h \
  --volume "$NEBIUS_CACHE_FS:/cache:rw"
# then: nebius ai job logs <aijob-id>
```

Wipe everything (full reset — the next run repays the cold download):

```bash
nebius ai job create --parent-id "$PARENT_ID" --subnet-id "$SUBNET_ID" \
  --name cache-wipe --image busybox:latest \
  --container-command find --args '/cache -mindepth 1 -delete' \
  --platform cpu-e2 --preset 4vcpu-16gb --timeout 1h \
  --volume "$NEBIUS_CACHE_FS:/cache:rw"
```

To clear only one tool's cache, target its subdir, e.g. `--args '/cache/uv -mindepth 1
-delete'`. Two gotchas: `--volume` needs the filesystem **ID** (not name), and Nebius
space-splits `--args`, so use a no-shell command (`find`/`du`) — a quoted `sh -c "..."`
gets torn apart. Deleting and recreating the filesystem also works but loses the warm
cache for every workflow.
