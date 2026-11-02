# DreamZero in Positronic

> **Status: Work in Progress** – DreamZero integration is functional for inference and training but
> has not been validated in production evaluations. Not included in PhAIL v1.0 results.

## What is DreamZero?

[DreamZero](https://github.com/dreamzero0/dreamzero) is NVIDIA's World Action Model: it jointly
predicts actions and future video frames, conditioning on a short window of past frames (an
autoregressive video context). Two backbones are wired up here:

| Backbone | Params | Image (W×H) | Notes |
|----------|--------|-------------|-------|
| `wan2.1` | 14B | 320×180 | Public pretrained `GEAR-Dreams/DreamZero-DROID` checkpoint (the `droid` server preset); DiT caching supported |
| `wan2.2` | 5B | 320×160 | Causal chunked inference; what the Positronic fine-tunes below use |

Pick the backbone with `--backbone` at train time and `--model.backbone` at serve time; **it must
match between the two**.

## Hardware Requirements

| Phase | Requirement | Notes |
|-------|-------------|-------|
| **Inference** | 1× H100 (80GB) | wan2.2 (5B) ≈ half the VRAM of wan2.1 (14B, ~52GB bf16) |
| **Training** | 1× or 8× H100 | Full fine-tune or LoRA, DeepSpeed ZeRO-2. Single-H100 full fine-tune works (see presets) |

Train and serve run on an H100 box you reach through a Docker context. Examples use `<h100>` for the Docker
**context** name and `<h100-host>` for the box's **hostname/IP** (what the inference client connects to on
`:8000`) — the same machine, often the same string if your context targets the host by name. (The Nebius
serverless alternative is in the [Appendix](#appendix-nebius-serverless).)

## Zero-shot inference (wan2.1 DROID checkpoint)

The DROID reference uses a standalone model server. Image resizing, JPEG encoding and action decoding
run on the client through existing codecs. The server holds one active session and a bounded image
history. It calls the native model directly, without accumulating generated videos.

```bash
# Build from the repository root on a GPU host. Mount that host's Hugging Face cache.
docker build -f docker/Dockerfile.serve-dreamzero-droid -t dreamzero-reference .
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" dreamzero-reference

# RoboLab needs its own NVIDIA GPU for rendering. It can share a sufficiently large GPU with the model.
uv run --locked positronic eval run --eval=.sim.robolab.banana_in_bowl \
  --policy=.remote --policy.address.host=<gpu-host> --policy.address.port=8000 \
  --eval.trial_count=2
```

Every `sampling_period_sec` (default `1/15`), the client sends the latest sampled observation through
ordinary inference. Between predictions, the server returns `None` and playback continues. Each
prediction includes its newest frame in the same request. The four history offsets are expressed in
seconds: `[-23/15, -16/15, -8/15, 0]`; the first prediction uses only the current frame.

The session query selects `mode=chunked` (default) or `mode=rtc`. Both use the same scheduler and
player. Chunked playback appends after the previous trajectory; RTC-like playback starts a continuation
at its observation time and skips elapsed actions. **This is video continuation, not action-prefix
conditioning:** DreamZero does not freeze previous actions like GR00T RTC does.

`prediction_period_sec` defaults to the checkpoint's output duration at 15 Hz. Set it in the session
query to request another cadence, for example `--policy.address.query='mode=rtc&prediction_period_sec=1.6'`.
Chunked playback is limited to that period so predictions cannot accumulate a growing future plan.
The default camera mapping requires two exterior views and the wrist. `image_mappings` can map both
exterior slots to the same camera on a two-camera rig.

The runtime queue is bounded: a model that cannot keep up eventually fails explicitly. A larger queue
cannot fix sustained overload. Inference time remains charged during simulator evaluations.

The wan2.2 fine-tune commands below use the legacy serving path in [`server.py`](./server.py).
TODO: migrate that path after validating the fine-tuned checkpoints with the standalone adapter.

## Full pipeline (fine-tune your own checkpoint)

The end-to-end loop — [convert → train → serve → infer](../../../docs/training-workflow.md) — runs through
the Docker Compose services in [`docker/docker-compose.yml`](../../../docker/docker-compose.yml) on your
H100 box. Run the `docker compose` commands from the `docker/` directory.

**Prerequisites:**
- An H100 box reachable via a Docker context. The compose services mount `~/.cache` and `~/.aws` from
  `CACHE_ROOT` (defaults to `$HOME`). With a **remote** context `$HOME` expands to *your* machine, so set
  `CACHE_ROOT` to the box's home — `/home/<user>` for your account on the box (e.g. `/home/ubuntu`). Run the
  commands on the box itself and you can omit it. S3 credentials come from the mounted `~/.aws`.
- **Images**: the commands pull the public prebuilt images by default — `positro/positronic` (convert) and
  `positro/dreamzero` (train/serve) at the production `:latest` tag — so there is nothing to build. Only if
  you have **local repo changes** to test do you build and push your own image and pass `IMAGE_TAG=<your-tag>`;
  otherwise omit it. A `codecs.py` change affects both images. (Maintainer build targets:
  [`docker/Makefile`](../../../docker/Makefile).)
- The `s3://interim/…` / `s3://checkpoints/…` paths below are Positronic-internal buckets — substitute your own.

### 1. Convert the dataset

The DreamZero codecs reuse the shared `lerobot_0_3_3` converter (a CPU step — run it on any box, or locally
with `uv run`). The sim stack-cubes dataset is converted with the `joints_ik_sim` codec (joint targets
reconstructed from recorded EE-pose targets via the `dm_control` IK solver):

```bash
cd docker
CACHE_ROOT=/home/<user> docker --context <h100> compose run --rm lerobot-0_3_3-convert convert \
  --dataset.dataset=@positronic.cfg.ds.sim.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.dreamzero.codecs.joints_ik_sim \
  --output_dir=s3://interim/sim_stack/dreamzero/joints_ik/
```

### 2. Train

`dreamzero-train` runs `python -m positronic.vendors.dreamzero.train`, streaming the dataset from
`--input_path`. Presets (defined in [`train.py`](./train.py)) fix backbone + architecture + GPU count:

> **Only `wan22_full` (wan2.2 full fine-tune) is validated** — that is the configuration these docs were
> built from. We did **not** get LoRA working, and the `wan2.1` presets are untested here. The other presets
> exist in code but should be treated as unverified.

| Preset | Backbone | Arch | GPUs | Resumable |
|--------|----------|------|------|-----------|
| `wan22_full_h100x1` | wan2.2 (5B) | full | 1 | No — warm-start to continue |
| `wan22_full_h100x8` | wan2.2 (5B) | full | 8 | Optimizer state not restored |
| `wan22_lora_h100x1` | wan2.2 (5B) | LoRA | 1 | LoRA trains from zero |
| `wan22_lora_h100x8` | wan2.2 (5B) | LoRA | 8 | Yes (optimizer restored) |
| `wan21_*` | wan2.1 (14B) | full / LoRA | 1 / 8 | As above |

**Fresh run** (from the base backbone):

```bash
cd docker
CACHE_ROOT=/home/<user> docker --context <h100> compose run --rm dreamzero-train \
  wan22_full_h100x1 \
  --input_path=s3://interim/sim_stack/dreamzero/joints_ik \
  --output_path=s3://checkpoints/sim_stack/dreamzero/ \
  --exp_name=<exp_name> \
  --max_steps=30000 --save_steps=2500 --gradient_accumulation_steps=4 --save_total_limit=9999
```

Checkpoints land at `s3://checkpoints/sim_stack/dreamzero/<exp_name>/checkpoint-<step>`.

**Warm-start** from a prior run's weights to continue training. The DreamZero trainer does not persist
optimizer state, so resuming restarts Adam from scratch (the optimizer state is lost) — warm-starting just
loads the weights and accepts a fresh optimizer + schedule, so aim to reach `--max_steps` in one run:

```bash
  ... dreamzero-train wan22_full_h100x1 ... \
  --init_from_checkpoint=s3://checkpoints/sim_stack/dreamzero/<prior_exp>/checkpoint-<step>
```

Multi-GPU presets (`*_h100x8`) run `torchrun --nproc_per_node=8`, so use them on an 8-GPU box.

### 3. Serve a checkpoint

`dreamzero-server <pipeline>` downloads `--model.model_path` (an `s3://` checkpoint or HF repo) and
**needs `--model.backbone` to match training** (see [Codecs](#codecs); config + defaults:
[`server.py`](./server.py)). `--service-ports` publishes the WebSocket API on `8000`:

```bash
cd docker
CACHE_ROOT=/home/<user> docker --context <h100> compose run --rm --service-ports dreamzero-server joints \
  --model.model_path=s3://checkpoints/sim_stack/dreamzero/<exp_name>/checkpoint-<step> \
  --model.backbone=wan2.2
```

Sanity-check once warm: `curl -X POST http://<h100-host>:8000/api/v1/keepalive` answers.

### 4. Run sim inference

```bash
uv run --locked positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=.remote --policy.address.host=<h100-host> --policy.address.port=8000 \
  --eval.trial_count=<N> --output_dir=<dir-or-s3-path>
```

Sim runs locally on your machine; only inference is remote. Each episode records 3 camera views +
joint/EE/grip signals under `<output_dir>`. See the [Inference Guide](../../../docs/inference.md) for the
remote-policy protocol and options.

### 5. View results

```bash
uv run --locked python -m positronic.cfg.analysis sim \
  --dataset.base.path=<output_dir> --port=5001 --https --reset_cache
# → https://localhost:5001
```

Point `--dataset.base.path` at a parent directory to compare several runs side by side.

## Codecs

A [codec](../../policy/codecs/README.md) maps raw recordings into the state/action space a model expects.
DreamZero's codecs (source: [`codecs.py`](./codecs.py)) all share the same observation encoder
(3 cameras + joint state) and the same inference decode — the model emits a flat `(joints+grip)` vector
that decodes to a `JointPosition` command. They differ only in how **training labels** are built:

| Codec | Training label | Use case |
|-------|----------------|----------|
| `joints` | Commanded joints (`robot_command.joints`) + grip | Default; commanded-action targets |
| `joints_traj` | Recorded state (`robot_state.q`) + grip | Trajectory / executed-state targets |
| `joints_ik` | Joints solved from recorded EE-pose targets via IK (`dls_limits` solver) | EE-driven datasets |
| `joints_ik_sim` | `joints_ik` with the `dm_control` IK solver | Sim datasets (used for `sim_stack_cubes`) |

Each codec has a same-named serving pipeline (see [`server.py`](./server.py)), selected as the serve
subcommand. Since the four `joints*` codecs decode inference identically, `joints` serves any of their
checkpoints. The standalone DROID recipe sends 320×180 frames and decodes actions under DROID's
impedance gains (see [Control mode](../../policy/codecs/README.md#control-mode)).

## Legacy fine-tune session parameters

A client can tune the serving pipeline per connection: session query params become dotted overrides into the
pipeline config, applied server-side (values are literals; the model checkpoint and backbone are fixed at
launch). With `positronic eval run`, pass them through the remote policy:

```bash
# At episode start, send only the observed history (a growing frame stack) instead of
# padding the window with the current frame repeated.
uv run --locked positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=.remote --policy.address.host=<h100-host> --policy.address.port=8000 --policy.address.query='local.pad_start=false' \
  --eval.trial_count=2
```

`local` is the rig-side video-context stack, `codec` the server-side codec (cadence is `local.fps`);
protocol details in the [Inference Guide](../../../docs/inference.md).

## Technical Details

- **Action space**: 7-DoF joint targets + gripper. DreamZero predicts the joints **relative to the current
  state** (`relative_action: true` on `joint_position` in the `droid_relative_wan22` data config); the
  absolute joint target is reconstructed from the current state at serve time and decoded to a `JointPosition`.
- **Observation**: 3 cameras (2 exterior + 1 wrist) + joint state + language prompt. The wan2.2 fine-tunes
  use the `(320, 176)` codec and the trainer resizes to 160 at load; the pretrained DROID model (wan2.1)
  asserts exactly `(320, 180)`, so the `droid` codec (`codecs.droid`) feeds that resolution.
- **Action horizon**: 24 timesteps per inference; the server-declared `dreamzero_layers` re-query
  aligns the chunk schedule with the AR frame-stack window
- **Wire protocol**: Positronic's standard WebSocket protocol — see [Connect Your Model](../../../docs/connect-your-model.md)
- **A roboarena server with no Positronic server in front**: `roboarena_policy.RoboarenaPolicy(address, headers)` dials
  the server on the `roboarena` wire and runs the `droid_3cam` codec on the robot side.
- **No Positronic fork**: upstream DreamZero is used unmodified (pinned SHA in [`Dockerfile`](./Dockerfile));
  configs are injected via Hydra YAML. No sibling `../dreamzero` checkout is needed — the image bakes it in.

## Appendix: Nebius serverless

If you run on Nebius serverless instead of your own H100 box, the **same recipe** (same presets, codec, and
args) goes through the `workflows/nebius/*.sh` wrappers — Jobs for convert/train, an Endpoint for serving.
See [`workflows/nebius/README.md`](../../../workflows/nebius/README.md) for the one-time setup (Nebius CLI
auth, S3 / MysteryBox credentials, the shared cache filesystem) and `NEBIUS_*` overrides. It uses the
production image by default; set `NEBIUS_IMAGE_TAG=<tag>` only to test a specific build.

```bash
# Convert (CPU job)
bash workflows/nebius/convert.sh lerobot_0_3_3 \
  --dataset.dataset=@positronic.cfg.ds.sim.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.dreamzero.codecs.joints_ik_sim \
  --output_dir=s3://interim/sim_stack/dreamzero/joints_ik/

# Train (H100 job; ~3 days for 30k steps on one H100 — size the timeout accordingly)
NEBIUS_JOB_TIMEOUT=96h \
bash workflows/nebius/train.sh dreamzero wan22_full_h100x1 \
  --input_path=s3://interim/sim_stack/dreamzero/joints_ik \
  --output_path=s3://checkpoints/sim_stack/dreamzero/ \
  --exp_name=<exp_name> \
  --max_steps=30000 --save_steps=2500 --gradient_accumulation_steps=4 --save_total_limit=9999

# Serve (H100 endpoint, reachable at the managed https:// URL the banner prints)
bash workflows/nebius/serve.sh dreamzero <endpoint-name> joints \
  --model.model_path=s3://checkpoints/sim_stack/dreamzero/<exp_name>/checkpoint-<step> \
  --model.backbone=wan2.2
# ... infer with --policy=.authed_remote --policy.wire=websocket_tls --policy.address.host=<managed-host> --policy.address.port=443 (export AUTH_TOKEN first,
# see workflows/nebius/README.md), then tear down:
bash workflows/nebius/stop.sh <endpoint-name>
```
