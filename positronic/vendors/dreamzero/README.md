# DreamZero in Positronic

> **Status: Work in Progress** – DreamZero integration is functional for inference and training but
> has not been validated in production evaluations. Not included in PhAIL v1.0 results.

## What is DreamZero?

[DreamZero](https://github.com/dreamzero0/dreamzero) is NVIDIA's World Action Model: it jointly
predicts actions and future video frames, conditioning on a short window of past frames (an
autoregressive video context). Two backbones are wired up here:

| Backbone | Params | Image (W×H) | Notes |
|----------|--------|-------------|-------|
| `wan2.1` | 14B | 320×176 | Default server checkpoint `GEAR-Dreams/DreamZero-DROID` (trained on DROID); DiT caching supported |
| `wan2.2` | 5B | 320×160 | Causal chunked inference; what the Positronic fine-tunes below use |

Pick the backbone with `--backbone` at both train and serve time; **it must match between the two**.

## Hardware Requirements

| Phase | Requirement | Notes |
|-------|-------------|-------|
| **Inference** | 1× H100 (80GB) | wan2.2 (5B) ≈ half the VRAM of wan2.1 (14B, ~52GB bf16) |
| **Training** | 1× or 8× H100 | Full fine-tune or LoRA, DeepSpeed ZeRO-2. Single-H100 full fine-tune works (see presets) |

Train and serve run on an H100 box you reach through a Docker context. The examples below call it
`--context <h100>` — substitute your own context name. (The Nebius serverless alternative is in the
[Appendix](#appendix-nebius-serverless).)

## Zero-shot inference (wan2.1 DROID checkpoint)

To try the pretrained DROID model with no training. `positronic-inference` comes from `uv sync` (see
[Installation](../../../README.md#installation)); `CACHE_ROOT` must be the **remote** box's home directory.

```bash
# Serve the default checkpoint on your H100 box (auto-downloaded on first start, ~10-20 min via HuggingFace).
cd docker
CACHE_ROOT=/home/vertix docker --context <h100> compose run --rm --service-ports dreamzero-server

# Run sim inference locally (only inference is remote; MuJoCo runs on your machine).
uv run --locked positronic-inference sim \
  --policy=.remote --policy.host=<h100> --policy.port=8000 \
  --wrap=@positronic.vendors.dreamzero.codecs.dreamzero_wrappers \
  --trial_count=2 --show_gui=True
```

The server owns the codec, so you don't pass one client-side — it takes raw observations and returns
decoded joint commands. The client-side piece that matters is `--wrap`: it supplies the model's
autoregressive video context (`TemporalFrameStack`) and must run every control tick to record frames,
so omitting it strips the multi-frame history the model conditions on. (Server config and defaults:
[`server.py`](./server.py); wrappers: [`codecs.py`](./codecs.py); remote-policy protocol:
[Inference Guide](../../../docs/inference.md).)

## Full pipeline (fine-tune your own checkpoint)

The end-to-end loop — [convert → train → serve → infer](../../../docs/training-workflow.md) — runs through
the Docker Compose services in [`docker/docker-compose.yml`](../../../docker/docker-compose.yml) on your
H100 box. Run the `docker compose` commands from the `docker/` directory.

**Prerequisites:**
- An H100 box reachable via a Docker context; set `CACHE_ROOT` to that box's home so the `~/.cache` and
  `~/.aws` mounts resolve (S3 credentials come from the mounted `~/.aws`).
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
CACHE_ROOT=/home/vertix docker --context <h100> compose run --rm lerobot-0_3_3-convert convert \
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
CACHE_ROOT=/home/vertix docker --context <h100> compose run --rm dreamzero-train \
  wan22_full_h100x1 \
  --input_path=s3://interim/sim_stack/dreamzero/joints_ik \
  --output_path=s3://checkpoints/sim_stack/dreamzero/ \
  --exp_name=<exp_name> \
  --max_steps=30000 --save_steps=2500 --gradient_accumulation_steps=4 --save_total_limit=9999
```

Checkpoints land at `s3://checkpoints/sim_stack/dreamzero/<exp_name>/checkpoint-<step>`.

**Warm-start** from a prior run's weights (fresh optimizer + schedule) — the only way to extend an
`h100x1` full fine-tune, which uses `save_only_model=true` and can't restore DeepSpeed state, so a run must
reach `--max_steps` in one shot:

```bash
  ... dreamzero-train wan22_full_h100x1 ... \
  --init_from_checkpoint=s3://checkpoints/sim_stack/dreamzero/<prior_exp>/checkpoint-<step>
```

Multi-GPU presets (`*_h100x8`) run `torchrun --nproc_per_node=8`, so use them on an 8-GPU box.

### 3. Serve a checkpoint

`dreamzero-server` downloads `--model_path` (an `s3://` checkpoint or HF repo) and **needs `--backbone` to
match training** (config + defaults: [`server.py`](./server.py)). `--service-ports` publishes the WebSocket
API on `8000`:

```bash
cd docker
CACHE_ROOT=/home/vertix docker --context <h100> compose run --rm --service-ports dreamzero-server \
  --model_path=s3://checkpoints/sim_stack/dreamzero/<exp_name>/checkpoint-<step> \
  --backbone=wan2.2
```

Sanity-check once warm: `curl http://<h100>:8000/api/v1/models` → `{"models": ["<model_path>"]}`.

### 4. Run sim inference

```bash
uv run --locked positronic-inference sim \
  --policy=.remote --policy.host=<h100> --policy.port=8000 \
  --wrap=@positronic.vendors.dreamzero.codecs.dreamzero_wrappers \
  --trial_count=<N> --output_dir=<dir-or-s3-path>
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

A [codec](../../../docs/codecs.md) maps raw recordings into the state/action space a model expects.
DreamZero's codecs (source: [`codecs.py`](./codecs.py)) all share the same observation encoder
(3 cameras + joint state) and the same inference decode — the model emits a flat `(joints+grip)` vector
that decodes to a `JointPosition` command. They differ only in how **training labels** are built:

| Codec | Training label | Use case |
|-------|----------------|----------|
| `joints` | Commanded joints (`robot_command.joints`) + grip | Default; commanded-action targets |
| `joints_traj` | Recorded state (`robot_state.q`) + grip | Trajectory / executed-state targets |
| `joints_ik` | Joints solved from recorded EE-pose targets via IK (`dls_limits` solver) | EE-driven datasets |
| `joints_ik_sim` | `joints_ik` with the `dm_control` IK solver | Sim datasets (used for `sim_stack_cubes`) |

## Technical Details

- **Action space**: absolute joint positions (7 DoF) + gripper (1)
- **Observation**: 3 cameras (2 exterior + 1 wrist) + joint state + language prompt; 320×160 (wan2.2),
  320×176 (wan2.1). The wan2.2 trainer sets `image_resolution_height=160` and resizes at load, so the
  codec's intermediate `(320, 176)` does not need a per-backbone override.
- **Action horizon**: 24 timesteps per inference; the `dreamzero_wrappers` re-query aligns the
  chunk schedule with the AR frame-stack window
- **Wire protocol**: roboarena WebSocket + msgpack-numpy
- **DiT caching**: wan2.1 only (skipped for wan2.2)
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

# Serve (H100 endpoint; the public endpoint is exposed on :8000)
bash workflows/nebius/serve.sh dreamzero <endpoint-name> \
  --model_path=s3://checkpoints/sim_stack/dreamzero/<exp_name>/checkpoint-<step> \
  --backbone=wan2.2
# ... infer against the printed endpoint IP with --policy.port=8000, then tear down:
bash workflows/nebius/stop.sh <endpoint-name>
```
