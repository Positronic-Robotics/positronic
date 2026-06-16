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

## Zero-shot inference (wan2.1 DROID checkpoint)

To try the pretrained DROID model with no training:

```bash
# 1. Start an H100 box (see ../internal/scripts/start.sh) or use the Nebius serve path below.
# 2. Serve the default checkpoint (auto-downloaded on first start, ~10-20 min via HuggingFace).
cd docker
CACHE_ROOT=/home/vertix docker --context vm-dreamzero compose run --rm --service-ports dreamzero-server

# 3. Run sim inference locally (only inference is remote; MuJoCo runs on your machine).
uv run --locked positronic-inference sim \
  --policy=.remote --policy.host=vm-dreamzero --policy.port=8000 \
  --wrap=@positronic.vendors.dreamzero.codecs.dreamzero_wrappers \
  --trial_count=2 --show_gui=True
```

The server owns the codec (`server` defaults to `codec=joints`), so the client sends raw observations
and receives decoded joint commands — **don't** pass `--policy.codec` here or it double-encodes and the
first request fails. `--wrap` is client-side: it supplies the AR video context (`TemporalFrameStack`),
which must run every control tick to record frames, so without it the model loses the multi-frame
history it conditions on.

## Full pipeline (fine-tune your own checkpoint)

The end-to-end loop is **convert → train → serve → infer → view**. Convert runs on CPU; train and
serve run on H100. The cloud wrappers live in `workflows/nebius/` (see the `remote-training` skill and
`workflows/nebius/README.md`); pass `NEBIUS_IMAGE_TAG=<tag>` to pin the image (CI pushes `:latest` and
the commit SHA from `main` — pin to a SHA, e.g. `dc5e837`, for a reproducible long run).

### 1. Convert the dataset

The DreamZero codecs reuse the shared `lerobot_0_3_3` converter. The sim stack-cubes dataset is
converted with the `joints_ik_sim` codec (joint targets reconstructed from recorded EE-pose targets
via the `dm_control` IK solver):

```bash
bash workflows/nebius/convert.sh lerobot_0_3_3 \
  --dataset.dataset=@positronic.cfg.ds.sim.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.dreamzero.codecs.joints_ik_sim \
  --output_dir=s3://interim/sim_stack/dreamzero/joints_ik/
```

### 2. Train

`train.sh dreamzero <preset> [args]` runs `python -m positronic.vendors.dreamzero.train` as an H100
job, streaming the dataset from `--input_path` via `pos3`. Presets fix backbone + architecture + GPU
count:

| Preset | Backbone | Arch | GPUs | Resumable |
|--------|----------|------|------|-----------|
| `wan22_full_h100x1` | wan2.2 (5B) | full | 1 | No — warm-start to continue |
| `wan22_full_h100x8` | wan2.2 (5B) | full | 8 | Optimizer state not restored |
| `wan22_lora_h100x1` | wan2.2 (5B) | LoRA | 1 | LoRA trains from zero |
| `wan22_lora_h100x8` | wan2.2 (5B) | LoRA | 8 | Yes (optimizer restored) |
| `wan21_*` | wan2.1 (14B) | full / LoRA | 1 / 8 | As above |

Multi-GPU presets need a matching Nebius preset (`*_h100x8` runs `torchrun --nproc_per_node=8`, so set
`NEBIUS_PRESET=8gpu-128vcpu-1600gb`).

**Fresh run** (from the base backbone):

```bash
NEBIUS_IMAGE_TAG=dc5e837 NEBIUS_JOB_TIMEOUT=96h \
bash workflows/nebius/train.sh dreamzero wan22_full_h100x1 \
  --input_path=s3://interim/sim_stack/dreamzero/joints_ik \
  --output_path=s3://checkpoints/sim_stack/dreamzero/ \
  --exp_name=<exp_name> \
  --max_steps=30000 --save_steps=2500 --gradient_accumulation_steps=4 --save_total_limit=9999
```

Checkpoints land at `s3://checkpoints/sim_stack/dreamzero/<exp_name>/checkpoint-<step>`.

**Warm-start** from a prior run's weights (fresh optimizer + schedule) — the only way to extend an
`h100x1` full fine-tune, since it can't restore DeepSpeed state:

```bash
  ... wan22_full_h100x1 ... \
  --init_from_checkpoint=s3://checkpoints/sim_stack/dreamzero/<prior_exp>/checkpoint-<step>
```

`h100x1` full fine-tune uses `save_only_model=true` (skips DeepSpeed's ~91GB native checkpoint), so a
run must reach `--max_steps` in one shot — size `NEBIUS_JOB_TIMEOUT` accordingly (~3 days for 30k
steps on one H100 at the observed throughput).

### 3. Serve a checkpoint

The server downloads `--model_path` (an `s3://` checkpoint or HF repo) via `pos3` and **needs
`--backbone` to match training**. Two paths:

```bash
# A. Nebius serverless endpoint (public IP, auto-released on stop)
NEBIUS_IMAGE_TAG=dc5e837 bash workflows/nebius/serve.sh dreamzero <endpoint-name> \
  --model_path=s3://checkpoints/sim_stack/dreamzero/<exp_name>/checkpoint-<step> \
  --backbone=wan2.2
# ... run inference against the printed IP, then:
bash workflows/nebius/stop.sh <endpoint-name>

# B. Docker compose on a GPU box (no per-hour serverless cost)
cd docker
CACHE_ROOT=/home/vertix IMAGE_TAG=dc5e837 \
docker --context vm-dreamzero compose run --rm --service-ports dreamzero-server \
  --model_path=s3://checkpoints/sim_stack/dreamzero/<exp_name>/checkpoint-<step> \
  --backbone=wan2.2
```

Sanity-check once warm: `curl http://<host>:8000/api/v1/models` → `{"models": ["<model_path>"]}`.

### 4. Run sim inference

```bash
uv run --locked positronic-inference sim \
  --policy=.remote --policy.host=<host> --policy.port=8000 \
  --wrap=@positronic.vendors.dreamzero.codecs.dreamzero_wrappers \
  --trial_count=<N> --output_dir=<dir-or-s3-path>
```

`<host>` is the Nebius endpoint IP or the docker context hostname (`vm-dreamzero`). Each episode records
3 camera views + joint/EE/grip signals under `<output_dir>`.

### 5. View results

```bash
uv run --locked python -m positronic.cfg.analysis sim \
  --dataset.base.path=<output_dir> --port=5001 --https --reset_cache
# → https://localhost:5001
```

Point `--dataset.base.path` at a parent directory to compare several runs side by side.

## Codecs

All four share the same observation encoder (3 cameras + joint state) and the same inference decode —
the model emits a flat `(joints+grip)` vector that decodes to a `JointPosition` command. They differ
only in how **training labels** are built:

| Codec | Training label | Use case |
|-------|----------------|----------|
| `joints` | Commanded joints (`robot_command.joints`) + grip | Default; commanded-action targets |
| `joints_traj` | Recorded state (`robot_state.q`) + grip | Trajectory / executed-state targets |
| `joints_ik` | Joints solved from recorded EE-pose targets via IK (`dls_limits` solver) | EE-driven datasets |
| `joints_ik_sim` | `joints_ik` with the `dm_control` IK solver | Sim datasets (used for `sim_stack_cubes`) |

## Technical Details

- **Action space**: absolute joint positions (7 DoF) + gripper (1)
- **Observation**: 3 cameras (2 exterior + 1 wrist) + joint state + language prompt; 320×160 (wan2.2),
  320×176 (wan2.1)
- **Action horizon**: 24 timesteps per inference; the `dreamzero_wrappers` re-query aligns the
  chunk schedule with the AR frame-stack window
- **Wire protocol**: roboarena WebSocket + msgpack-numpy
- **DiT caching**: wan2.1 only (skipped for wan2.2)
- **No fork needed**: Hydra YAML configs, injectable from outside
