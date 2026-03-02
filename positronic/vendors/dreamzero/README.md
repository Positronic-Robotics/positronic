# DreamZero in Positronic

## What is DreamZero?

[DreamZero](https://github.com/dreamzero0/dreamzero) is NVIDIA's 14B-parameter World Action Model that jointly predicts actions and future video frames, enabling zero-shot generalization to new tasks and environments. The pretrained checkpoint (`GEAR-Dreams/DreamZero-DROID`) is trained on the DROID dataset.

## Hardware Requirements

| Phase | Requirement | Notes |
|-------|-------------|-------|
| **Inference** | 1× H100 (80GB) | ~52GB VRAM in bf16; multi-GPU via `--num_gpus=N` |
| **Training** | 4-8× H100 | LoRA or full fine-tune with DeepSpeed ZeRO-2 |

## Quick Start

```bash
# 1. Start an H100 VM
../internal/scripts/start.sh dreamzero

# 2. Start DreamZero server (checkpoint auto-downloaded on first start)
cd docker
CACHE_ROOT=/home/vertix docker --context vm-dreamzero compose run --rm --service-ports dreamzero-server

# 3. Run sim inference locally
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=vm-dreamzero \
  --policy.port=8000 \
  --driver.simulation_time=20 \
  --driver.show_gui=True
```

First start downloads the 14B checkpoint via HuggingFace (~10-20 min). Subsequent starts use cache.

## Available Codecs

| Codec | Observation | Action | Use Case |
|-------|-------------|--------|----------|
| `joints` | Joint positions + grip + 3 cameras | Absolute joint position | Default for DROID checkpoint |
| `joints_traj` | Same | Trajectory (recorded state) | Training data from recorded trajectories |

## Technical Details

- **Action space**: Absolute joint positions (7 DoF) + gripper (1)
- **Observation**: 3 cameras (2 exterior + 1 wrist) at 320×180 + joint state + language prompt
- **Action horizon**: 24 timesteps per inference
- **Wire protocol**: roboarena WebSocket + msgpack-numpy
- **Inference latency**: ~5s on 1× H100 (with DiT caching)
- **No fork needed**: Uses Hydra YAML configs, injectable from outside
