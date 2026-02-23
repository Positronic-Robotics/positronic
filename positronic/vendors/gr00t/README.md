# GR00T in Positronic

## What is GR00T?

cGR00T is [NVIDIA's](https://developer.nvidia.com/isaac/groot) generalist robot foundation model for versatile robot control.

Positronic provides first-class support for GR00T including:
- Training on single capable server GPU (~50GB)
- Inference on smaller GPU (~7.5GB, can run closer to robot)
- Relative modalities support (uses same codecs, different groot model internally to match OpenPI's relative actions by default)
- Unified inference API compatible with all Positronic hardware
- Integration with our fork: [Positronic-Robotics/gr00t](https://github.com/Positronic-Robotics/gr00t), kept up to date with upstream

See [Model Selection Guide](../../docs/model-selection.md) for comparison with other options.

## Hardware Requirements

| Phase | Requirement | Notes |
|-------|-------------|-------|
| **Training** | capable sever GPU (~50GB) | NVIDIA's training config optimized for a single capable GPU |
| **Inference** | GPU (~7.5GB) | RTX 4070, A10, or better (can run on robot) |
| **Training Time** | 0.5-2 days | Typical for GR00T |

## Quick Start

```bash
# 1. Convert dataset
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.dataset.path=~/datasets/my_task_raw \
  --dataset.codec=@positronic.vendors.gr00t.codecs.ee_rot6d_joints \
  --output_dir=~/datasets/groot/my_task \
  --fps=15

# 2. Train
cd docker && docker compose run --rm groot-train \
  --input_path=~/datasets/groot/my_task \
  --output_path=~/checkpoints/groot \
  --exp_name=my_task_v1 \
  --modality_config=ee_rot6d_q

# 3. Serve
cd docker && docker compose run --rm --service-ports groot-server \
  ee_rot6d_joints \
  --checkpoints_dir=~/checkpoints/groot/my_task_v1/

# 4. Run inference
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=localhost \
  --driver.show_gui=True
```

See [Training Workflow](../../docs/training-workflow.md) for detailed step-by-step instructions.

## Available Codecs

GR00T supports multiple codecs with different rotation representations and observation spaces.

| Codec | Observation | Action | Modality Config | Use Case |
|-------|-------------|--------|-----------------|----------|
| `ee_quat` | EE pose (quat) + grip + images | Absolute EE position (quat) + grip | `ee` | Default EE control, quaternion rotation |
| `ee_rot6d` | EE pose (rot6d) + grip + images | Absolute EE position (rot6d) + grip | `ee_rot6d` | 6D rotation representation |
| `ee_quat_joints` | EE pose + joints + grip + images | Absolute EE position + grip | `ee_q` | Combined EE + joint feedback |
| `ee_rot6d_joints` | EE pose (rot6d) + joints + grip + images | Absolute EE position (rot6d) + grip | `ee_rot6d_q` | 6D rotation + joint feedback (recommended) |

**Key features:**
- **Rotation representations**: Quaternion (4D) vs rot6d (6D continuous)
- **Joint feedback**: Optional joint position observations for richer state representation
- Images automatically resized to 224x224
- Sets `gr00t_modality` metadata for training compatibility

**Codec must match modality config during training:**

| Codec | Training Modality |
|-------|-------------------|
| `ee_quat` | `ee` |
| `ee_rot6d` | `ee_rot6d` |
| `ee_quat_joints` | `ee_q` |
| `ee_rot6d_joints` | `ee_rot6d_q` |

**Recommendation:** Use `ee_rot6d_joints` for best performance (6D rotation is continuous, joint feedback improves learning).

See [Codecs Guide](../../docs/codecs.md) for comprehensive codec documentation.

## Configuration Reference

### Training Configuration

**Common parameters:**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--modality_config` | Modality configuration (must match codec) | `ee` | `ee_rot6d_q` |
| `--exp_name` | Experiment name (unique ID) | Required | `my_task_v1` |
| `--num_train_steps` | Total training steps | Config default | `100000` |
| `--learning_rate` | Override learning rate | Config default | `1e-4` |
| `--save_steps` | Checkpoint save interval | Config default | `10000` |
| `--num_workers` | Dataloader workers | Config default | `8` |
| `--resume` | Resume from existing checkpoint | `False` | `True` |
| `--output_path` | Checkpoint destination | Required | `~/checkpoints/groot` |

**WandB logging:** Enabled by default if `WANDB_API_KEY` is set in `docker/.env.wandb`.

### Inference Server Configuration

GR00T server uses pre-configured variants that combine codec and modality configuration:

```bash
cd docker && docker compose run --rm --service-ports groot-server \
  ee_rot6d_joints \
  --checkpoints_dir=~/checkpoints/groot/my_task_v1/ \
  --port=8000
```

**Available variants:**
- `ee` - End-effector pose (quaternion)
- `ee_joints` - End-effector pose + joint positions (quaternion)
- `ee_rot6d` - End-effector pose (rot6d)
- `ee_rot6d_joints` - End-effector pose + joint positions (rot6d, recommended)
- `ee_rot6d_rel` - End-effector pose (rot6d, relative actions)
- `ee_rot6d_joints_rel` - End-effector pose + joint positions (rot6d, relative actions)

**Server parameters:**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `variant` | Pre-configured variant (positional arg) | Required | `ee_rot6d_joints` |
| `--checkpoints_dir` | Experiment directory (contains `checkpoint-N` folders) | Required | `~/checkpoints/groot/my_task_v1/` |
| `--checkpoint` | Specific checkpoint ID | Latest | `10000`, `50000` |
| `--port` | Server port | `8000` | `8001` |

## Troubleshooting

### GR00T Modality Mismatch

**Problem:** Training or inference fails with modality-related errors

**Cause:** Codec and modality config don't match

**Solution:** Use the correct pairing (see table in [Available Codecs](#available-codecs)):

```bash
# Codec: ee_rot6d_joints → Modality: ee_rot6d_q

# Training
cd docker && docker compose run --rm groot-train \
  --modality_config=ee_rot6d_q \
  --input_path=~/datasets/groot/my_task  # (converted with ee_rot6d_joints codec)

# Inference (use matching variant)
cd docker && docker compose run --rm --service-ports groot-server \
  ee_rot6d_joints \
  --checkpoints_dir=~/checkpoints/groot/my_task_v1/
```

## See Also

**Positronic Documentation:**
- [Model Selection Guide](../../docs/model-selection.md) — When to use GR00T vs OpenPI vs LeRobot
- [Codecs Guide](../../docs/codecs.md) — Understanding observation/action encoding
- [Training Workflow](../../docs/training-workflow.md) — Unified training steps across all models
- [Inference Guide](../../docs/inference.md) — Deployment and evaluation patterns

**Other Models:**
- [OpenPI (π₀.₅)](../openpi/README.md) — Recommended for most tasks, most capable foundation model
- [LeRobot ACT](../lerobot/README.md) — Single-task transformer, fast training

**External:**
- [NVIDIA GR00T](https://developer.nvidia.com/isaac/groot) — Official GR00T page
- [Positronic GR00T Fork](https://github.com/Positronic-Robotics/gr00t) — Our integration repository
