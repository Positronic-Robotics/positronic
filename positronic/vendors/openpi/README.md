# OpenPI (π₀.₅) in Positronic

## What is OpenPI?

OpenPI (π₀.₅) is a foundation model for robotics trained by [Physical Intelligence](https://www.physicalintelligence.company/) on diverse manipulation tasks. It represents the most capable robotic foundation model, offering strong generalization across different manipulation scenarios.

Positronic provides first-class support for OpenPI including:
- Optimized training configuration for single capable GPU (LoRA-based, ~78GB)
- Inference support (~62GB, likely cloud deployment)
- Multiple codec variants for different robot platforms
- Unified inference API compatible with all Positronic hardware
- Integration with our fork: [Positronic-Robotics/openpi](https://github.com/Positronic-Robotics/openpi), kept up to date with upstream

See [Model Selection Guide](../../docs/model-selection.md) for model comparison.

## Hardware Requirements

| Phase | Requirement | Notes |
|-------|-------------|-------|
| **Training** | capable server GPU (~78GB) | LoRA config (`pi05_positronic_lowmem`) fits on a single capable GPU |
| **Inference** | GPU (~62GB) | Likely cloud deployment (e.g., capable GPU) |
| **Training Time** | Multiple days | Typical for OpenPI |

## Quick Start

```bash
# 1. Convert dataset
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.dataset.path=~/datasets/my_task_raw \
  --dataset.codec=@positronic.vendors.openpi.codecs.eepose_absolute \
  --output_dir=~/datasets/openpi/my_task \
  --fps=15

# 2. Generate assets (required for OpenPI)
cd docker && docker compose run --rm openpi-stats \
  --input_path=~/datasets/openpi/my_task \
  --output_path=~/datasets/openpi_assets

# 3. Train
cd docker && docker compose run --rm openpi-train \
  --input_path=~/datasets/openpi/my_task \
  --stats_path=~/datasets/openpi_assets/assets/ \
  --output_path=~/checkpoints/openpi \
  --exp_name=my_task_v1 \
  --config_name=pi05_positronic_lowmem

# 4. Serve
cd docker && docker compose run --rm --service-ports openpi-server \
  --checkpoints_dir=~/checkpoints/openpi/pi05_positronic_lowmem/my_task_v1/ \
  --codec=@positronic.vendors.openpi.codecs.eepose_absolute

# 5. Run inference
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=localhost \
  --driver.show_gui=True
```

See [Training Workflow](../../docs/training-workflow.md) for detailed step-by-step instructions.

## Available Codecs

OpenPI supports multiple codecs for different robot platforms and observation/action formats.

| Codec | Observation | Action | Use Case |
|-------|-------------|--------|----------|
| `eepose_absolute` | EE pose (7D quat) + grip + images | Absolute EE position (7D) | Default for Positronic datasets, task-space control |
| `openpi_positronic` | EE pose (OpenPI format) + grip + images | Absolute EE position (7D) | OpenPI-native key format |
| `droid` | Joint positions + grip + images | Joint delta (velocity) | DROID dataset compatibility, joint-space control |
| `eepose_q` | EE pose + joints + grip + images | Absolute EE position (7D) | Combined feedback (richer observations) |
| `joints` | Joint positions + grip + images | Absolute EE position (7D) | Joint observations, task-space control |

**Key differences:**
- **`eepose_absolute` vs `openpi_positronic`**: Same semantics, different key format (`observation.state` vs `observation/state`)
- **`droid`**: For inference with existing DROID datasets (joint-based observations, delta actions). Use for evaluation, not training from scratch.
- **`eepose_q`**: Includes both EE pose and joint positions for richer feedback

**Choosing a codec:**
- **Positronic datasets (simulated/hardware)**: Use `eepose_absolute`
- **DROID dataset**: Use `droid` codec (joint-based observations, delta actions)
- **Want joint feedback**: Use `eepose_q` (may improve performance)

See [Codecs Guide](../../docs/codecs.md) for comprehensive codec documentation.

## Configuration Reference

### Training Configuration

Default config: `pi05_positronic_lowmem` (LoRA-based, fits on 1x H100 GPU)

**Common parameters:**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--config_name` | OpenPI training config | `pi05_positronic_lowmem` | `pi05_droid` |
| `--exp_name` | Experiment name (unique ID) | Required | `my_task_v1` |
| `--num_train_steps` | Total training steps | Config default | `100000` |
| `--resume` | Resume from existing checkpoint | `False` | `True` |
| `--stats_path` | Path to generated assets | Required | `~/datasets/openpi_assets/assets/` |
| `--output_path` | Checkpoint destination | Required | `~/checkpoints/openpi` |

**WandB logging:** Enabled by default if `WANDB_API_KEY` is set in `docker/.env.wandb`.

### Inference Server Configuration

```bash
cd docker && docker compose run --rm --service-ports openpi-server \
  --checkpoints_dir=~/checkpoints/openpi/pi05_positronic_lowmem/my_task_v1/ \
  --codec=@positronic.vendors.openpi.codecs.eepose_absolute \
  --config_name=pi05_positronic_lowmem \
  --port=8000
```

**Server parameters:**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--checkpoints_dir` | Experiment directory (contains `checkpoint-N` folders) | Required | `~/checkpoints/openpi/.../my_task_v1/` |
| `--checkpoint` | Specific checkpoint step | Latest | `10000`, `50000` |
| `--codec` | Codec (must match training) | Based on variant | `@positronic.vendors.openpi.codecs.eepose_absolute` |
| `--config_name` | OpenPI config name | `pi05_positronic_lowmem` | Same as training |
| `--port` | Server port | `8000` | `8001` |
| `--openpi_ws_port` | Internal OpenPI subprocess port | `8001` | `8002` |

## Troubleshooting

### Server Fails to Start

**Problem:** Server exits with "OpenPI subprocess exited with code 1"

**Solutions:**
1. Verify checkpoint directory exists and contains `checkpoint-N/` folders
2. Check `--config_name` matches the training config used
3. Ensure OpenPI repository is available at `../openpi/` (sibling directory)

## See Also

**Positronic Documentation:**
- [Model Selection Guide](../../docs/model-selection.md) — When to use OpenPI vs GR00T vs LeRobot
- [Codecs Guide](../../docs/codecs.md) — Understanding observation/action encoding
- [Training Workflow](../../docs/training-workflow.md) — Unified training steps across all models
- [Inference Guide](../../docs/inference.md) — Deployment and evaluation patterns

**Other Models:**
- [GR00T](../groot/README.md) — NVIDIA's generalist robot policy
- [LeRobot ACT](../lerobot/README.md) — Single-task transformer

**External:**
- [Physical Intelligence](https://www.physicalintelligence.company/) — OpenPI creators
- [Positronic OpenPI Fork](https://github.com/Positronic-Robotics/openpi) — Our integration branch
