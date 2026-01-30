# LeRobot ACT in Positronic

## What is LeRobot ACT?

LeRobot ACT (Action Chunking Transformer) is a single-task imitation learning model from [HuggingFace LeRobot](https://github.com/huggingface/lerobot). Based on the [ACT paper](https://arxiv.org/abs/2304.13705) (foundational work in recent ML robotics), it's designed for efficient learning on focused manipulation tasks, offering excellent performance with modest hardware requirements and fast training times.

ACT uses action chunking to output sequences of future actions, enabling smooth execution and reducing compounding errors. This makes it particularly effective for precise manipulation tasks where consistency and repeatability are critical.

## Standing on LeRobot's Shoulders

**We use HuggingFace LeRobot's training scripts and workflows.** LeRobot provides battle-tested implementations of imitation learning algorithms and a strong open-source community.

Positronic adds the operational infrastructure LeRobot doesn't focus on:
- **Data ops**: Format-agnostic dataset storage, codec projection, multi-rate recording
- **Hardware integration**: Drivers for Franka, Kinova, SO101, DROID; WebXR teleoperation
- **Unified inference**: Vendor-agnostic API for deploying LeRobot alongside GR00T and OpenPI
- **Lifecycle management**: Curation tools, iteration workflows, monitoring

**LeRobot**: Training-centric. Fast experiments on reference hardware and public datasets.
**Positronic**: Lifecycle-centric. Adds infrastructure for hardware integration and lifecycle management.

**Roadmap**: Support all LeRobot models.


See [Model Selection Guide](../../docs/model-selection.md) for comparison.

## Hardware Requirements

| Phase | Requirement | Notes |
|-------|-------------|-------|
| **Training** | Consumer GPU (RTX 3090, 4090) | 16GB+ VRAM recommended, 8GB minimum |
| **Inference** | Consumer GPU (4GB+) | RTX 3060, 4060, or similar |
| **Development** | CPU acceptable | For testing (slower inference) |

## Quick Start

```bash
# 1. Convert dataset
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.dataset.path=~/datasets/my_task_raw \
  --dataset.codec=@positronic.vendors.lerobot.codecs.eepose_absolute \
  --output_dir=~/datasets/lerobot/my_task \
  --fps=30

# 2. Train
cd docker && docker compose run --rm lerobot-train \
  --input_path=~/datasets/lerobot/my_task \
  --exp_name=my_task_v1 \
  --output_dir=~/checkpoints/lerobot/ \
  --num_train_steps=50000 \
  --save_freq=10000

# 3. Serve
cd docker && docker compose run --rm --service-ports lerobot-server \
  --checkpoints_dir=~/checkpoints/lerobot/my_task_v1/ \
  --codec=@positronic.vendors.lerobot.codecs.eepose_absolute

# 4. Run inference
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=localhost \
  --driver.show_gui=True
```

See [Training Workflow](../../docs/training-workflow.md) for detailed step-by-step instructions.

## Available Codecs

LeRobot supports two primary codecs for different observation/action configurations.

| Codec | Observation | Action | Use Case |
|-------|-------------|--------|----------|
| `eepose_absolute` | EE pose (7D quat) + grip (1D) + images | Absolute EE position (7D quat) + grip | Default codec for end-effector control, task-space manipulation |
| `joints_absolute` | Joint positions (7D) + grip (1D) + images | Absolute EE position (7D quat) + grip | Joint-space observations with task-space control |

**Key features:**
- Uses `task_field='task'` (LerobotPolicy filters this before passing to ACT)
- Images resized to 480x480
- Quaternion rotation representation (7D)
- Absolute action space (not delta)

**Choosing a codec:**
- **Most tasks**: Use `eepose_absolute` (task-space observations and control)
- **Want joint feedback**: Use `joints_absolute` (may improve performance with joint position information)

See [Codecs Guide](../../docs/codecs.md) for comprehensive codec documentation.

## Configuration Reference

### Training Configuration

**Common parameters:**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--codec` | Override codec | `eepose_absolute` | `joints_absolute` |
| `--exp_name` | Experiment name (unique ID) | Required | `my_task_v1` |
| `--num_train_steps` | Total training steps | `50000` | `100000` |
| `--save_freq` | Checkpoint save interval | `10000` | `5000` |
| `--resume` | Resume from existing checkpoint | `False` | `True` |
| `--output_dir` | Checkpoint destination | Required | `~/checkpoints/lerobot/` |

**WandB logging:** Enabled by default if `WANDB_API_KEY` is set in `docker/.env.wandb`.

### Inference Server Configuration

```bash
cd docker && docker compose run --rm --service-ports lerobot-server \
  --checkpoints_dir=~/checkpoints/lerobot/my_task_v1/ \
  --codec=@positronic.vendors.lerobot.codecs.eepose_absolute \
  --port=8000 \
  --host=0.0.0.0
```

**Server parameters:**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--checkpoints_dir` | Experiment directory (contains `checkpoints/` folder) | Required | `~/checkpoints/lerobot/my_task_v1/` |
| `--checkpoint` | Specific checkpoint step | Latest | `10000`, `20000` |
| `--codec` | Codec (must match training) | `eepose_absolute` | `joints_absolute` |
| `--port` | Server port | `8000` | `8001` |
| `--host` | Server host | `0.0.0.0` | Binds to all interfaces |

## Troubleshooting

See vendor-specific guides and [Model Selection Guide](../../docs/model-selection.md) for issues.

## See Also

**Positronic Documentation:**
- [Model Selection Guide](../../docs/model-selection.md) — When to use LeRobot vs GR00T vs OpenPI
- [Codecs Guide](../../docs/codecs.md) — Understanding observation/action encoding
- [Training Workflow](../../docs/training-workflow.md) — Unified training steps across all models
- [Inference Guide](../../docs/inference.md) — Deployment and evaluation patterns

**Other Models:**
- [OpenPI (π₀.₅)](../openpi/README.md) — Recommended for most tasks, most capable foundation model
- [GR00T](../groot/README.md) — NVIDIA's generalist robot policy

**External:**
- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) — Official LeRobot repository
- [LeRobot Documentation](https://huggingface.co/docs/lerobot) — Training algorithms and datasets
