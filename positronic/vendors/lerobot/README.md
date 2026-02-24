# LeRobot ACT in Positronic

## What is LeRobot ACT?

LeRobot ACT (Action Chunking Transformer) is a single-task imitation learning model from [HuggingFace LeRobot](https://github.com/huggingface/lerobot). Based on the [ACT paper](https://arxiv.org/abs/2304.13705) (foundational work in recent ML robotics), it's designed for efficient learning on focused manipulation tasks, offering excellent performance with modest hardware requirements and fast training times.

ACT uses action chunking to output sequences of future actions, enabling smooth execution and reducing compounding errors. This makes it particularly effective for precise manipulation tasks where consistency and repeatability are critical.

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
  --dataset.codec=@positronic.vendors.lerobot.codecs.ee \
  --output_dir=~/datasets/lerobot/my_task

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
  --codec=@positronic.vendors.lerobot.codecs.ee

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
| `ee` | EE pose (7D quat) + grip (1D) + images | Absolute EE position (7D quat) + grip | Default codec for end-effector control, task-space manipulation |
| `joints` | Joint positions (7D) + grip (1D) + images | Absolute EE position (7D quat) + grip | Joint-space observations with task-space control |
| `ee_traj` | EE pose (7D quat) + grip + images | Absolute EE trajectory (7D quat) + grip (binarized) | Training on actual robot trajectory |
| `joints_traj` | Joint positions (7D) + grip + images | Absolute joint trajectory (7D) + grip (binarized) | Joint-space trajectory training |

**Key features:**
- Uses `task_field='task'` (LerobotPolicy filters this before passing to ACT)
- Images resized to 480x480
- Quaternion rotation representation (7D)
- Absolute action space (not delta)

**Choosing a codec:**
- **Most tasks**: Use `ee` (task-space observations and control)
- **Want joint feedback**: Use `joints` (may improve performance with joint position information)
- **Trajectory training**: Use `ee_traj` or `joints_traj` (trains on actual robot trajectory with binarized grip)

See [Codecs Guide](../../docs/codecs.md) for comprehensive codec documentation.

## Configuration Reference

### Training Configuration

**Common parameters:**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--codec` | Override codec | `ee` | `joints` |
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
  --codec=@positronic.vendors.lerobot.codecs.ee \
  --port=8000 \
  --host=0.0.0.0
```

**Server parameters:**

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--checkpoints_dir` | Experiment directory (contains `checkpoints/` folder) | Required | `~/checkpoints/lerobot/my_task_v1/` |
| `--checkpoint` | Specific checkpoint step | Latest | `10000`, `20000` |
| `--codec` | Codec (must match training) | `ee` | `joints` |
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
