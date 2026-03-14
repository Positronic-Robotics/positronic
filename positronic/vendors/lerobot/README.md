# SmolVLA / LeRobot 0.4.x in Positronic

## What is SmolVLA?

SmolVLA is a compact vision-language-action model from [HuggingFace LeRobot](https://github.com/huggingface/lerobot) (0.4.x). It combines a VLM backbone with action prediction for language-conditioned manipulation. This vendor also supports ACT, Diffusion, and any other lerobot 0.4.x policy — the policy type is auto-detected from the checkpoint config.

See [Model Selection Guide](../../docs/model-selection.md) for comparison with other models.

## Hardware Requirements

| Phase | Requirement | Notes |
|-------|-------------|-------|
| **Training** | Consumer GPU (RTX 3090, 4090) | 16GB+ VRAM recommended |
| **Inference** | Consumer GPU (4GB+) | RTX 3060, 4060, or similar |
| **Development** | CPU acceptable | For testing (slower inference) |

## Quick Start

```bash
# 1. Convert dataset (output_dir supports both local paths and s3://)
cd docker && docker compose run --rm lerobot-convert convert \
  --dataset.dataset.path=~/datasets/my_task_raw \
  --dataset.codec=@positronic.vendors.lerobot.codecs.ee \
  --output_dir=~/datasets/lerobot/my_task \
  --task="pick up the green cube and place it on the red cube"

# 2. Train (expert-only by default — frozen vision encoder)
cd docker && docker compose run --rm lerobot-train train \
  --input_path=~/datasets/lerobot/my_task \
  --exp_name=my_task_v1 \
  --output_dir=~/checkpoints/lerobot/ \
  --num_train_steps=50000

# Or full finetune (all parameters trainable)
cd docker && docker compose run --rm lerobot-train full_finetune \
  --input_path=~/datasets/lerobot/my_task \
  --exp_name=my_task_v1_ft \
  --output_dir=~/checkpoints/lerobot/ \
  --num_train_steps=50000

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

| Codec | Observation | Action | Use Case |
|-------|-------------|--------|----------|
| `ee` | EE pose (7D quat) + grip (1D) + images (512x512) | Absolute EE position (7D quat) + grip | Default, end-effector control |
| `joints` | Joint positions (7D) + grip (1D) + images (512x512) | Absolute EE position (7D quat) + grip | Joint-space observations |

**Key features:**
- Language-conditioned via `task` field (natural language instructions)
- Images resized to 512x512 (SmolVLA VLM backbone requirement)
- Quaternion rotation representation (7D)
- Absolute action space

See [Codecs Guide](../../docs/codecs.md) for comprehensive codec documentation.

## Configuration Reference

### Training Configuration

Two training modes are available:

| Mode | Command | Description |
|------|---------|-------------|
| `train` | `lerobot-train train` | Expert-only — frozen vision encoder (default) |
| `full_finetune` | `lerobot-train full_finetune` | All parameters trainable |

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--codec` | Override codec | `ee` | `joints` |
| `--exp_name` | Experiment name (unique ID) | Required | `my_task_v1` |
| `--base_model` | Base pretrained model | `lerobot/smolvla_base` | HuggingFace model ID |
| `--num_train_steps` | Total training steps | `100000` | `50000` |
| `--batch_size` | Batch size | `64` | `32` |
| `--resume` | Resume from existing checkpoint | `False` | `True` |
| `--output_dir` | Checkpoint destination | Required | `~/checkpoints/lerobot/` |

**WandB logging:** Enabled automatically if `WANDB_API_KEY` is set in `docker/.env.wandb`.

### Inference Server Configuration

```bash
cd docker && docker compose run --rm --service-ports lerobot-server \
  --checkpoints_dir=~/checkpoints/lerobot/my_task_v1/ \
  --codec=@positronic.vendors.lerobot.codecs.ee \
  --port=8000
```

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--checkpoints_dir` | Experiment directory (contains `checkpoints/` folder) | Required | `~/checkpoints/lerobot/my_task_v1/` |
| `--checkpoint` | Specific checkpoint step | Latest | `10000`, `20000` |
| `--codec` | Codec (must match training) | `ee` | `joints` |
| `--port` | Server port | `8000` | `8001` |
| `--host` | Server host | `0.0.0.0` | Binds to all interfaces |

## See Also

**Positronic Documentation:**
- [Model Selection Guide](../../docs/model-selection.md) — When to use SmolVLA vs ACT vs GR00T vs OpenPI
- [Codecs Guide](../../docs/codecs.md) — Understanding observation/action encoding
- [Training Workflow](../../docs/training-workflow.md) — Unified training steps across all models
- [Inference Guide](../../docs/inference.md) — Deployment and evaluation patterns

**Other Models:**
- [LeRobot ACT (0.3.3)](../lerobot_0_3_3/README.md) — Single-task transformer
- [OpenPI (pi0.5)](../openpi/README.md) — Most capable foundation model
- [GR00T](../groot/README.md) — NVIDIA's generalist robot policy

**External:**
- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) — Official LeRobot repository
- [SmolVLA Paper](https://huggingface.co/papers/2506.01844) — Vision-language-action model
