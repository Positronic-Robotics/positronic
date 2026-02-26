# Model Selection Guide

Positronic supports three foundation models with different capabilities and resource requirements. This guide helps you choose the right model for your task.

## Quick Recommendation

**Start with [LeRobot ACT](../positronic/vendors/lerobot_0_3_3/README.md)** if you want something quick and low-cost. Progress to [OpenPI (π₀.₅)](../positronic/vendors/openpi/README.md) or [GR00T](../positronic/vendors/groot/README.md) when you need more capable models. Positronic makes switching easy.

## Detailed Comparison

| Aspect | OpenPI (π₀.₅) | GR00T | LeRobot ACT |
|--------|---------------|-------|-------------|
| **Capability** | Most capable, generalist | Generalist | Single-task specialist |
| **Training Hardware** | capable cloud GPU (~78GB, LoRA) | capable cloud GPU (~50GB) | Consumer GPU (RTX 3090, 4090) |
| **Training Time** | Multiple days | 0.5-2 days | Several hours |
| **Inference Hardware** | GPU (~62GB, likely cloud) | GPU (~7.5GB, can run on robot) | Consumer GPU (4GB+) |
| **Inference Speed** | Moderate | Moderate | Fast |
| **Best For** | Complex multi-task manipulation, generalization | General robotics tasks | Specific manipulation tasks, fast iteration |
| **When to Use** | Need generalization, multi-task scenarios, leveraging foundation models | Prefer NVIDIA stack | Single task, resource constraints, rapid experimentation |

## Model Profiles

### OpenPI (π₀.₅) — Recommended for Most Tasks

**What it is:** Foundation model for robotics trained by Physical Intelligence on diverse manipulation tasks.

**Strengths:**
- Most capable generalization across manipulation scenarios
- Can leverage pretrained knowledge from large-scale training
- State-of-the-art performance on complex tasks

**Limitations:**
- Requires ~78GB GPU for training, ~62GB for inference (likely cloud deployment)
- Training takes multiple days


→ [OpenPI Documentation](../positronic/vendors/openpi/README.md)

### GR00T — NVIDIA's Generalist Robot Policy

**What it is:** NVIDIA's foundation model for generalist robot control.

**Strengths:**
- Generalist capabilities
- Can run on smaller GPU (~7.5GB inference, can run closer to robot)
- Requires ~50GB for training (less than OpenPI)
- Training takes 1-2 days (faster than OpenPI)

**Limitations:**
- Requires capable GPU for training
- Slower than single-task models


→ [GR00T Documentation](../positronic/vendors/gr00t/README.md)

### LeRobot ACT — Single-Task Transformer

**What it is:** Action Chunking Transformer from HuggingFace LeRobot, designed for efficient single-task imitation learning. Can be multi-task with sufficient data and task conditioning.

**Strengths:**
- Fast training (several hours on consumer GPUs)
- Efficient inference on consumer hardware
- Easy and quick to get started
- Rapid iteration cycles
- LeRobot ecosystem, cheaper models, bigger community

**Limitations:**
- Single-task focus (multi-task possible but not primary strength)
- Less generalization than foundation models


→ [LeRobot ACT Documentation](../positronic/vendors/lerobot_0_3_3/README.md)

## Implementation Notes

Positronic relies on our forks of [openpi](https://github.com/Positronic-Robotics/openpi) and [gr00t](https://github.com/Positronic-Robotics/gr00t). We do our best to keep them up to date with upstream repositories.

## Why Positronic Supports Multiple Models

Positronic's goal is to **democratize ML/AI in robotics**. You shouldn't be locked to a single vendor or architecture.

### Plug-and-Play Structure

- **Same dataset format** — Record once, train on any model
- **Same inference API** — Swap models without changing hardware code
- **Easy experimentation** — Try all models with your data, pick what works best
- **Future-proof** — We'll keep adding foundation models as they emerge

### Practical Workflow

1. **Start with your data** — Collect demonstrations using Positronic
2. **Experiment freely** — Try LeRobot ACT for quick baseline, then GR00T or OpenPI
3. **Compare results** — Use the same dataset and inference code across models
4. **Deploy the winner** — Pick the model that balances performance and resources for your use case

## Common Questions

### Do I need to re-record data for different models?

No! Positronic's dataset library stores data in a format-agnostic way. Record once, then use [codecs](codecs.md) to project your data to different model formats. You can:
- Train LeRobot ACT for fast baseline
- Train GR00T for comparison
- Train OpenPI for best performance

All from the same raw dataset using different codecs.

### What if I'm not sure which model to use?

Start with **LeRobot ACT**:
- Fastest to train and iterate
- Lowest resource requirements
- Validates your dataset and task setup

Then progress to **GR00T** or **OpenPI** if you need:
- Multi-task generalization
- Better performance on complex scenarios
- More capable models

### Can I use models not listed here?

Positronic's architecture is extensible. We'll continue adding foundation models as they emerge. You can also implement custom model integrations following our vendor patterns.

## Next Steps

1. **Choose your model** using the decision tree above
2. **Review the model-specific README** for detailed workflow
3. **Check the [Codecs Guide](codecs.md)** to understand observation/action encoding
4. **Follow the [Training Workflow](training-workflow.md)** for end-to-end steps

## See Also

- [OpenPI Documentation](../positronic/vendors/openpi/README.md)
- [GR00T Documentation](../positronic/vendors/gr00t/README.md)
- [LeRobot ACT Documentation](../positronic/vendors/lerobot_0_3_3/README.md)
- [Codecs Guide](codecs.md) — Understanding observation encoding and action decoding
- [Training Workflow](training-workflow.md) — Unified training steps across all models
