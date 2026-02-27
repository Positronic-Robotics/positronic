# Codec Guide

A **codec** transforms raw robot data to model-specific formats. Each codec implements `encode()` (raw observations → model input) and `decode()` (model output → robot commands), used at both training and inference time.

Codecs compose via two operators:

- `|` (sequential): left's output feeds into right. Use for codecs that modify data before others see it (e.g. grip binarization before observation/action encoders).
- `&` (parallel): both see the same input, outputs merged. Use for independent codecs (e.g. observation encoder & action decoder).

The standard layout is `ActionTiming | BinarizeGrip | obs & action`.

Codecs enable **store once, use everywhere** – record demonstrations once, then project the same raw data to different model formats (LeRobot, GR00T, OpenPI) without re-recording.

## Why Codecs Matter

Traditional workflows lock you into a single format. Different models expect different state/action spaces (joint vs EE, absolute vs delta, quaternion vs rot6d). This forces re-recording datasets for each model and throws away data when switching formats.

Positronic uses codecs to project raw data to any model format. Record once, try different state representations (joint space vs EE space, with/without joint feedback) and action formats (absolute vs delta) on identical raw data. See [Dataset Library](../positronic/dataset/README.md) for details on transforms and lazy evaluation.

## Target vs Trajectory Codecs

By default, codecs use **commanded** targets (`robot_commands.pose`, `target_grip`) as action labels — "what the controller was told to do". The `_traj` variants use the **actual** robot trajectory (`robot_state.ee_pose`, `grip`) — "what the robot actually did". This lets you compare training on commanded vs observed actions using identical raw data.

Trajectory codecs automatically binarize grip signals (threshold at 0.5) since observed grip values are continuous but the model should learn discrete open/close.

## Available Codecs by Vendor

### LeRobot Codecs (0.4.x — SmolVLA)

See [`positronic/vendors/lerobot/codecs.py`](../positronic/vendors/lerobot/codecs.py) for implementation.

| Codec | Observation | Action |
|-------|-------------|--------|
| `ee` | EE pose (7D quat) + grip + images (512x512) | Absolute EE position (7D quat) + grip |
| `joints` | Joint positions (7D) + grip + images (512x512) | Absolute EE position (7D quat) + grip |

```bash
cd docker && docker compose run --rm lerobot-convert convert \
  --dataset.codec=@positronic.vendors.lerobot.codecs.ee \
  --output_dir=~/datasets/lerobot/my_task
```

### LeRobot Codecs (0.3.3 — ACT)

See [`positronic/vendors/lerobot_0_3_3/codecs.py`](../positronic/vendors/lerobot_0_3_3/codecs.py) for implementation.

| Codec | Observation | Action |
|-------|-------------|--------|
| `ee` | EE pose (7D quat) + grip + images (224x224) | Absolute EE position (7D quat) + grip |
| `joints` | Joint positions (7D) + grip + images | Absolute EE position (7D quat) + grip |
| `ee_traj` | EE pose (7D quat) + grip + images (224x224) | Absolute EE trajectory (7D quat) + grip (binarized) |
| `joints_traj` | Joint positions (7D) + grip + images | Absolute joint trajectory (7D) + grip (binarized) |

```bash
cd docker && docker compose run --rm lerobot-0_3_3-convert convert \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
  --output_dir=~/datasets/lerobot/my_task
```

### GR00T Codecs

See [`positronic/vendors/gr00t/codecs.py`](../positronic/vendors/gr00t/codecs.py) for implementation.

| Codec | Observation | Action | Modality Configs |
|-------|-------------|--------|------------------|
| `ee_quat` | EE pose (quat) + grip + images (224x224) | Absolute EE position (quat) + grip | `ee`, `ee_rel` |
| `ee_rot6d` | EE pose (rot6d) + grip + images | Absolute EE position (rot6d) + grip | `ee_rot6d`, `ee_rot6d_rel` |
| `ee_quat_joints` | EE pose + joints + grip + images | Absolute EE position + grip | `ee_q`, `ee_q_rel` |
| `ee_rot6d_joints` | EE pose (rot6d) + joints + grip + images | Absolute EE position (rot6d) + grip | `ee_rot6d_q`, `ee_rot6d_q_rel` |
| `ee_quat_traj` | EE pose (quat) + grip + images | Absolute EE trajectory (quat) + grip (binarized) | `ee`, `ee_rel` |
| `ee_rot6d_traj` | EE pose (rot6d) + grip + images | Absolute EE trajectory (rot6d) + grip (binarized) | `ee_rot6d`, `ee_rot6d_rel` |
| `ee_quat_joints_traj` | EE pose + joints + grip + images | Absolute EE trajectory + grip (binarized) | `ee_q`, `ee_q_rel` |
| `ee_rot6d_joints_traj` | EE pose (rot6d) + joints + grip + images | Absolute EE trajectory (rot6d) + grip (binarized) | `ee_rot6d_q`, `ee_rot6d_q_rel` |
| `joints_traj` | Joints + grip + images (no EE pose) | Absolute joint trajectory + grip (binarized) | — |

Codec must match modality config during training.

```bash
# Convert with codec
cd docker && docker compose run --rm lerobot-convert convert \
  --dataset.codec=@positronic.vendors.gr00t.codecs.ee_rot6d_joints \
  --output_dir=~/datasets/groot/my_task

# Train with matching modality
cd docker && docker compose run --rm groot-train \
  --modality_config=ee_rot6d_q \
  --input_path=~/datasets/groot/my_task
```

### OpenPI Codecs

See [`positronic/vendors/openpi/codecs.py`](../positronic/vendors/openpi/codecs.py) for implementation.

| Codec | Observation | Action |
|-------|-------------|--------|
| `ee` | EE pose (7D quat) + grip + images (224x224) | Absolute EE position (7D) |
| `ee_joints` | EE pose + joints (7D) + grip + images | Absolute EE position (7D) |
| `ee_traj` | EE pose (7D quat) + grip + images (224x224) | Absolute EE trajectory (7D) + grip (binarized) |
| `ee_joints_traj` | EE pose + joints (7D) + grip + images | Absolute EE trajectory (7D) + grip (binarized) |
| `joints_traj` | Joints (7D) + grip + images | Absolute joint trajectory (7D) + grip (binarized) |
| `droid` | Joint positions (7D) + grip + images | Joint delta (velocity) |

`droid` codec is inference-only for use with pretrained DROID models (not for training).

```bash
cd docker && docker compose run --rm lerobot-convert convert \
  --dataset.codec=@positronic.vendors.openpi.codecs.ee \
  --output_dir=~/datasets/openpi/my_task
```

## Choosing a Codec

Positronic aims to represent the same action and state space across different models when possible. The primary choice is observation space: **end-effector pose only** vs **end-effector + joint positions**. Joint feedback can improve learning but isn't always necessary. Different action spaces (absolute position, joint delta) are supported but vary by vendor – check vendor codec docs for specifics.

## Codec Mismatch Troubleshooting

**Problem:** "Shape mismatch" or "Feature mismatch" during inference.

**Cause:** Codec used for inference doesn't match training codec.

**Solution:** Verify training and inference use identical codec.

```bash
# Training (SmolVLA)
cd docker && docker compose run --rm lerobot-convert convert \
  --dataset.codec=@positronic.vendors.lerobot.codecs.ee

# Inference must match
cd docker && docker compose run --rm lerobot-server \
  --codec=@positronic.vendors.lerobot.codecs.ee
```

## Writing Custom Codecs

Subclass `positronic.policy.codec.Codec` and implement `encode()` and/or `_decode_single()`. The base class returns `{}` from both — observation codecs override `encode()`, action codecs override `_decode_single()`. Middleware codecs that pass data through (like `BinarizeGrip`) must explicitly `return data`. Compose observation and action codecs with `&`, chain middleware with `|`. See existing implementations in vendor codec files for reference patterns.

## See Also

- [Dataset Library README](../positronic/dataset/README.md) – Raw storage and transforms
- [Training Workflow](training-workflow.md) – Using codecs in pipeline
- [Model Selection](model-selection.md) – Choosing models
- Vendor docs: [SmolVLA](../positronic/vendors/lerobot/README.md) | [LeRobot ACT](../positronic/vendors/lerobot_0_3_3/README.md) | [GR00T](../positronic/vendors/gr00t/README.md) | [OpenPI](../positronic/vendors/openpi/README.md)
