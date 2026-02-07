# Codec Guide

A **codec** (coder-decoder) is a pair of classes that transforms raw robot data to model-specific formats:

- **Observation encoder**: Raw robot state → model input (training and inference)
- **Action decoder**: Model output → robot commands (training and inference)

Codecs enable **store once, use everywhere** – record demonstrations once, then project the same raw data to different model formats (LeRobot, GR00T, OpenPI) without re-recording.

## Why Codecs Matter

Traditional workflows lock you into a single format. Different models expect different state/action spaces (joint vs EE, absolute vs delta, quaternion vs rot6d). This forces re-recording datasets for each model and throws away data when switching formats.

Positronic uses codecs to project raw data to any model format. Record once, try different state representations (joint space vs EE space, with/without joint feedback) and action formats (absolute vs delta) on identical raw data. See [Dataset Library](../positronic/dataset/README.md) for details on transforms and lazy evaluation.

## Target vs Trajectory Codecs

By default, codecs use **commanded** targets (`robot_commands.pose`, `target_grip`) as action labels — "what the controller was told to do". The `_traj` variants use the **actual** robot trajectory (`robot_state.ee_pose`, `grip`) — "what the robot actually did". This lets you compare training on commanded vs observed actions using identical raw data.

## Available Codecs by Vendor

### LeRobot Codecs

See [`positronic/vendors/lerobot/codecs.py`](../positronic/vendors/lerobot/codecs.py) for implementation.

| Codec | Observation | Action |
|-------|-------------|--------|
| `eepose_absolute` | EE pose (7D quat) + grip + images (480x480) | Absolute EE position (7D quat) + grip |
| `joints_absolute` | Joint positions (7D) + grip + images | Absolute EE position (7D quat) + grip |
| `eepose_absolute_traj` | EE pose (7D quat) + grip + images (480x480) | Absolute EE trajectory (7D quat) + grip |
| `joints_absolute_traj` | Joint positions (7D) + grip + images | Absolute EE trajectory (7D quat) + grip |

```bash
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.codec=@positronic.vendors.lerobot.codecs.eepose_absolute \
  --output_dir=~/datasets/lerobot/my_task
```

### GR00T Codecs

See [`positronic/vendors/gr00t/codecs.py`](../positronic/vendors/gr00t/codecs.py) for implementation.

| Codec | Observation | Action | Modality Configs |
|-------|-------------|--------|------------------|
| `ee_absolute` | EE pose (quat) + grip + images (224x224) | Absolute EE position (quat) + grip | `ee`, `ee_rel` |
| `ee_rot6d` | EE pose (rot6d) + grip + images | Absolute EE position (rot6d) + grip | `ee_rot6d`, `ee_rot6d_rel` |
| `ee_joints` | EE pose + joints + grip + images | Absolute EE position + grip | `ee_q`, `ee_q_rel` |
| `ee_rot6d_joints` | EE pose (rot6d) + joints + grip + images | Absolute EE position (rot6d) + grip | `ee_rot6d_q`, `ee_rot6d_q_rel` |
| `ee_absolute_traj` | EE pose (quat) + grip + images | Absolute EE trajectory (quat) + grip | `ee`, `ee_rel` |
| `ee_rot6d_traj` | EE pose (rot6d) + grip + images | Absolute EE trajectory (rot6d) + grip | `ee_rot6d`, `ee_rot6d_rel` |
| `ee_joints_traj` | EE pose + joints + grip + images | Absolute EE trajectory + grip | `ee_q`, `ee_q_rel` |
| `ee_rot6d_joints_traj` | EE pose (rot6d) + joints + grip + images | Absolute EE trajectory (rot6d) + grip | `ee_rot6d_q`, `ee_rot6d_q_rel` |

Some codecs support both absolute and relative modality configs (e.g., `ee_absolute` works with `ee` or `ee_rel`). Codec must match modality config during training.

```bash
# Convert with codec
cd docker && docker compose run --rm positronic-to-lerobot convert \
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
| `eepose` | EE pose (7D quat) + grip + images (224x224) | Absolute EE position (7D) |
| `eepose_q` | EE pose + joints (7D) + grip + images | Absolute EE position (7D) |
| `eepose_traj` | EE pose (7D quat) + grip + images (224x224) | Absolute EE trajectory (7D) |
| `eepose_q_traj` | EE pose + joints (7D) + grip + images | Absolute EE trajectory (7D) |
| `droid` | Joint positions (7D) + grip + images | Joint delta (velocity) |

`droid` codec is inference-only for use with pretrained DROID models (not for training).

```bash
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.codec=@positronic.vendors.openpi.codecs.eepose \
  --output_dir=~/datasets/openpi/my_task
```

## Choosing a Codec

Positronic aims to represent the same action and state space across different models when possible. The primary choice is observation space: **end-effector pose only** vs **end-effector + joint positions**. Joint feedback can improve learning but isn't always necessary. Different action spaces (absolute position, joint delta) are supported but vary by vendor – check vendor codec docs for specifics.

## Codec Mismatch Troubleshooting

**Problem:** "Shape mismatch" or "Feature mismatch" during inference.

**Cause:** Codec used for inference doesn't match training codec.

**Solution:** Verify training and inference use identical codec.

```bash
# Training
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.codec=@positronic.vendors.lerobot.codecs.eepose_absolute

# Inference must match
cd docker && docker compose run --rm lerobot-server \
  --codec=@positronic.vendors.lerobot.codecs.eepose_absolute
```

## Writing Custom Codecs

For custom robot platforms or action spaces, see existing implementations in vendor codec files for reference patterns. API details in [Dataset Library documentation](../positronic/dataset/README.md).

## See Also

- [Dataset Library README](../positronic/dataset/README.md) – Raw storage and transforms
- [Training Workflow](training-workflow.md) – Using codecs in pipeline
- [Model Selection](model-selection.md) – Choosing models
- Vendor docs: [LeRobot](../positronic/vendors/lerobot/README.md) | [GR00T](../positronic/vendors/gr00t/README.md) | [OpenPI](../positronic/vendors/openpi/README.md)
