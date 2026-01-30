# Codec Guide — Observation Encoding and Action Decoding

## What is a Codec?

A **codec** (coder-decoder) in Positronic is a pair of classes that bridges raw robot data and model-specific formats:

- **Observation encoder**: Transforms raw robot state → model input format (training and inference)
- **Action decoder**: Transforms model output → robot commands (training and inference)

Codecs need transforms for both training time (converting datasets) AND inference time (live robot control). Positronic's codec system enables **store once, use everywhere**.

## Why Codecs Matter

Traditional robotics workflows suffer from format lock-in. Different models expect different state and action spaces (joint space vs EE space, absolute vs delta actions, quaternion vs rot6d rotations). This forces you to:

1. Re-record datasets for each model
2. Throw away data when switching formats
3. Lock into a single vendor/model architecture

```
❌ Traditional Workflow:
Raw data → LeRobot format → Train LeRobot model
Raw data → GR00T format → Train GR00T model (need to re-record!)
Raw data → OpenPI format → Train OpenPI model (need to re-record again!)
```

Positronic uses codecs to project raw data to any format:

```
✅ Positronic Workflow:
Raw data (immutable)
  ├─ Codec A → LeRobot format → Train LeRobot model
  ├─ Codec B → GR00T format → Train GR00T model
  └─ Codec C → OpenPI format → Train OpenPI model
```

**Benefits:**
- Record demonstrations once
- Try different state representations (joint space vs EE space) without re-recording
- Experiment with action formats (absolute vs delta) on the same data
- Compare models using identical raw data
- Future-proof your dataset for new models

## How Codecs Work

### Raw Data Storage

Positronic stores robot data in a format-agnostic way:

```python
# Example raw signals
'robot_state.ee_pose'       # [x, y, z, qx, qy, qz, qw] (7D)
'robot_state.joint_position' # [j1, j2, ..., j7] (7D for Franka)
'grip'                      # [width] (1D)
'robot_commands.pose'       # [x, y, z, qx, qy, qz, qw] (7D)
'target_grip'               # [target_width] (1D)
'image.wrist'               # RGB image (H, W, 3)
'image.exterior'            # RGB image (H, W, 3)
```

### Codec Projection

Each codec defines how to transform raw signals:

```python
# LeRobot ACT codec (eepose_absolute)
observation = {
    'ee_pose': robot_state.ee_pose,           # 7D quaternion
    'grip': grip,                             # 1D
    'wrist_image': resize(image.wrist, 480),  # Resized RGB
    'exterior_image': resize(image.exterior, 480)
}
action = {
    'target_pose': robot_commands.pose,       # Absolute position (7D)
    'target_grip': target_grip                # 1D
}

# GR00T codec (ee_rot6d_joints)
observation = {
    'ee_pose': quat_to_rot6d(robot_state.ee_pose[:3], robot_state.ee_pose[3:]),  # 6D rotation
    'joint_position': robot_state.joint_position,  # 7D joints
    'grip': grip,
    'wrist_image': resize(image.wrist, 224),
    'exterior_image_1': resize(image.exterior, 224)
}
action = {
    'target_pose': quat_to_rot6d(...),
    'target_joints': joint_commands,
    'target_grip': target_grip
}
```

### Lazy Evaluation

Codecs use **lazy transforms** — computation happens only when data is accessed:

```python
# No computation yet
dataset_with_codec = apply_codec(raw_dataset, codec=eepose_absolute)

# Computation happens here (on-demand)
episode = dataset_with_codec[0]
observation = episode['ee_pose']  # Lazily computed from raw data
```

**Benefits:**
- Zero-copy views of raw data
- Compose multiple transforms without materializing intermediates
- Change codecs without re-processing datasets

## Available Codecs by Vendor

### LeRobot Codecs

Located in `positronic/vendors/lerobot/codecs.py`

| Codec | Observation | Action | Use Case |
|-------|-------------|--------|----------|
| `eepose_absolute` | EE pose (7D quat) + grip (1D) + images | Absolute EE position (7D quat) + grip | Default codec for end-effector control |
| `joints_absolute` | Joint positions (7D) + grip (1D) + images | Absolute EE position (7D quat) + grip | Joint-space observations with task-space control |

**Key features:**
- Uses `task_field='task'` (LeRobotPolicy filters this before passing to ACT)
- Images resized to 480x480
- Quaternion rotation representation (7D)
- Absolute action space (not delta)

**Example:**
```bash
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.codec=@positronic.vendors.lerobot.codecs.eepose_absolute \
  --output_dir=~/datasets/lerobot/my_task
```

### GR00T Codecs

Located in `positronic/vendors/gr00t/codecs.py`

| Codec | Observation | Action | Modality Config | Use Case |
|-------|-------------|--------|-----------------|----------|
| `ee_absolute` | EE pose (quat) + grip + images | Absolute EE position (quat) + grip | `ee` | Default EE control |
| `ee_rot6d` | EE pose (rot6d) + grip + images | Absolute EE position (rot6d) + grip | `ee_rot6d` | 6D rotation representation |
| `ee_joints` | EE pose + joints + grip + images | Absolute EE position + grip | `ee_q` | Combined EE + joint feedback |
| `ee_rot6d_joints` | EE pose (rot6d) + joints + grip + images | Absolute EE position (rot6d) + grip | `ee_rot6d_q` | 6D rotation + joint feedback |

**Key features:**
- Multiple rotation representations (quaternion, rot6d)
- Optional joint position observations
- Images resized to 224x224
- Sets `gr00t_modality` metadata for training
- Dual-purpose encoder (training + inference)

**Codec must match modality config during training:**

| Codec | Training Modality |
|-------|-------------------|
| `ee_absolute` | `ee` |
| `ee_rot6d` | `ee_rot6d` |
| `ee_joints` | `ee_q` |
| `ee_rot6d_joints` | `ee_rot6d_q` |

**Example:**
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

Located in `positronic/vendors/openpi/codecs.py`

| Codec | Observation | Action | Use Case |
|-------|-------------|--------|----------|
| `eepose_absolute` | EE pose (7D quat) + grip + images | Absolute EE position (7D) | Default for Positronic datasets |
| `openpi_positronic` | EE pose (OpenPI format) + grip + images | Absolute EE position (7D) | OpenPI-native format |
| `droid` | Joint positions + grip + images | Joint delta (velocity) | DROID dataset compatibility (inference only, training not supported) |
| `eepose_q` | EE pose + joints + grip + images | Absolute EE position (7D) | Combined feedback |
| `joints` | Joint positions + grip + images | Absolute EE position (7D) | Joint observations, task-space control |

**Key differences:**
- **`eepose_absolute` vs `openpi_positronic`**: Same semantics, different key format (`observation.state` vs `observation/state`)
- **`droid`**: Uses joint delta actions (velocity control) instead of absolute position — critical for DROID dataset
- **`eepose_q`**: Includes both EE pose and joint positions for richer feedback
- Uses `task_field='prompt'` instead of `task`

**Example:**
```bash
# Default codec (EE pose → absolute position)
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.codec=@positronic.vendors.openpi.codecs.eepose_absolute \
  --output_dir=~/datasets/openpi/my_task

# DROID dataset (joints → joint delta)
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.codec=@positronic.vendors.openpi.codecs.droid \
  --output_dir=~/datasets/openpi/droid_task
```

## Choosing a Codec

### Decision Guide

**1. What state representation do you want for observations?**

- **End-effector pose** → `eepose_*` codecs
- **Joint positions** → `joints` or `droid` codecs
- **Both (richer feedback)** → `eepose_q` or `ee_joints` codecs

**2. What rotation representation?**

- **Quaternion (7D)** → Default for most codecs
- **Rot6d (6D)** → GR00T `ee_rot6d` codecs (continuous representation, better for learning)

**3. What action space?**

- **Absolute position** → Most codecs (task-space control)
- **Joint delta (velocity)** → OpenPI `droid` codec (DROID compatibility)

**4. What model are you using?**

- **LeRobot ACT** → LeRobot codecs (`eepose_absolute`, `joints_absolute`)
- **GR00T** → GR00T codecs (match modality config)
- **OpenPI** → OpenPI codecs (use `droid` for DROID data, `eepose_absolute` for Positronic data)

### Common Patterns

**For simulated data (MuJoCo):**
- Start with `eepose_absolute` (task-space observations and control)
- Try `eepose_q` if you want joint feedback (can improve performance)

**For DROID hardware data:**
- Use OpenPI `droid` codec for inference (inference only, we don't support training with it yet)

**For Franka/Kinova data:**
- Start with `eepose_absolute` (task-space control)
- Experiment with `joints` codecs if joint-space control is preferred

**For GR00T:**
- Use `ee_rot6d_joints` for best performance (6D rotation + joint feedback)
- Match codec to modality config during training

## Codec Mismatch Troubleshooting

### Problem: "Shape mismatch" or "Feature mismatch" during inference

**Cause:** The codec used for inference doesn't match the codec used during training.

**Solution:**
1. Verify training command used same codec
2. Ensure inference server uses identical codec

**Example:**
```bash
# Training used this codec
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.codec=@positronic.vendors.lerobot.codecs.eepose_absolute

# Inference must use same codec
cd docker && docker compose run --rm lerobot-server \
  --codec=@positronic.vendors.lerobot.codecs.eepose_absolute
```

## Writing Custom Codecs

You can create custom codecs for specific robot platforms or action spaces. We recommend looking at existing codec implementations in the vendor directories for reference patterns:

- `positronic/vendors/lerobot/codecs.py`
- `positronic/vendors/gr00t/codecs.py`
- `positronic/vendors/openpi/codecs.py`

For API details, refer to the [Dataset Library documentation](../positronic/dataset/README.md).

## See Also

- [Dataset Library README](../positronic/dataset/README.md) — Raw format storage and transforms
- [Training Workflow](training-workflow.md) — Using codecs in training pipeline
- [Model Selection](model-selection.md) — Choosing the right model
- Vendor-specific codec docs: [LeRobot](../positronic/vendors/lerobot/README.md) | [GR00T](../positronic/vendors/gr00t/README.md) | [OpenPI](../positronic/vendors/openpi/README.md)
