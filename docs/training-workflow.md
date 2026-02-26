# Unified Training Workflow

All vendors (LeRobot, GR00T, OpenPI) follow the same 4-step workflow in Positronic. This guide covers the common patterns — see vendor-specific READMEs for model-specific details.

## Overview

**Workflow steps:**

1. Prepare Data: Convert Positronic dataset to model format using codec
2. Train Model: Run training job with Docker service
3. Serve Inference: Start inference server with trained checkpoint
4. Run Inference: Connect hardware/simulator to server with .remote policy

## Step 1: Prepare Data

Convert your Positronic dataset into the model's expected format using a codec.

### Basic Conversion

```bash
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.dataset=.local \
  --dataset.dataset.path=~/datasets/stack_cubes_raw \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
  --output_dir=~/datasets/lerobot/stack_cubes
```

### Parameters Explained

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--dataset.dataset` | Dataset configuration | `.local` for local directories, `@positronic.cfg.ds.phail.phail` for public datasets |
| `--dataset.dataset.path` | Path to raw Positronic dataset (only for `.local`, not needed for phail datasets) | `~/datasets/stack_cubes_raw` |
| `--dataset.codec` | Codec for observation/action encoding | `@positronic.vendors.lerobot_0_3_3.codecs.ee` |
| `--output_dir` | Destination for converted dataset | `~/datasets/lerobot/stack_cubes` or `s3://bucket/path` |
| `--task` | (Optional) Task description to embed | `"pick up the green cube"` |

**Note:** `--dataset.dataset.path` is only required when providing your own local datasets (with `--dataset.dataset=.local`). For available phail datasets (like `@positronic.cfg.ds.phail.sim_stack_cubes`), this parameter is not needed.

### Available Public Datasets

| Dataset | Description | Size | Episodes |
|---------|-------------|------|----------|
| `@positronic.cfg.ds.phail.phail` | DROID teleoperation data | 12GB | 352 |
| `@positronic.cfg.ds.phail.sim_stack_cubes` | Simulated cube stacking | 499MB | 317 |
| `@positronic.cfg.ds.phail.sim_pick_place` | Simulated pick-and-place | 1.3GB | 214 |

**Example:**
```bash
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
  --output_dir=~/datasets/lerobot/sim_stack_cubes
```

### Choosing a Codec

See the [Codecs Guide](codecs.md) for detailed codec documentation.

**Quick reference:**

| Model | Common Codecs |
|-------|---------------|
| **LeRobot ACT** | `ee`, `joints`, `ee_traj`, `joints_traj` |
| **GR00T** | `ee_rot6d_joints`, `ee_quat`, `ee_quat_joints` |
| **OpenPI** | `ee`, `ee_joints`, `droid` |

### S3 Support

Both input and output paths support S3 URLs. Data is cached locally and synced automatically. Positronic relies on [pos3](https://github.com/Positronic-Robotics/pos3) for S3 integration.

```bash
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.dataset.path=s3://bucket/raw_data/stack_cubes \
  --output_dir=s3://bucket/converted/lerobot/stack_cubes \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee
```

### Appending to Existing Datasets

To add more data to an existing LeRobot dataset:

```bash
cd docker && docker compose run --rm positronic-to-lerobot append \
  --output_dir=~/datasets/lerobot/stack_cubes \
  --dataset.dataset=.local \
  --dataset.dataset.path=~/datasets/stack_cubes_new \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee
```

**Important:** Codec must match the original dataset's codec.

## Step 2: Train Model

Run the training job using vendor-specific Docker services. All services are defined in [`docker/docker-compose.yml`](../docker/docker-compose.yml).

### LeRobot Training

```bash
cd docker && docker compose run --rm lerobot-train \
  --input_path=~/datasets/lerobot/stack_cubes \
  --exp_name=experiment_v1 \
  --output_dir=~/checkpoints/lerobot/ \
  --num_train_steps=50000 \
  --save_freq=10000
```

### GR00T Training

```bash
cd docker && docker compose run --rm groot-train \
  --input_path=~/datasets/groot/stack_cubes \
  --output_path=~/checkpoints/groot \
  --exp_name=experiment_v1 \
  --modality_config=ee_rot6d_q
```

**Modality config must match codec** (see [GR00T README](../positronic/vendors/gr00t/README.md#1-prepare-data)).

### OpenPI Training

**Generate assets first** (required for OpenPI):

```bash
cd docker && docker compose run --rm openpi-stats \
  --input_path=~/datasets/openpi/stack_cubes \
  --output_path=~/datasets/openpi_assets
```

**Then train:**

```bash
cd docker && docker compose run --rm openpi-train \
  --input_path=~/datasets/openpi/stack_cubes \
  --stats_path=~/datasets/openpi_assets/assets/ \
  --output_path=~/checkpoints/openpi \
  --exp_name=experiment_v1 \
  --config_name=pi05_positronic_lowmem
```

### Common Training Parameters

These parameters work across all vendors:

| Concept | Parameter | Description |
|---------|-----------|-------------|
| **Experiment name** | `--exp_name` | Unique identifier for this run |
| **Training steps** | `--num_train_steps` | Total training iterations |
| **Checkpoint frequency** | `--save_freq` or `--save_steps` | How often to save checkpoints |
| **Resume training** | `--resume=True` | Continue from existing checkpoint |
| **Learning rate** | `--learning_rate` | Optimizer learning rate |

### WandB Integration

WandB logging is enabled by default if `WANDB_API_KEY` is set in `docker/.env.wandb`.

## Step 3: Serve Inference

Start an inference server that exposes a unified WebSocket API. All vendors implement the same Protocol v1 (see [Offboard README](../positronic/offboard/README.md) for details), enabling a single `.remote` policy client that works across all models.

### Starting Servers

> **Note:** Use `--service-ports` flag to expose ports from Docker

**LeRobot Server:**

```bash
cd docker && docker compose run --rm --service-ports lerobot-server \
  --checkpoints_dir=~/checkpoints/lerobot/experiment_v1/ \
  --codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
  --port=8000
```

**GR00T Server:**

```bash
cd docker && docker compose run --rm --service-ports groot-server \
  ee_rot6d_joints \
  --checkpoints_dir=~/checkpoints/groot/experiment_v1/
```

**OpenPI Server:**

```bash
cd docker && docker compose run --rm --service-ports openpi-server \
  --checkpoints_dir=~/checkpoints/openpi/pi05_positronic_lowmem/experiment_v1/ \
  --codec=@positronic.vendors.openpi.codecs.ee
```

### Server Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--checkpoints_dir` | Path to experiment directory (contains checkpoint folders) | `~/checkpoints/lerobot/experiment_v1/` |
| `--checkpoint` | (Optional) Specific checkpoint ID to load | `10000`, `20000` |
| `--codec` | Codec (must match training) | `@positronic.vendors.lerobot_0_3_3.codecs.ee` |
| `--port` | Server port | `8000` (default) |
| `--host` | Server host | `0.0.0.0` (default, binds to all interfaces) |

### Checking Server Status

```bash
# List available checkpoints
curl http://localhost:8000/api/v1/models

# Example response:
# {"models": ["10000", "20000", "30000"]}
```

### Long Model Loading

GR00T and OpenPI servers can take 120-300s to load on first startup (model download + weight loading). The server sends periodic status updates to prevent WebSocket keepalive timeouts.

## Step 4: Run Inference

See [Inference Guide](inference.md) for detailed deployment and evaluation patterns.

## Common Workflows

### Experiment Workflow

```bash
# 1. Collect demonstrations
uv run positronic-data-collection sim --output_dir=~/datasets/my_task

# 2. Review in viewer
uv run positronic-server --dataset.path=~/datasets/my_task

# 3. Bake dataset into LeRobot format
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.dataset.path=~/datasets/my_task \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
  --output_dir=~/datasets/lerobot/my_task

# 4. Train ACT policy
cd docker && docker compose run --rm lerobot-train \
  --input_path=~/datasets/lerobot/my_task \
  --exp_name=baseline_v1 \
  --output_dir=~/checkpoints/lerobot/ \
  --num_train_steps=50000

# 5. Evaluate
cd docker && docker compose run --rm --service-ports lerobot-server \
  --checkpoints_dir=~/checkpoints/lerobot/baseline_v1/ \
  --codec=@positronic.vendors.lerobot_0_3_3.codecs.ee &

uv run positronic-inference sim \
  --policy=.remote \
  --output_dir=~/datasets/inference_logs/baseline_v1
```

### Multi-Model Comparison

```bash
# Convert same dataset for all models
cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.dataset.path=~/datasets/my_task \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
  --output_dir=~/datasets/lerobot/my_task

cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.dataset.path=~/datasets/my_task \
  --dataset.codec=@positronic.vendors.gr00t.codecs.ee_rot6d_joints \
  --output_dir=~/datasets/groot/my_task

cd docker && docker compose run --rm positronic-to-lerobot convert \
  --dataset.dataset.path=~/datasets/my_task \
  --dataset.codec=@positronic.vendors.openpi.codecs.ee \
  --output_dir=~/datasets/openpi/my_task

# Train all models (can run in parallel)
cd docker && docker compose run --rm lerobot-train --input_path=~/datasets/lerobot/my_task ...
cd docker && docker compose run --rm groot-train --input_path=~/datasets/groot/my_task ...
cd docker && docker compose run --rm openpi-train --input_path=~/datasets/openpi/my_task ...

# Evaluate all models using same .remote policy
# (Just swap server, client code unchanged!)
```

## See Also

- **Model-specific guides:**
  - [OpenPI Workflow](../positronic/vendors/openpi/README.md)
  - [GR00T Workflow](../positronic/vendors/gr00t/README.md)
  - [LeRobot Workflow](../positronic/vendors/lerobot_0_3_3/README.md)

- **Related documentation:**
  - [Codecs Guide](codecs.md) — Understanding observation encoding and action decoding
  - [Model Selection](model-selection.md) — Choosing the right model
  - [Inference Guide](inference.md) — Deployment and evaluation patterns
