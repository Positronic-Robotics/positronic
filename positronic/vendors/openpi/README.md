# OpenPI Workflow in Positronic

This guide details the end-to-end workflow for training and deploying OpenPI models using the Positronic stack. The pipeline leverages Docker for reproducibility and supports both local directories and S3 for data storage. This integration relies on [our fork of OpenPI](https://github.com/Positronic-Robotics/openpi) (branch `main-positronic`). The default training configuration is `pi05_positronic_lowmem`, which is LoRA that works with single H100 machine.

> All `docker compose` commands below assume you are in the [`docker`](https://github.com/Positronic-Robotics/positronic/tree/main/docker) directory (`cd docker`)

## 1. Prepare Data

Positronic datasets must be converted into the LeRobot format to be consumed by the OpenPI training pipeline. Use the `positronic-to-lerobot` service for this conversion.

**Command:**
```bash
docker compose run --rm -v ~/datasets:/data positronic-to-lerobot convert \
  --dataset=@positronic.cfg.dataset.encoded \
  --dataset.observation=.eepose_mujoco \
  --dataset.action=.absolute_position \
  --dataset.task='Pick up the green cube and place it on the red cube.' \
  --dataset.base.path=/data/my_raw_data \
  --output_dir=/data/my_lerobot_data \
  --fps=15
```

- `--dataset`: The dataset configuration. `@positronic.cfg.dataset.encoded` provides the structure.
- `--dataset.observation` and `--dataset.action` define how the Positronic dataset gets converted to LeRobot format. Our OpenPI configuration expects state and action in absolute end effector cooridantes.
- `--dataset.task`: The task description for the dataset.
- `--dataset.base.path`: Path to your raw Positronic dataset (collected via `positronic-data-collection`).
- `--output_dir`: Destination for the converted LeRobot dataset (can be local or `s3://bucket/path`).

## 2. Generate Assets

Before training, you must compute dataset statistics (normalization constants). The `openpi-stats` service handles this.

**Command:**
```bash
docker compose run --rm -v ~/datasets:/data openpi-stats \
  --input_path=/data/my_lerobot_data \
  --output_path=/data/openpi_assets
```

- `--input_path`: The directory containing the LeRobot dataset (from step 1).
- `--output_path`: Destination for the computed assets.

## 3. Train Model

Run the training job using the `openpi-train` service. You can customize the training process with various arguments provided [by training script](train.py).

**Command:**
```bash
docker compose run --rm -v ~/datasets:/data -v ~/checkpoints:/checkpoints openpi-train \
  --input_path=/data/my_lerobot_data \
  --stats_path=/data/openpi_assets/assets/ \
  --output_path=/checkpoints/openpi \
  --exp_name=experiment_v1
```

**Common Parameters:**
- `--config_name`: The OpenPI config to use (default: `pi05_positronic_lowmem`).
- `--exp_name`: Unique name for this run.
- `--num_train_steps`: Total training steps (optional).
- `--resume`: Set to `True` to resume an existing run from the same experiment directory.
- `--stats_path`: Path to the generated assets (must end in `.../assets/`).
- `--output_path`: Destination for checkpoints and logs.

If you want your run to report to wandb, add `docker/.env.wandb` containing your `WANDB_API_KEY`.

## 4. Serve Inference

To serve the trained model, launch the `openpi-server`. This exposes the policy via ZeroMQ or HTTP.

**Command:**
```bash
docker compose run --rm --service-ports -v ~/checkpoints:/checkpoints openpi-server \
  --checkpoints_dir=/checkpoints/openpi/pi05_positronic_lowmem/experiment_v1/
```

- `--checkpoints_dir`: Full path to the experiment directory containing checkpoints.
- `--checkpoint`: (Optional) Specific checkpoint step to load. If omitted, the server automatically loads the latest checkpoint.
- `--port`: (Optional) Port to serve on (default: 8000).

## 5. Run Inference

To evaluate the policy with a visual interface, run the inference client locally.

**Command:**
```bash
uv run positronic-inference \
  sim_openpi_positronic \
  --driver.simulation_time=20 \
  --driver.show_gui=True \
  --output_dir=~/datasets/inference_logs \
  --policy.host=vm-h100 \
  --policy.port=8000
```

- `sim_openpi_positronic`: The inference configuration preset.
- `--policy.host`: The machine that runs the inference server.
- `--policy.port`: The port that the inference server exposes.
