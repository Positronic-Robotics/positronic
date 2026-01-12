# GR00T N1.6 Workflow in Positronic

This guide details the end-to-end workflow for training and deploying GR00T N1.6 models using the Positronic stack. The pipeline leverages Docker for reproducibility and supports both local directories and S3 for data storage. This integration relies on [our fork of GR00T](https://github.com/Positronic-Robotics/gr00t).

> All `docker compose` commands below assume you are in the [`docker`](https://github.com/Positronic-Robotics/positronic/tree/main/docker) directory (`cd docker`)

> Note: if you customize compose volumes, **do not bind-mount** host `~/.local/share/uv` into `/root/.local/share/uv` for GR00T containers. The GR00T image uses a pre-built Python 3.10 venv at `/.venv/`, and the bind mount can hide that path and break `/.venv/bin/python` with `ENOENT`.

## 1. Prepare Data

Positronic datasets must be converted into the LeRobot format to be consumed by the GR00T training pipeline. Use the `positronic-to-lerobot` service for this conversion.

**Command:**
```bash
docker compose run --rm -v ~/datasets:/data positronic-to-lerobot convert \
  --dataset=.encoded \
  --dataset.observation=.groot_ee_absolute \
  --dataset.action=.absolute_position \
  --dataset.task='Pick up the green cube and place it on the red cube.' \
  --dataset.base.path=/data/my_raw_data \
  --output_dir=/data/my_lerobot_data \
  --fps=15
```

- `--dataset`: The dataset configuration. `@positronic.cfg.dataset.encoded` provides the structure.
- `--dataset.observation`: Use `.groot_ee_absolute` to match GR00T's expected observation input.
- `--dataset.base.path`: Container-side path to your raw Positronic dataset (mounted via `-v`).
- `--output_dir`: Destination for the converted LeRobot dataset.

## 2. Train Model

Run the training job using the `groot-train` service. We use one H100 machine to train the model.

**Command:**
```bash
docker compose run --rm -v ~/datasets:/data -v ~/checkpoints:/checkpoints groot-train \
  --input_path=/data/my_lerobot_data \
  --output_path=/checkpoints/groot \
  --exp_name=experiment_v1
```

**Common Parameters:**
- `--modality_config`: The modality configuration to use. Options: `ee` (default, EE pose control), `ee_q` (EE pose + joint feedback). Can also be a path to a custom config file.
- `--exp_name`: Unique name for this run.
- `--num_train_steps`: Total training steps (optional).
- `--learning_rate`: Override default learning rate (optional).
- `--save_steps`: Checkpoint save interval (optional).
- `--num_workers`: Number of dataloader workers (optional).
- `--resume`: Set to `True` to resume an existing run.
- `--output_path`: Destination for checkpoints and logs.

WandB logging is enabled by default. Add `docker/.env.wandb` containing your `WANDB_API_KEY`.

## 3. Serve Inference

To serve the trained model, launch the `groot-server`. This exposes a WebSocket API (same interface as lerobot-server) on port 8000. The model can be served from a machine with 8GB of GPU memory.

**Command:**
```bash
docker compose run --rm --service-ports -v ~/checkpoints:/checkpoints groot-server \
  --checkpoints_dir=/checkpoints/groot/experiment_v1/ \
  --observation_encoder=.groot \
  --action_decoder=.groot \
  --modality_config=ee
```

**Parameters:**
- `--checkpoints_dir`: Full path to the experiment directory containing checkpoints.
- `--checkpoint`: (Optional) Specific checkpoint ID (e.g., `50000`). If omitted, loads the latest `checkpoint-N` folder.
- `--observation_encoder`: Observation encoder config. Options: `.groot`, `.groot_joints`, `.groot_rot6d`, `.groot_rot6d_joints`.
- `--action_decoder`: Action decoder config. Options: `.groot`, `.groot_rot6d`.
- `--modality_config`: Modality config for the gr00t model. Must match training config.
- `--port`: (Optional) Port to serve on (default: 8000).

**Endpoints:**
- `GET /api/v1/models` - List available checkpoints
- `WebSocket /api/v1/session` - Inference session (uses latest checkpoint)
- `WebSocket /api/v1/session/{checkpoint_id}` - Inference with specific checkpoint

## 4. Run Inference

To evaluate the policy with a visual interface, run the inference client locally. The client uses the same `.remote` policy for all server types (GR00T, LeRobot, OpenPI).

**Command:**
```bash
uv run positronic-inference sim \
  --driver.simulation_time=20 \
  --driver.show_gui=True \
  --output_dir=~/datasets/inference_logs \
  --policy=.remote \
  --policy.host=vm-h100 \
  --policy.port=8000
```

- `--policy=.remote`: The remote policy client (WebSocket).
- `--policy.host`: The machine that runs the inference server.
- `--policy.port`: The port that the inference server exposes (default: 8000).
