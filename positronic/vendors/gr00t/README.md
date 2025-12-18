# GR00T Workflow in Positronic

This guide details the end-to-end workflow for training and deploying GR00T models using the Positronic stack. The pipeline leverages Docker for reproducibility and supports both local directories and S3 for data storage. This integration relies on [our fork of GR00T](https://github.com/Positronic-Robotics/gr00t) (note main-positronic branch).

> All `docker compose` commands below assume you are in the [`docker`](https://github.com/Positronic-Robotics/positronic/tree/main/docker) directory (`cd docker`)

> Note: if you customize compose volumes, **do not bind-mount** host `~/.local/share/uv` into `/root/.local/share/uv` for GR00T containers. The GR00T image may rely on uv-managed CPython inside the image, and the bind mount can hide that path and break `/.venv/bin/python` with `ENOENT`.

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
- `--data_config`: The data configuration to use (default: `ee_absolute`).
- `--exp_name`: Unique name for this run.
- `--num_train_steps`: Total training steps (optional).
- `--resume`: Set to `True` to resume an existing run.
- `--output_path`: Destination for checkpoints and logs.

If you want your run to report to wandb, add `docker/.env.wandb` containing your `WANDB_API_KEY`.

## 3. Serve Inference

To serve the trained model, launch the `groot-server`. This exposes the policy via ZeroMQ or HTTP. The model can be served from the machine with 8GB of GPU memory.

**Command:**
```bash
docker compose run --rm --service-ports -v ~/checkpoints:/checkpoints groot-server \
  --checkpoints_dir=/checkpoints/groot/experiment_v1/
```

- `--checkpoints_dir`: Full path to the experiment directory containing checkpoints.
- `--checkpoint`: (Optional) Specific checkpoint ID (e.g., `50000`). If omitted, loads the latest `checkpoint-N` folder.
- `--port`: (Optional) Port to serve on (default: 9000).

## 4. Run Inference

To evaluate the policy with a visual interface, run the inference client locally.

**Command:**
```bash
uv run positronic-inference \
  sim_groot \
  --driver.simulation_time=20 \
  --driver.show_gui=True \
  --output_dir=~/datasets/inference_logs \
  --policy.host=vm-h100 \
  --policy.port=9000
```

- `sim_groot`: The inference configuration preset for GR00T.
- `--policy.host`: The machine that runs the inference server.
- `--policy.port`: The port that the inference server exposes (default: 9000).
