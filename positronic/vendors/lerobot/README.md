# LeRobot ACT Workflow in Positronic

This guide details the end-to-end workflow for training and deploying LeRobot ACT (Action Chunking Transformer) policies using the Positronic stack. The pipeline leverages Docker for reproducibility and supports both local directories and S3 for data storage.

> All `docker compose` commands below assume you are in the [`docker`](https://github.com/Positronic-Robotics/positronic/tree/main/docker) directory (`cd docker`)

## Available Codecs

LeRobot supports multiple codecs (observation encoder + action decoder pairs):

| Codec | Observation | Action | Use Case |
|-------|-------------|--------|----------|
| `eepose_absolute` | EE pose + grip | Absolute position | Default codec for end-effector control |
| `joints_absolute` | Joint positions + grip | Absolute position | Joint-space observations with task-space control |

**Key differences:**
- **`eepose_absolute`**: Uses end-effector pose (position + quaternion) + gripper state
- **`joints_absolute`**: Uses joint positions + gripper state

## 1. Prepare Data

Positronic datasets must be converted into LeRobot format using a codec.

**Command:**
```bash
docker compose run --rm positronic-to-lerobot convert \
  --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.lerobot.codecs.eepose_absolute \
  --output_dir=s3://interim/sim_stack_cubes/lerobot/eepose_absolute/ \
  --fps=15
```

**Available public datasets:**
- `@positronic.cfg.ds.phail.phail` - DROID teleoperation data (12GB, 352 episodes)
- `@positronic.cfg.ds.phail.sim_stack_cubes` - Simulated cube stacking (499MB, 317 episodes)
- `@positronic.cfg.ds.phail.sim_pick_place` - Simulated pick-and-place (1.3GB, 214 episodes)

**Parameters:**
- `--dataset.dataset`: The raw dataset configuration (see available datasets above)
- `--dataset.codec`: LeRobot codec that defines observation/action encoding (see table above)
- `--output_dir`: Destination for the converted LeRobot dataset (can be local or `s3://bucket/path`)
- `--fps`: Target frames per second for the converted dataset

**Standard path pattern:** `s3://interim/{dataset}/{vendor}/{codec}/`

Example: `s3://interim/sim_stack_cubes/lerobot/eepose_absolute/`

## 2. Train Model

Run the training job using the `lerobot-train` service.

**Command:**
```bash
docker compose run --rm lerobot-train \
  --dataset_root=s3://interim/sim_stack_cubes/lerobot/eepose_absolute/ \
  --run_name=experiment_v1 \
  --output_dir=s3://checkpoints/lerobot/ \
  --steps=50000 \
  --save_freq=10000
```

**Common Parameters:**
- `--codec`: Override the codec (default: `@positronic.vendors.lerobot.codecs.eepose_absolute`)
- `--run_name`: Unique name for this run
- `--steps`: Total training steps
- `--save_freq`: Checkpoint save interval
- `--resume`: Set to `True` to resume an existing run
- `--output_dir`: Destination for checkpoints and logs

**Example with different codec:**
```bash
docker compose run --rm lerobot-train \
  --dataset_root=s3://interim/sim_stack_cubes/lerobot/joints_absolute/ \
  --run_name=experiment_joints \
  --output_dir=s3://checkpoints/lerobot/ \
  --codec=@positronic.vendors.lerobot.codecs.joints_absolute \
  --steps=50000
```

WandB logging is enabled by default if `WANDB_API_KEY` is set. Add `docker/.env.wandb` containing your `WANDB_API_KEY`.

## 3. Serve Inference

To serve the trained model, launch the `lerobot-server`. This exposes a WebSocket API (same interface as GR00T/OpenPI servers) on port 8000.

**Command:**
```bash
docker compose run --rm --service-ports lerobot-server \
  --checkpoints_dir=s3://checkpoints/lerobot/experiment_v1/ \
  --codec=@positronic.vendors.lerobot.codecs.eepose_absolute \
  --port=8000 \
  --host=0.0.0.0
```

**Parameters:**
- `--checkpoints_dir`: Full path to the experiment directory containing checkpoints
- `--checkpoint`: (Optional) Specific checkpoint step to load (e.g., `10000`). If omitted, loads the latest checkpoint
- `--codec`: Codec used during training (must match training codec)
- `--port`: (Optional) Port to serve on (default: 8000)
- `--host`: (Optional) Host to bind to (default: 0.0.0.0)

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
  --policy.host=desktop \
  --policy.port=8000
```

- `--policy=.remote`: The remote policy client (WebSocket)
- `--policy.host`: The machine that runs the inference server
- `--policy.port`: The port that the inference server exposes (default: 8000)

## Architecture Notes

### LeRobot vs GR00T/OpenPI

LeRobot intentionally differs from GR00T/OpenPI in architecture:

| Component | GR00T/OpenPI | LeRobot |
|-----------|--------------|---------|
| **Server** | Subprocess-based | In-process (faster loading) |
| **Training** | Subprocess wrapper | Direct integration |

This is appropriate because:
- LeRobot ACT policies load quickly (<20s)
- Direct integration allows better error handling
- In-process training provides better control flow

### Task Field Handling

LeRobot ACT policies do NOT support task conditioning (single-task models). The observation encoder uses `task_field='task'` by default, but the LerobotPolicy filters out this field before passing observations to the ACT model.

For multi-task models (like OpenPI), use `task_field='prompt'` in the codec configuration.

## Troubleshooting

### Training fails with "invalid data type 'str'" error

**Problem:** ACT policies receive string fields (task, prompt) that they cannot process.

**Solution:** This should be fixed in the current version. Ensure you're using the latest code where:
1. LeRobot codecs use `task_field='task'` (default)
2. OpenPI codecs use `task_field='prompt'`
3. LerobotPolicy filters out `task` field before passing to ACT

### Server fails to load checkpoint

**Problem:** Server returns "Checkpoint not found" error.

**Solutions:**
1. Run `curl http://localhost:8000/api/v1/models` to see available checkpoints
2. Verify `--checkpoints_dir` path is correct (should end with experiment directory)
3. Check checkpoint directory structure: `checkpoints/<step>/pretrained_model/`
4. If using specific checkpoint, verify the checkpoint ID exists

### Codec mismatch error

**Problem:** Inference fails with shape or feature mismatch errors.

**Solution:** Ensure the codec used for inference matches the codec used during training. Check:
1. Training dataset codec: `s3://interim/{dataset}/lerobot/{codec}/`
2. Training command codec: `--codec=@positronic.vendors.lerobot.codecs.{codec}`
3. Server codec: Must match training codec exactly
