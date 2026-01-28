# OpenPI Workflow in Positronic

This guide details the end-to-end workflow for training and deploying OpenPI models using the Positronic stack. The pipeline leverages Docker for reproducibility and supports both local directories and S3 for data storage. This integration relies on [our fork of OpenPI](https://github.com/Positronic-Robotics/openpi) (branch `main-positronic`). The default training configuration is `pi05_positronic_lowmem`, which is LoRA that works with single H100 machine.

> All `docker compose` commands below assume you are in the [`docker`](https://github.com/Positronic-Robotics/positronic/tree/main/docker) directory (`cd docker`)

## Available Codecs

OpenPI supports multiple codecs (observation encoder + action decoder pairs) for different use cases:

| Codec | Observation | Action | Use Case |
|-------|-------------|--------|----------|
| `eepose_absolute` | EE pose + grip | Absolute position | Default codec for LeRobot/ACT-style policies |
| `openpi_positronic` | EE pose + grip (OpenPI format) | Absolute position | OpenPI policies trained with positronic config |
| `droid` | Joint positions + grip | Joint delta (velocity) | DROID dataset compatibility |
| `eepose_q` | EE pose + grip + joints | Absolute position | Combined feedback for better performance |
| `joints` | Joint positions + grip | Absolute position | Joint-space observations with task-space control |

**Key differences:**
- **`eepose_absolute`** vs **`openpi_positronic`**: Same semantic content, different key format (`observation.state` vs `observation/state`)
- **`droid`**: Uses joint delta actions instead of absolute position (critical for DROID dataset)
- **`eepose_q`**: Includes both EE pose and joint positions in observation for richer feedback

## 1. Prepare Data

Positronic datasets must be converted into the LeRobot format using an OpenPI codec.

**Command:**
```bash
docker compose run --rm -v ~/datasets:/data positronic-to-lerobot convert \
  --dataset.dataset=@positronic.cfg.ds.phail.phail \
  --dataset.codec=@positronic.vendors.openpi.codecs.droid \
  --output_dir=/data/my_lerobot_data \
  --fps=15
```

**Available public datasets:**
- `@positronic.cfg.ds.phail.phail` - DROID teleoperation data (12GB, 352 episodes)
- `@positronic.cfg.ds.phail.sim_stack_cubes` - Simulated cube stacking (499MB, 317 episodes)
- `@positronic.cfg.ds.phail.sim_pick_place` - Simulated pick-and-place (1.3GB, 214 episodes)

**Examples for different codecs:**
```bash
# Default codec (EE pose + grip → absolute position)
--dataset.codec=@positronic.vendors.openpi.codecs.eepose_absolute

# DROID dataset (joint positions → joint delta)
--dataset.codec=@positronic.vendors.openpi.codecs.droid

# Combined feedback (EE pose + joints → absolute position)
--dataset.codec=@positronic.vendors.openpi.codecs.eepose_q
```

**Parameters:**
- `--dataset.dataset`: The raw dataset configuration (see available datasets above)
- `--dataset.codec`: OpenPI codec that defines observation/action encoding (see table above)
- `--output_dir`: Destination for the converted LeRobot dataset (can be local or `s3://bucket/path`)
- `--fps`: Target frames per second for the converted dataset

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

The OpenPI inference server wraps the OpenPI policy in a FastAPI server that provides a unified API across all vendors (GR00T, LeRobot, OpenPI). The server manages the OpenPI subprocess and handles observation encoding/action decoding.

### Starting the Server

The server supports multiple codec variants as pre-configured commands:

```bash
# Default codec (eepose_absolute)
docker compose run --rm --service-ports -v ~/checkpoints:/checkpoints openpi-server \
  eepose_absolute \
  --checkpoints_dir=/checkpoints/openpi/pi05_positronic_lowmem/experiment_v1/

# OpenPI positronic codec
docker compose run --rm --service-ports -v ~/checkpoints:/checkpoints openpi-server \
  openpi_positronic \
  --checkpoints_dir=/checkpoints/openpi/pi05_positronic_lowmem/experiment_v1/

# DROID codec (joint delta actions)
docker compose run --rm --service-ports -v ~/checkpoints:/checkpoints openpi-server \
  droid \
  --checkpoints_dir=/checkpoints/openpi/pi05_droid/experiment_v1/

# Combined feedback (EE pose + joints)
docker compose run --rm --service-ports -v ~/checkpoints:/checkpoints openpi-server \
  eepose_q \
  --checkpoints_dir=/checkpoints/openpi/pi05_positronic_lowmem/experiment_v1/
```

**Parameters:**
- `<codec_variant>`: One of `eepose_absolute`, `openpi_positronic`, `droid`, `eepose_q`, `joints`
- `--checkpoints_dir`: Full path to the experiment directory containing checkpoints
- `--checkpoint`: (Optional) Specific checkpoint step to load. If omitted, loads the latest checkpoint
- `--config_name`: (Optional) OpenPI config name (default: `pi05_positronic_lowmem`)
- `--port`: (Optional) Port to serve on (default: 8000)
- `--openpi_ws_port`: (Optional) Internal port for OpenPI subprocess (default: 8001)

### API Endpoints

The server exposes the following endpoints:

**GET `/api/v1/models`**
- Returns list of available checkpoints
- Response: `{"models": ["checkpoint-1000", "checkpoint-2000", ...]}`

**WebSocket `/api/v1/session`**
- Default session (uses latest checkpoint)
- Sends metadata on connection, then enters inference loop
- Client sends serialized observations, server responds with serialized actions

**WebSocket `/api/v1/session/{checkpoint_id}`**
- Session with specific checkpoint
- Same protocol as default session

**Message Protocol:**
1. Client connects to WebSocket
2. Server sends: `{'meta': {...}}` (checkpoint info, codec metadata)
3. For each inference step:
   - Client sends: serialized observation dict
   - Server responds: `{'result': action_dict}` or `{'error': error_message}`

### Example Client Connection

```python
from websockets.sync.client import connect
from positronic.utils.serialization import serialise, deserialise

# Connect to server
ws = connect('ws://localhost:8000/api/v1/session')

# Receive metadata
metadata = deserialise(ws.recv())
print(f"Connected to checkpoint: {metadata['meta']['checkpoint_id']}")

# Send observation and receive action
observation = {
    'robot_state.ee_pose': [0.1, 0.2, 0.3, 0, 0, 0, 1],
    'grip': [0.5],
    'image.wrist': wrist_image,
    'image.exterior': exterior_image,
}
ws.send(serialise(observation))
response = deserialise(ws.recv())
action = response['result']
```

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

## Troubleshooting

### Server fails to start

**Problem:** Server exits with "OpenPI subprocess exited with code 1"

**Solutions:**
1. Check checkpoint directory exists and contains valid checkpoint files
2. Verify config_name matches the training config used
3. Check OpenPI subprocess logs for dependency issues
4. Ensure OpenPI repository is available at `../openpi/` (sibling directory)

### WebSocket connection refused

**Problem:** Client cannot connect to server WebSocket endpoint

**Solutions:**
1. Verify server is running with `--service-ports` flag (exposes port 8000)
2. Check firewall settings allow connections on port 8000
3. Try `curl http://localhost:8000/api/v1/models` to verify server is responsive
4. Check server logs for startup errors

### Checkpoint not found

**Problem:** Server returns "Checkpoint not found" error

**Solutions:**
1. Run `curl http://localhost:8000/api/v1/models` to see available checkpoints
2. Verify `--checkpoints_dir` path is correct (should end with experiment directory)
3. Check checkpoint directory structure: `checkpoints/<checkpoint-id>/`
4. If using specific checkpoint, verify the checkpoint ID exists

### Action decoding fails

**Problem:** Server returns error during action decoding

**Solutions:**
1. Verify codec matches the model training config:
   - DROID models need `droid` codec (joint delta actions)
   - Positronic models need `openpi_positronic` codec
2. Check observation format matches codec requirements
3. Verify image shapes are correct (will be resized to 224x224)
4. Check action space dimensions match expected values

### Subprocess startup timeout

**Problem:** "OpenPI subprocess did not become ready within 120s"

**Solutions:**
1. First startup may be slow (model download, loading weights)
2. Check available GPU memory (OpenPI requires ~8GB VRAM)
3. Increase timeout by modifying `_wait_for_ready(timeout=...)` in server.py
4. Check OpenPI subprocess logs for slow operations

### Migration from old server

**Problem:** Need to migrate from old `main()` subprocess wrapper to new FastAPI server

**Solutions:**
1. Old wrapper still works for local use: `uv run positronic/vendors/openpi/server.py main ...`
2. New server provides unified API: use pre-configured variants (`eepose_absolute`, `droid`, etc.)
3. Update docker-compose.yml to use new server format
4. See "Starting the Server" section for examples
