---
name: remote-training
description: Manages remote training infrastructure on Nebius VMs. Use for building/pushing Docker images, starting/stopping VM machines (train, train2, train3), running training jobs, dataset generation, and starting inference servers.
---

# Remote Training Infrastructure

## Overview

This skill manages the Positronic training infrastructure on Nebius GPU VMs. It covers Docker image management, VM lifecycle, training jobs, dataset generation, and inference server deployment.

## Prerequisites

- Docker contexts configured for VMs: `vm-train`, `vm-train2`, `vm-train3`
- AWS S3 access configured for checkpoint/dataset storage
- Nebius CLI authenticated (for VM start/stop)

## Available Machines

| Context | GPU | Use Case |
|---------|-----|----------|
| `desktop` | RTX 3060 (12GB) | Dataset generation, GR00T inference, lerobot training |
| `notebook` | RTX 4060 Laptop (8GB) | Light tasks, testing, dataset generation |
| `vm-train` | H100 (80GB) | GR00T/OpenPI training and inference |
| `vm-train2` | H100 (80GB) | GR00T/OpenPI training and inference |
| `vm-train3` | H100 (80GB) | GR00T/OpenPI training and inference |

**Important**: Only GR00T training/inference and OpenPI training/inference require H100. Other jobs (dataset generation, lerobot) can run on `desktop`.

## Docker Images

### Image Overview

| Image | Source | Depends On | Used For |
|-------|--------|------------|----------|
| `positro/positronic` | `positronic/docker/` | - | Dataset conversion, lerobot training/inference |
| `positro/gr00t` | `positronic/docker/` | `positro/gr00t-base` | GR00T training and inference |
| `positro/gr00t-base` | `gr00t/docker/` | - | Base image for GR00T |
| `positro/openpi` | `positronic/docker/` | `positro/openpi-base` | OpenPI training and inference |
| `positro/openpi-base` | `openpi/docker/` | - | Base image for OpenPI |

### Build Order for Cross-Repo Changes

If you modify code in `../gr00t` or `../openpi`:

1. **For gr00t changes:**
   ```bash
   cd /home/vertix/dev/gr00t/docker
   make push  # Pushes positro/gr00t-base
   cd /home/vertix/dev/positronic/docker
   make push-groot  # Rebuilds and pushes positro/gr00t with new base
   ```

2. **For openpi changes:**
   ```bash
   cd /home/vertix/dev/openpi/docker
   make push  # Pushes positro/openpi-base
   cd /home/vertix/dev/positronic/docker
   make push-openpi  # Rebuilds and pushes positro/openpi with new base
   ```

3. **For positronic-only changes:**
   ```bash
   cd /home/vertix/dev/positronic/docker
   make push-training  # Just positro/positronic
   # Or for specific images:
   make push-groot     # positro/gr00t
   make push-openpi    # positro/openpi
   make push           # All images
   ```

## VM Machine Management

**IMPORTANT**: Before using any VM, always check if it is already running a job. Never assume a machine is free. For experiments and validation runs, start a fresh VM rather than reusing one that may be occupied.

### Selecting a Machine

1. Check which VMs are reachable and what they're running:
   ```bash
   # Check connectivity
   ssh -o ConnectTimeout=5 vertix@vm-train 'echo connected' 2>&1
   ssh -o ConnectTimeout=5 vertix@vm-train2 'echo connected' 2>&1
   ssh -o ConnectTimeout=5 vertix@vm-train3 'echo connected' 2>&1

   # Check running containers on reachable VMs
   docker --context vm-train ps 2>/dev/null
   docker --context vm-train2 ps 2>/dev/null
   docker --context vm-train3 ps 2>/dev/null
   ```

2. If a VM has running containers, it is **taken** — pick a different one or start a stopped VM.

3. If no free VM is available, start one that is currently stopped:
   ```bash
   ../internal/scripts/start.sh train   # or train2, train3
   ```

### Start a VM

```bash
../internal/scripts/start.sh train
../internal/scripts/start.sh train2
../internal/scripts/start.sh train3
```

**Note**: Requires Nebius CLI authentication. Must be run from a terminal with browser access for OAuth flow.

### Docker Contexts

```bash
docker context ls                     # List available contexts
docker --context vm-train ps          # Check containers on vm-train
docker --context vm-train2 ps         # Check containers on vm-train2
```

## Pipeline Overview

```
1. Data Collection (positronic-data-collection)
        ↓
2. Dataset Conversion (positronic-to-lerobot) [desktop]
        ↓
3. [OpenPI only] Generate Stats (openpi-stats) [desktop]
        ↓
4. Training (groot-train / openpi-train) [H100]
        ↓
5. Inference Server (groot-server / openpi-server) [H100 or desktop]
        ↓
6. Inference Client (positronic-inference) [local]
```

## Dataset Generation

### Convert Positronic Dataset to LeRobot Format

From `docker/` directory (can run on `desktop`):

```bash
CACHE_ROOT=/home/vertix docker --context desktop compose run --rm --pull always positronic-to-lerobot convert \
  --dataset.dataset=@positronic.cfg.ds.phail.phail \
  --dataset.codec=@positronic.vendors.gr00t.codecs.ee_rot6d_joints \
  --output_dir=s3://interim/phail/groot/ee_rot6d_joints/
```

> **Note**: `CACHE_ROOT=/home/vertix` is needed when running from Mac — `$HOME` expands locally to `/Users/vertix`, but volume mounts must reference the remote host's paths. Harmless on Linux.

### Available Codecs

| Vendor | Codec | Description |
|--------|-------|-------------|
| GR00T | `@positronic.vendors.gr00t.codecs.ee_quat` | EE pose (quaternion) + grip |
| GR00T | `@positronic.vendors.gr00t.codecs.ee_quat_joints` | EE pose + joint positions + grip |
| GR00T | `@positronic.vendors.gr00t.codecs.ee_rot6d` | EE pose (6D rotation) + grip |
| GR00T | `@positronic.vendors.gr00t.codecs.ee_rot6d_joints` | 6D rotation + joint positions + grip |
| LeRobot | `@positronic.vendors.lerobot_0_3_3.codecs.ee` | EE pose (quat) + grip, absolute actions |
| LeRobot | `@positronic.vendors.lerobot_0_3_3.codecs.joints` | Joint positions + grip, absolute actions |
| OpenPI | `@positronic.vendors.openpi.codecs.ee` | EE pose + grip, absolute actions |
| OpenPI | `@positronic.vendors.openpi.codecs.ee_joints` | EE pose + joints + grip, absolute actions |

## GR00T Training

From `docker/` directory, on H100 VM:

```bash
docker --context vm-train compose run --rm --pull=always groot-train \
  --input_path=s3://interim/sim_ft/groot_rot6d_q/ \
  --output_path=s3://checkpoints/sim_ft/groot_rot6d_q/ \
  --exp_name=YYMMDD \
  --num_train_steps=20000 \
  --save_steps=2000 \
  --num_workers=4 \
  --modality_config=ee_rot6d_q
```

### GR00T Modality Configs

| Config | Description |
|--------|-------------|
| `ee` | End-effector pose (quaternion) |
| `ee_q` | EE pose + joint feedback |
| `ee_rot6d` | EE pose with 6D rotation |
| `ee_rot6d_q` | 6D rotation + joint feedback |
| `ee_rot6d_rel` | 6D rotation, relative actions |
| `ee_rot6d_q_rel` | 6D rotation + joints, relative actions |

## OpenPI Training

From `docker/` directory, on H100 VM:

```bash
# 1. Generate stats (can run on desktop)
docker compose run --rm openpi-stats \
  --input_path=s3://interim/my_lerobot_data \
  --output_path=s3://interim/openpi_assets

# 2. Train (requires H100)
docker --context vm-train compose run --rm --pull=always openpi-train \
  --input_path=s3://interim/my_lerobot_data \
  --stats_path=s3://interim/openpi_assets/assets/ \
  --output_path=s3://checkpoints/openpi \
  --exp_name=experiment_v1
```

## Inference Servers

### GR00T Server (requires GPU)

```bash
docker compose run --rm --service-ports groot-server \
  ee_rot6d_joints \
  --checkpoints_dir=s3://checkpoints/sim_ft/groot_rot6d_q/040126/
```

**Available variants:** `ee`, `ee_joints`, `ee_rot6d`, `ee_rot6d_joints`, `ee_rot6d_rel`, `ee_rot6d_joints_rel`

The server exposes a WebSocket API on port 8000 (same as lerobot-server for interchangeability).

### OpenPI Server (requires H100)

```bash
docker --context vm-train compose run --rm --service-ports openpi-server \
  --checkpoints_dir=s3://checkpoints/openpi/pi05_positronic_lowmem/experiment_v1/
```

### LeRobot/ACT Server (can run on desktop)

```bash
docker compose run --rm --service-ports lerobot-server \
  --checkpoints_dir=s3://checkpoints/act/experiment_v1/
```

## Inference Client

All servers (GR00T, LeRobot, OpenPI) now expose the same WebSocket API on port 8000, so the client uses the same `.remote` policy config.

### With GUI (requires display)

```bash
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=desktop \
  --policy.port=8000 \
  --driver.show_gui
```

### Headless (no display required)

```bash
MUJOCO_GL=egl uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=desktop \
  --policy.port=8000 \
  --driver.show_gui=False \
  --driver.simulation_time=10
```

### Server Types

| Server Type | Codec Config | Notes |
|-------------|--------------|-------|
| GR00T | `ee_rot6d_joints` (positional variant arg) | Matches `modality_config=ee_rot6d_q` |
| LeRobot ACT | `--codec=@positronic.vendors.lerobot_0_3_3.codecs.ee` | Default codec |
| OpenPI | `--codec=@positronic.vendors.openpi.codecs.ee` | Default codec |

## Sim Eval End-to-End

Full workflow: start inference server → run sim episodes → view results.

### 1. Start LeRobot inference server (on GPU machine)

```bash
CACHE_ROOT=/home/vertix docker --context <machine> compose run -d --rm --pull always --service-ports lerobot-server \
  --checkpoints_dir=s3://checkpoints/<path_to_experiment>/
```

**Important**: Use `CACHE_ROOT=/home/vertix` when targeting remote Docker contexts (notebook, vm-train, etc.) because `$HOME` expands to the local Mac path, but volume mounts must reference paths on the remote host.

Wait for the server to be ready by checking logs:
```bash
docker --context <machine> logs --tail 5 <container_id>
# Look for: "Uvicorn running on http://0.0.0.0:8000"
```

### 2. Run sim inference (on same or different GPU machine)

```bash
CACHE_ROOT=/home/vertix docker --context <machine> compose run --rm --pull always positronic-inference \
  sim \
  --policy=.remote \
  --policy.host=<server_machine> \
  --policy.port=8000 \
  --driver.num_iterations=50 \
  --driver.simulation_time=30 \
  --output_dir=s3://inference/sim_stack_validation/<run_name>/<model_type>
```

### 3. View results (locally)

Use the eval server (not `positronic-server` directly) to get grouping by model/checkpoint with success rates, UPH, and MTBF:

```bash
uv run python -m positronic.cfg.eval sim \
  --dataset.base.path=s3://inference/sim_stack_validation/<run_name> \
  --reset_cache
```

**Note**: Always use `--reset_cache` to clear stale RRD files from previous runs.

Opens on http://localhost:5001. The path should point to the parent directory containing model subdirs (e.g., `170226`, not `170226/lerobot`). Episodes are grouped by model and checkpoint on the home page.

### 4. Clean up: stop the inference server

```bash
docker --context <machine> stop <container_id>
```

### Naming convention

Inference results go to `s3://inference/sim_stack_validation/<DDMMYY[-suffix]>/<model_type>/` where:
- `<DDMMYY[-suffix]>` — date with optional descriptor (e.g., `160226-dinov3`)
- `<model_type>` — `lerobot`, `groot`, or `openpi`

## Monitoring Background Jobs

When running jobs in background:

```bash
# Check progress percentage
grep -o '[0-9]*%' /tmp/claude/-home-vertix-dev-positronic/tasks/<task_id>.output | tail -1

# View recent output
tail -50 /tmp/claude/-home-vertix-dev-positronic/tasks/<task_id>.output

# Check for completion/errors
grep -i "error\|complete\|finished" /tmp/claude/-home-vertix-dev-positronic/tasks/<task_id>.output
```

## Common Issues

### CUDA Out of Memory
Each GR00T server uses ~6GB GPU memory. On 12GB GPUs (desktop), only run one server at a time.

### Port Already Allocated
```bash
docker ps -a | grep -E "groot-server|openpi-server"
docker stop <container_id> && docker rm <container_id>
```

### VM Not Reachable
1. Start the VM: `../internal/scripts/start.sh train2`
2. Verify SSH: `ssh -o ConnectTimeout=5 vertix@vm-train2 'echo connected'`

### Parquet Object Array Error
If dataset generation fails with `ValueError: setting an array element with a sequence`, the fix is in `positronic/dataset/vector.py` - use `np.stack()` to convert object arrays to proper 2D arrays.

### gladLoadGL Error (Headless)
Use `MUJOCO_GL=egl` environment variable for headless rendering:
```bash
MUJOCO_GL=egl uv run positronic-inference sim --driver.show_gui=False ...
```

### Nebius Auth (Manual Flow for Headless Environments)

When running from a headless environment without browser access:

1. **Start nebius in background with `--no-browser`:**
   ```bash
   nebius --no-browser --auth-timeout 5m iam whoami 2>&1
   ```
   Run this in background and extract the auth URL from output.

2. **Give the auth URL to the user** - they click it and authenticate in their browser.

3. **User's browser redirects to localhost URL** like:
   ```
   http://127.0.0.1:PORT/?code=XXX&state=YYY
   ```
   The page won't load (expected). User copies this full URL from address bar.

4. **Curl the localhost URL on the machine running nebius:**
   ```bash
   curl -s "http://127.0.0.1:PORT/?code=XXX&state=YYY"
   # Returns: "Login is successful, you may close the browser tab"
   ```

5. **Auth completes** - nebius background process finishes, credentials are cached.

After authentication, VM start scripts will work:
```bash
../internal/scripts/start.sh train
```
