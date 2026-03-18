---
name: remote-training
description: Manages remote training infrastructure on Nebius VMs. Use for building/pushing Docker images, starting/stopping VM machines (train, train2, train3), running training jobs, dataset generation, and starting inference servers.
---

# Remote Training Infrastructure

This skill manages the Positronic training infrastructure on Nebius GPU VMs. It covers Docker image management, VM lifecycle, training jobs, dataset generation, and inference server deployment.

## S3 Convention

```
s3://interim/{dataset}/{vendor}/{codec}/          — converted LeRobot datasets
s3://checkpoints/{dataset}/{vendor}/{codec_or_experiment}/  — training output
s3://inference/{dataset}/{date_or_exp}/{vendor}/  — inference eval results
```

Every run writes `run_metadata_*.yaml` with the full CLI command — read it to reconstruct the pipeline.

### Current Artifacts

**sim_stack** (automated testing — `@positronic.cfg.ds.phail.sim_stack_cubes`):

| Vendor | Codec | Interim | Latest Checkpoint |
|--------|-------|---------|-------------------|
| groot | `ee_rot6d` | `s3://interim/sim_stack/groot/ee_rot6d/` | `s3://checkpoints/sim_stack/groot/ee_rot6d/230226/` |
| lerobot (0.4.x) | `ee` | `s3://interim/sim_stack/lerobot_04/ee/` | `s3://checkpoints/sim_stack/lerobot_04/smolvla_150k/` |
| lerobot (0.3.3) | `ee` | `s3://interim/sim_stack/lerobot/ee/` | `s3://checkpoints/sim_stack/lerobot/230226-ee/` |
| openpi | `ee` | `s3://interim/sim_stack/openpi/ee/` | `s3://checkpoints/sim_stack/openpi/ee/pi05_positronic_lowmem/230226/` |

**phail_unified** (production — `@positronic.cfg.ds.phail.phail_unified`):

| Vendor | Codec | Interim | Latest Checkpoint |
|--------|-------|---------|-------------------|
| groot | `ee_rot6d` | `s3://interim/phail_unified/groot/ee_rot6d/` | — |
| lerobot (0.4.x) / SmolVLA | `ee` | — | `s3://checkpoints/phail_unified/smolvla/170316_ee/` |
| lerobot (0.3.3) | `ee` | `s3://interim/phail_unified/lerobot/ee/` | — |
| openpi | `ee` | `s3://interim/phail_unified/openpi/ee/` | — |

## Two LeRobot Versions

Positronic ships two LeRobot integrations because other vendors (GR00T, OpenPI) depend on the 0.3.3 dataset format:

| | LeRobot 0.4.x | LeRobot 0.3.3 |
|---|---|---|
| **Convert** | `lerobot-convert` | `lerobot-0_3_3-convert` |
| **Train** | `lerobot-train` | `lerobot-0_3_3-train` |
| **Serve** | `lerobot-server` | `lerobot-0_3_3-server` |
| **Codecs** | `@positronic.vendors.lerobot.codecs.*` | `@positronic.vendors.lerobot_0_3_3.codecs.*` |
| **GPU** | Desktop (consumer GPU) | Desktop (consumer GPU) |

`lerobot-0_3_3-convert` is also used for GR00T and OpenPI dataset conversion.

## Machines

| Context | GPU | Use Case |
|---------|-----|----------|
| `desktop` | RTX 3060 (12GB) | Dataset generation, lerobot training/inference, GR00T inference |
| `notebook` | RTX 4060 Laptop (8GB) | Light tasks, testing, dataset generation |
| `vm-train` | H100 (80GB) | GR00T/OpenPI training and inference |
| `vm-train2` | H100 (80GB) | GR00T/OpenPI training and inference |
| `vm-train3` | H100 (80GB) | GR00T/OpenPI training and inference |

Only GR00T and OpenPI training/inference require H100. Everything else runs on `desktop`.

### VM Management

**IMPORTANT**: Always check if a VM is running a job before using it.

```bash
# Check connectivity and running containers
ssh -o ConnectTimeout=5 vertix@vm-train 'echo connected' 2>&1
docker --context vm-train ps 2>/dev/null

# Start a stopped VM
../internal/scripts/start.sh train   # or train2, train3
```

## Docker Images

| Image | Source | Used For |
|-------|--------|----------|
| `positro/positronic` | `positronic/docker/` | Dataset conversion, lerobot training/inference |
| `positro/gr00t` | `positronic/docker/` (depends on `positro/gr00t-base`) | GR00T training and inference |
| `positro/openpi` | `positronic/docker/` (depends on `positro/openpi-base`) | OpenPI training and inference |
| `positro/dreamzero` | `positronic/docker/` (depends on `positro/dreamzero-base`) | DreamZero inference |

Images are tagged by branch name. `make push` in `docker/` auto-derives the tag from the current git branch.

```bash
cd /home/vertix/dev/positronic/docker
make push-training  # Just positro/positronic
make push-groot     # positro/gr00t (rebuild base first if ../gr00t changed)
make push-openpi    # positro/openpi (rebuild base first if ../openpi changed)
make push           # All images
```

For cross-repo base image rebuilds: `cd ../gr00t/docker && make push` then `cd ../positronic/docker && make push-groot`.

## Pipeline

All commands run from `docker/` directory. Use `CACHE_ROOT=/home/vertix` when targeting remote Docker contexts from Mac.

### 1. Convert Dataset

```bash
# GR00T / OpenPI / LeRobot 0.3.3 — use lerobot-0_3_3-convert
CACHE_ROOT=/home/vertix docker --context desktop compose run --rm --pull always lerobot-0_3_3-convert convert \
  --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.gr00t.codecs.ee_rot6d \
  --output_dir=s3://interim/sim_stack/groot/ee_rot6d/

# LeRobot 0.4.x (SmolVLA) — use lerobot-convert
CACHE_ROOT=/home/vertix docker --context desktop compose run --rm --pull always lerobot-convert convert \
  --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.lerobot.codecs.ee \
  --output_dir=s3://interim/sim_stack/lerobot_04/ee/
```

**Default codecs**: groot `ee_rot6d`, lerobot `ee`, openpi `ee`.

### 2. Train

Each vendor has a training script at `positronic/vendors/{vendor}/train.py` with usage examples in its docstring. Read the script's docstring for available subcommands and parameters.

General pattern:

```bash
# From docker/ directory
[CACHE_ROOT=/home/vertix] docker [--context <machine>] compose run --rm --pull always <service> \
  [subcommand] --input_path=<interim_path> --exp_name=<name> --output_dir=<checkpoint_path> ...
```

| Vendor | Docker service | Machine | Script (read docstring for usage) |
|--------|---------------|---------|-----------------------------------|
| LeRobot 0.4.x (SmolVLA) | `lerobot-train` | desktop | `positronic/vendors/lerobot/train.py` |
| LeRobot 0.3.3 (ACT) | `lerobot-0_3_3-train` | desktop | `positronic/vendors/lerobot_0_3_3/train.py` |
| GR00T | `groot-train` | H100 | `positronic/vendors/gr00t/train.py` |
| OpenPI | `openpi-train` (needs `openpi-stats` first) | H100 | `positronic/vendors/openpi/train.py` |

**Resume any training**: add `--resume=true` to the same command.

### 3. Start Inference Server

All servers use subcommands: `serve` for custom checkpoints, or named presets like `phail`, `sim_stack`.

```bash
# LeRobot 0.4.x SmolVLA — preset (desktop)
CACHE_ROOT=/home/vertix docker --context desktop compose run --rm --pull always --service-ports lerobot-server \
  phail

# LeRobot 0.4.x SmolVLA — custom checkpoint (desktop)
CACHE_ROOT=/home/vertix docker --context desktop compose run --rm --pull always --service-ports lerobot-server \
  serve \
  --checkpoints_dir=s3://checkpoints/sim_stack/lerobot_04/smolvla_150k/

# LeRobot 0.3.3 ACT (desktop)
CACHE_ROOT=/home/vertix docker --context desktop compose run --rm --pull always --service-ports lerobot-0_3_3-server \
  serve \
  --checkpoints_dir=s3://checkpoints/sim_stack/lerobot/230226-ee/

# GR00T (desktop or H100)
CACHE_ROOT=/home/vertix docker --context desktop compose run --rm --pull always --service-ports groot-server \
  ee_rot6d \
  --checkpoints_dir=s3://checkpoints/sim_stack/groot/ee_rot6d/230226/

# OpenPI (H100)
docker --context vm-train compose run --rm --pull always --service-ports openpi-server \
  serve \
  --checkpoints_dir=s3://checkpoints/sim_stack/openpi/ee/pi05_positronic_lowmem/230226/
```

All servers expose WebSocket API on port 8000. Available presets per server:

| Server | Presets |
|--------|---------|
| `lerobot-server` | `serve`, `phail` |
| `lerobot-0_3_3-server` | `serve`, `phail`, `sim_stack` |
| `groot-server` | `serve`, `ee`, `ee_rot6d`, `phail`, `sim_stack`, ... |
| `openpi-server` | `serve`, `phail`, `sim_stack` |

### 4. Run Inference Client

```bash
# With GUI
uv run positronic-inference sim \
  --policy=.remote --policy.host=desktop --policy.port=8000 \
  --driver.show_gui

# Headless
MUJOCO_GL=egl uv run positronic-inference sim \
  --policy=.remote --policy.host=desktop --policy.port=8000 \
  --driver.show_gui=False --driver.simulation_time=10
```

## Sim Eval End-to-End

```bash
# 1. Start server in background (-d flag)
CACHE_ROOT=/home/vertix docker --context <machine> compose run -d --rm --pull always --service-ports <server-service> \
  <variant> --checkpoints_dir=<checkpoint_path>

# Wait for ready
docker --context <machine> logs --tail 5 <container_id>
# Look for: "Uvicorn running on http://0.0.0.0:8000"

# 2. Run sim episodes
CACHE_ROOT=/home/vertix docker --context <machine> compose run --rm --pull always positronic-inference \
  sim --policy=.remote --policy.host=<server_machine> --policy.port=8000 \
  --driver.num_iterations=50 --driver.simulation_time=30 \
  --output_dir=s3://inference/sim_stack_validation/<run_name>/<model_type>

# 3. View results (locally) — pass top-level dir to compare multiple runs
uv run python -m positronic.cfg.eval sim \
  --dataset.base.path=s3://inference/sim_stack_validation/<run_name> --reset_cache --https
# Opens http://localhost:5001

# 4. Clean up
docker --context <machine> stop <container_id>
```

**Naming**: `s3://inference/sim_stack_validation/<DDMMYY[-suffix]>/<model_type>/` where model_type is `lerobot`, `groot`, `openpi`, or `dreamzero`.

## Monitoring Background Jobs

```bash
grep -o '[0-9]*%' /tmp/claude/-home-vertix-dev-positronic/tasks/<task_id>.output | tail -1
tail -50 /tmp/claude/-home-vertix-dev-positronic/tasks/<task_id>.output
grep -i "error\|complete\|finished" /tmp/claude/-home-vertix-dev-positronic/tasks/<task_id>.output
```

## Common Issues

- **CUDA OOM**: Each GR00T server uses ~6GB. On 12GB GPUs (desktop), only one server at a time.
- **Port conflict**: `docker ps -a | grep -E "groot-server|openpi-server"` then `docker stop <id>`.
- **VM unreachable**: `../internal/scripts/start.sh train2` then verify SSH.
- **Headless rendering**: Use `MUJOCO_GL=egl` env var.

### Nebius Auth (Headless)

1. `nebius --no-browser --auth-timeout 5m iam whoami 2>&1` — extract auth URL
2. User clicks URL, browser redirects to `http://127.0.0.1:PORT/?code=XXX&state=YYY`
3. `curl -s "http://127.0.0.1:PORT/?code=XXX&state=YYY"` on the machine running nebius
4. Auth completes, VM scripts work
