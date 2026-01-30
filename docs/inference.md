# Inference Guide

This guide covers deploying trained policies for evaluation and production use in Positronic.

## Overview

Positronic supports two inference patterns:
- **Local inference**: Load model directly on the robot/simulator machine
- **Remote inference**: Connect to inference server via WebSocket (`.remote` policy)

Both patterns use the same hardware/simulator code — only the policy configuration changes.

## Remote Inference (Recommended)

Remote inference uses Positronic's unified WebSocket protocol to connect ANY hardware to ANY model.

### Architecture

```
Hardware/Simulator (Client)
    ↓
RemotePolicy (.remote)
    ↓
WebSocket Protocol v1
    ↓
Inference Server
    ├─ LeRobot Server
    ├─ GR00T Server
    └─ OpenPI Server
```

**Key benefit:** Run heavy models remotely on capable GPU hardware, separate from the robot/simulator machine.

### Starting an Inference Server

See [Training Workflow - Step 3](training-workflow.md#step-3-serve-inference) for detailed server startup instructions.

**Quick reference:**

```bash
# LeRobot
cd docker && docker compose run --rm --service-ports lerobot-server \
  --checkpoints_dir=~/checkpoints/lerobot/experiment_v1/ \
  --codec=@positronic.vendors.lerobot.codecs.eepose_absolute

# GR00T
cd docker && docker compose run --rm --service-ports groot-server \
  ee_rot6d_joints \
  --checkpoints_dir=~/checkpoints/groot/experiment_v1/

# OpenPI
cd docker && docker compose run --rm --service-ports openpi-server \
  --checkpoints_dir=~/checkpoints/openpi/experiment_v1/ \
  --codec=@positronic.vendors.openpi.codecs.eepose_absolute
```

**Check server status:**
```bash
curl http://localhost:8000/api/v1/models
# Response: {"models": ["10000", "20000", "30000"]}
```

### Running Remote Inference

**Simulation:**
```bash
uv run positronic-inference sim \
  --driver.simulation_time=60 \
  --driver.show_gui=True \
  --output_dir=~/datasets/inference_logs/experiment_v1 \
  --policy=.remote \
  --policy.host=localhost \
  --policy.port=8000
```

**Hardware (Franka):**
```bash
uv run positronic-inference real \
  --output_dir=~/datasets/inference_logs/franka_eval \
  --policy=.remote \
  --policy.host=gpu-server \
  --policy.port=8000
```

### Remote Policy Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--policy` | Policy type | N/A | `.remote` |
| `--policy.host` | Server machine hostname/IP | `localhost` | `desktop`, `192.168.1.100` |
| `--policy.port` | Server port | `8000` | `8000`, `8001` |
| `--policy.resize` | Client-side image resize (bandwidth optimization) | `640` | `480`, `None` (no resize) |
| `--policy.model_id` | Specific checkpoint ID | `None` (latest) | `10000`, `20000` |

### How Remote Inference Works

1. **Client connects** to WebSocket endpoint (`ws://host:port/api/v1/session`)
2. **Server sends metadata** (checkpoint ID, codec info, action dimensions)
3. **Inference loop** at fixed rate (15-30 Hz):
   - Client encodes observation (robot state, images) → dict
   - Client sends serialized observation via WebSocket
   - Server decodes observation → model input format
   - Server runs forward pass → model output
   - Server decodes action → robot command format
   - Server sends serialized action back to client
   - Client applies action to robot/simulator
4. **Loop continues** until episode ends or timeout

## Local Inference

Load the model directly on the robot/simulator machine.

### Example (LeRobot ACT)

```bash
uv run positronic-inference sim \
  --driver.simulation_time=60 \
  --driver.show_gui=True \
  --output_dir=~/datasets/inference_logs/local_eval \
  --policy=@positronic.cfg.policy.act_absolute \
  --policy.base.checkpoints_dir=~/checkpoints/lerobot/experiment_v1/ \
  --policy.base.checkpoint=10000
```

### Local Policy Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--policy` | Policy configuration | `@positronic.cfg.policy.act_absolute` |
| `--policy.base.checkpoints_dir` | Path to experiment directory | `~/checkpoints/lerobot/experiment_v1/` |
| `--policy.base.checkpoint` | (Optional) Specific checkpoint ID | `10000` (default: latest) |

**Available local policy configs:**
- `@positronic.cfg.policy.act_absolute` — LeRobot ACT with absolute actions
- Custom policies can be added in `positronic/cfg/policy.py`

**Note:** Only ACT is supported in local inference. GR00T and OpenPI use remote inference.

## Recording Inference Runs

Use `--output_dir` to record inference runs as Positronic datasets for replay and analysis.

```bash
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=localhost \
  --output_dir=~/datasets/inference_logs/experiment_v1_eval_run1
```

**What gets recorded:**
- Robot state (joint positions, EE pose)
- Camera feeds
- Actions sent to robot
- Gripper commands
- Timing information

**Replay in Positronic server:**
```bash
uv run positronic-server \
  --dataset.path=~/datasets/inference_logs/experiment_v1_eval_run1 \
  --port=5001
```

## Evaluation Patterns

### Manual Evaluation

1. **Run inference with recording**:
   ```bash
   uv run positronic-inference sim \
     --policy=.remote \
     --policy.host=localhost \
     --driver.simulation_time=60 \
     --output_dir=~/datasets/eval/checkpoint_10000_run1
   ```

2. **Review in Positronic server**:
   ```bash
   uv run positronic-server --dataset.path=~/datasets/eval/checkpoint_10000_run1
   ```

3. **Score manually**:
   - Success: Task completed correctly
   - Partial: Task attempted but failed
   - Failure: No progress or collision

4. **Repeat** for multiple runs (10-50 trials typical)

5. **Calculate metrics**:
   - Success rate: `successes / total_trials`
   - Average completion time (from timing data)
   - Qualitative notes (common failure modes)

### Checkpoint Comparison

Compare multiple checkpoints using the same evaluation scenarios:

```bash
# Checkpoint 10000
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=localhost \
  --policy.model_id=10000 \
  --output_dir=~/datasets/eval/ckpt_10000 \
  --driver.simulation_time=60

# Checkpoint 20000
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=localhost \
  --policy.model_id=20000 \
  --output_dir=~/datasets/eval/ckpt_20000 \
  --driver.simulation_time=60
```

**Switch server checkpoint** without restarting:
```bash
# Server automatically loads requested checkpoint via model_id
curl http://localhost:8000/api/v1/session/10000  # WebSocket endpoint
```

### Batch Evaluation

Use `utilities/validate_server.py` for batch evaluation of checkpoints.

### Local vs Remote Trade-offs

| Aspect | Local | Remote |
|--------|-------|--------|
| **Latency** | Lower (no network) | Higher (network + serialization) |
| **Hardware** | Requires GPU on robot machine | GPU on separate server |
| **Deployment** | Tight coupling, single machine | Distributed, flexible |
| **Model swapping** | Requires code changes | Change server, keep client |
| **Use case** | Lab testing, low-latency control | Production, expensive GPUs, multi-robot |

**When to use remote:**
- GPU server is separate from robot/simulator
- Heavy models require powerful GPU (OpenPI needs ~62GB, GR00T needs ~8GB)
- Multiple robots connecting to single inference server
- Need flexibility to compare checkpoints or models

**When to use local:**
- Latency is critical (<50ms required)
- Robot control machine has built-in GPU
- Single deployment, no need for flexibility
- Offline operation required

## Iteration Workflow

**Typical workflow after initial training:**

1. **Evaluate checkpoint** on test scenarios
   ```bash
   uv run positronic-inference sim \
     --policy=.remote \
     --output_dir=~/datasets/eval/initial
   ```

2. **Identify failure modes** in Positronic server
   - Which scenarios fail?
   - What does policy do wrong?

3. **Collect targeted demonstrations** for failure modes
   ```bash
   uv run positronic-data-collection sim \
     --output_dir=~/datasets/additional_demos
   ```

4. **Append to dataset and retrain**
   ```bash
   cd docker && docker compose run --rm positronic-to-lerobot append \
     --output_dir=~/datasets/lerobot/my_task \
     --dataset.dataset.path=~/datasets/additional_demos

   cd docker && docker compose run --rm lerobot-train \
     --input_path=~/datasets/lerobot/my_task \
     --exp_name=iteration_v2
   ```

5. **Re-evaluate** and repeat

## See Also

- [Training Workflow](training-workflow.md) — Preparing data and training models
- [Codecs Guide](codecs.md) — Understanding observation/action encoding
- [Offboard README](../positronic/offboard/README.md) — Unified WebSocket protocol details
- Vendor-specific guides: [OpenPI](../positronic/vendors/openpi/README.md) | [GR00T](../positronic/vendors/gr00t/README.md) | [LeRobot](../positronic/vendors/lerobot/README.md)
