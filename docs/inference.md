# Inference Guide

Deploy trained policies for evaluation and production use. Positronic supports local inference (model loaded on robot/simulator machine) and inference with remote server (model runs on separate GPU server via WebSocket).

## Inference with Remote Server

Positronic's unified WebSocket protocol connects any hardware to any model (LeRobot, GR00T, OpenPI). The key benefit is running heavy models on powerful GPU hardware (OpenPI needs ~62GB, GR00T ~8GB) separate from the robot/simulator machine.

**Start inference server:**
```bash
# LeRobot (SmolVLA — 0.4.x)
cd docker && docker compose run --rm --service-ports lerobot-server \
  --checkpoints_dir=~/checkpoints/lerobot/experiment_v1/ \
  --codec=@positronic.vendors.lerobot.codecs.ee

# LeRobot (ACT — 0.3.3)
cd docker && docker compose run --rm --service-ports lerobot-0_3_3-server \
  --checkpoints_dir=~/checkpoints/lerobot/experiment_v1/ \
  --codec=@positronic.vendors.lerobot_0_3_3.codecs.ee

# GR00T (pre-configured variant)
cd docker && docker compose run --rm --service-ports groot-server \
  ee_rot6d_joints \
  --checkpoints_dir=~/checkpoints/groot/experiment_v1/

# OpenPI
cd docker && docker compose run --rm --service-ports openpi-server \
  --checkpoints_dir=~/checkpoints/openpi/experiment_v1/ \
  --codec=@positronic.vendors.openpi.codecs.ee
```

Check server: `curl http://localhost:8000/api/v1/models` returns available model IDs.

**Run inference:**
```bash
# Simulation
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=localhost \
  --output_dir=~/datasets/inference_logs/exp_v1

# Hardware
uv run positronic-inference real \
  --policy=.remote \
  --policy.host=gpu-server \
  --output_dir=~/datasets/inference_logs/franka_eval
```

**Remote policy parameters:** `--policy.host` (server hostname/IP), `--policy.port` (default 8000), `--policy.model_id` (specific checkpoint, default latest), `--policy.resize` (client-side image resize for bandwidth optimization).

> **Recording inference I/O:** Pass `--policy.recording_dir=s3://bucket/path` to `.remote` or `.weighted_remote` to write a rerun `.rrd` file per episode capturing the raw and server-side observation/action boundaries. Useful for debugging codec behavior and visualizing what the policy actually received.

**Multi-checkpoint sampling:** To evaluate several models at once, sampling configs route each episode to a different policy. Use `--policy=@positronic.cfg.policy.production` to combine per-vendor `.weighted_remote` endpoints, optionally with a balanced sampler and grouping fields. The `phail` subcommand wires this up end-to-end, defaulting to the `phail_multiple` policy (a `production` preset) and the `eval_ui` driver.

## Local Inference

Load model directly on robot/simulator machine. Only ACT is supported locally (GR00T and OpenPI use remote inference).

```bash
uv run positronic-inference sim \
  --policy=@positronic.cfg.policy.act_absolute \
  --policy.base.checkpoints_dir=~/checkpoints/lerobot/experiment_v1/ \
  --policy.base.checkpoint=10000
```

Use local when latency is critical (<50ms), robot has built-in GPU, or offline operation required. Use remote when GPU server is separate, models are heavy, or multiple robots share one server.

## Inference Drivers

Positronic provides three drivers for managing inference episodes (see [`positronic/inference.py`](../positronic/inference.py)):

**Timed driver (automatic):** Runs inference automatically for a fixed duration per episode — the task's `timeout` (override with `--eval.timeout=60`, seconds per episode); also set `--trial_count=10` (number of episodes), and optionally `--show_gui=True` for DearPyGui visualization. The default for `sim`. Useful for batch evaluation without manual intervention.

**Keyboard driver (manual):** Control inference with keyboard. Press `s` to start episode, `p` to stop and save, `r` to home the robot, `q` to quit. The default for `real`; optionally pass `--driver.show_gui=True` for DearPyGui visualization. Useful for manual evaluation and debugging.

**Eval UI driver:** Dedicated evaluation interface for policy assessment. The default for `phail` — graphical controls and metrics visualization. Useful for systematic policy evaluation with visual feedback.

## Recording and Replay

Specify `--output_dir` to record runs as Positronic datasets. Recorded data includes robot state, camera feeds, actions, gripper commands, and timing information.

Replay recorded runs: `uv run positronic-server --dataset.path=~/datasets/inference_logs/run1 --port=5001` and open `http://localhost:5001` to review episodes, identify failure modes, and extract clips for dataset augmentation.

## Evaluation Workflow

Run inference with recording, review in Positronic server, score manually (success/partial/failure), repeat for 10-50 trials, calculate success rate and note common failure modes. Compare checkpoints by running inference with different `--policy.model_id` values. For batch evaluation, use [`utilities/validate_server.py`](../utilities/validate_server.py).

**Iteration:** Evaluate checkpoint → identify failures in server → collect targeted demos for failure modes → append to dataset → retrain → re-evaluate. Convergence typically occurs after 3-5 iterations.

## See Also

- [Training Workflow](training-workflow.md) – Preparing data and training
- [Codecs Guide](codecs.md) – Observation/action encoding
- [Offboard README](../positronic/offboard/README.md) – WebSocket protocol
- Vendor guides: [OpenPI](../positronic/vendors/openpi/README.md) | [GR00T](../positronic/vendors/gr00t/README.md) | [SmolVLA](../positronic/vendors/lerobot/README.md) | [LeRobot ACT](../positronic/vendors/lerobot_0_3_3/README.md)
