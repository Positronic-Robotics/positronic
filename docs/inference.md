# Inference Guide

Deploy trained policies for evaluation and production use. Positronic supports local inference (model loaded on robot/simulator machine) and inference with remote server (model runs on separate GPU server via WebSocket).

## Inference with Remote Server

Positronic's unified WebSocket protocol connects any hardware to any model (LeRobot, GR00T, OpenPI). The key benefit is running heavy models on powerful GPU hardware (OpenPI needs ~62GB, GR00T ~8GB) separate from the robot/simulator machine.

Each server carries a full **policy pipeline** — one chain naming the rig-side stack, the `remote` split marker, the server-side codec, and the model source that loads checkpoints (see `positronic.policy.spec`). The server runs the half right of the marker and declares the half left of it in its handshake; the client builds the declared stack automatically. Vendors ship their pipelines by name, and every name is a server subcommand — `groot-server ee_rot6d_joints` launches that one. The available names are listed in each vendor's README.

**Start inference server:**
```bash
# The subcommand names the pipeline; everything the model is lives inside it
# LeRobot (SmolVLA — 0.4.x)
cd docker && docker compose run --rm --service-ports lerobot-server ee \
  --pipeline.source.checkpoints_dir=~/checkpoints/lerobot/experiment_v1/

# LeRobot (ACT — 0.3.3)
cd docker && docker compose run --rm --service-ports lerobot-0_3_3-server ee \
  --pipeline.source.checkpoints_dir=~/checkpoints/lerobot/experiment_v1/

# GR00T
cd docker && docker compose run --rm --service-ports groot-server ee_rot6d_joints \
  --pipeline.source.checkpoints_dir=~/checkpoints/groot/experiment_v1/

# OpenPI
cd docker && docker compose run --rm --service-ports openpi-server ee \
  --pipeline.source.checkpoints_dir=~/checkpoints/openpi/experiment_v1/
```

Check server: `curl http://localhost:8000/api/v1/models` returns available model IDs.

**Run inference:**
```bash
# Simulation
uv run positronic-inference sim \
  --policy=.remote \
  --policy.url=localhost:8000 \
  --output_dir=~/datasets/inference_logs/exp_v1

# Hardware
uv run positronic-inference real \
  --policy=.remote \
  --policy.url=gpu-server:8000 \
  --output_dir=~/datasets/inference_logs/franka_eval
```

**One URL is the whole endpoint.** `--policy.url` carries the host, port, TLS, model id, and session params, so a server can be handed out as a single string:

```bash
uv run positronic-inference sim \
  --policy=.remote \
  --policy.url='https://gpu-server/api/v1/session/checkpoint-20000?codec.fps=10&local.pad_start=false'
```

Accepted forms: `host`, `host:port`, and `https://host[:port][/api/v1/session[/<model_id>]]` (`http`, `ws` and `wss` work too), each with an optional query. `https`/`wss` enable TLS. An omitted port is the scheme's own — 443 for TLS and 80 otherwise — so name the port a server listens on (`:8000` for every vendor server's default). Naming no model id serves the checkpoint the server pinned at startup.

**Credentials stay out of the URL**, which is meant to be safe to paste around: they ride headers instead. A server gated on a bearer token — whether it checks the token itself, as every endpoint [`workflows/nebius/serve.sh`](../workflows/nebius/README.md) creates does, or sits behind a proxy that checks it — is reached with `.authed_remote`, which reads the token from `AUTH_TOKEN` and raises if it is unset. For any other scheme, name the headers yourself: `--policy.headers='{"Modal-Key": "..."}'`.

```bash
uv run positronic-inference sim \
  --policy=.authed_remote \
  --policy.url=https://<endpoint-managed-url> \
  --output_dir=~/datasets/inference_logs/exp_v1
```

**Session parameters** are the URL's query string: the server applies them as overrides to its pipeline config, so you can tune the served pipeline without restarting the server. Keys are dotted paths into that config and values are JSON literals, forwarded verbatim so they arrive exactly as written (`fps=10`, `pad=false`, `name="s3"`).

The model source (`checkpoints_dir`, `checkpoint`, device...) is fixed at server launch — `source.*` params are rejected; name a checkpoint in the URL path instead. Bad params fail at connect with a clear server error. Full rules in the [Offboard README](../positronic/offboard/README.md).

**What crosses the wire is the server's call, not the client's.** A server that wants smaller frames declares `RestrictImageSize` in its rig-side stack (640x640 by default); one behind a proxy with a message-size cap declares `remote(compress_images=True)` and the rig JPEG-encodes frames before sending. A server whose checkpoint speaks a different end-effector frame declares `ChangeEEFrame` with the transform placing that frame relative to the rig's `default`, and the rig converts poses (see [End-effector frames](codecs.md#end-effector-frames)). The client builds whatever the handshake declares — a server that declares nothing gets the standard `ChunkedSchedule`.

`--policy.local=@...` and `--policy.compress_images` are deprecated stand-ins for a server too old to declare either; against a server that does declare, they raise rather than quietly winning. A server that declares no stack at all but still reports `image_sizes` gets a third stand-in: the client bounds frames to those sizes, logging what it did. All three go away once every server declares (see [#514](https://github.com/Positronic-Robotics/positronic/issues/514)).

> **Recording inference I/O:** Pass `--policy.recording_dir=s3://bucket/path` to write a rerun `.rrd` file per episode capturing the raw and server-side observation/action boundaries. Useful for debugging codec behavior and visualizing what the policy actually received.

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

Positronic provides two interactive drivers for managing inference episodes (see [`positronic/inference.py`](../positronic/inference.py)), plus an unattended mode:

**Unattended (automatic):** The default for `sim` — runs `--eval.trial_count=10` episodes back-to-back, each ending when the task's `timeout` expires (override with `--eval.timeout=60`, seconds per episode); optionally `--show_gui=True` for DearPyGui visualization. Useful for batch evaluation without manual intervention.

**Keyboard driver (manual):** Control inference with keyboard. Press `s` to start episode, `p` to stop and save, `r` to home the robot, `q` to quit. The default for `real`; optionally pass `--driver.show_gui=True` for DearPyGui visualization. Useful for manual evaluation and debugging.

**Eval UI driver:** Dedicated evaluation interface for policy assessment. The default for `phail` — graphical controls and metrics visualization. Useful for systematic policy evaluation with visual feedback.

## Recording and Replay

Specify `--output_dir` to record runs as Positronic datasets. Recorded data includes robot state, camera feeds, actions, gripper commands, and timing information.

Replay recorded runs: `uv run positronic-server --dataset.path=~/datasets/inference_logs/run1 --port=5001` and open `http://localhost:5001` to review episodes, identify failure modes, and extract clips for dataset augmentation.

## Evaluation Workflow

Run inference with recording, review in Positronic server, score manually (success/partial/failure), repeat for 10-50 trials, calculate success rate and note common failure modes. Compare checkpoints by pointing `--policy.url` at different `/api/v1/session/<checkpoint>` paths. For batch evaluation, use [`utilities/validate_server.py`](../utilities/validate_server.py).

**Iteration:** Evaluate checkpoint → identify failures in server → collect targeted demos for failure modes → append to dataset → retrain → re-evaluate. Convergence typically occurs after 3-5 iterations.

## See Also

- [Training Workflow](training-workflow.md) – Preparing data and training
- [Codecs Guide](codecs.md) – Observation/action encoding
- [Offboard README](../positronic/offboard/README.md) – WebSocket protocol
- Vendor guides: [OpenPI](../positronic/vendors/openpi/README.md) | [GR00T](../positronic/vendors/gr00t/README.md) | [SmolVLA](../positronic/vendors/lerobot/README.md) | [LeRobot ACT](../positronic/vendors/lerobot_0_3_3/README.md)
