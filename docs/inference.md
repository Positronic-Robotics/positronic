# Inference Guide

Deploy trained policies for evaluation and production use. Positronic supports local inference (model loaded on robot/simulator machine) and inference with remote server (model runs on a separate GPU server, over a websocket or gRPC).

## Inference with Remote Server

Positronic's unified session protocol connects any hardware to any model (LeRobot, GR00T, OpenPI); the same frames cross either wire, a websocket or gRPC. A heavy model (OpenPI needs ~62GB, GR00T ~8GB) runs on GPU hardware separate from the robot/simulator machine.

Each server carries a full **policy pipeline** — one chain naming the rig-side stack, the `remote` split marker, the server-side codec, and the model source that loads checkpoints (see `positronic.policy.spec`). The server runs the half right of the marker and declares the half left of it in its handshake; the client builds the declared stack automatically. Vendors ship their pipelines by name, and every name is a server subcommand — `groot-server droid` launches that one. The available names are listed in each vendor's README.

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
cd docker && docker compose run --rm --service-ports -v "$PWD/groot-data:/data" groot-server droid \
  --pipeline.source.model_source=/data/checkpoints/experiment_v1/

# OpenPI (--pipeline.ee_frame states the EE frame the checkpoint speaks; None means the rig's `default`)
cd docker && docker compose run --rm --service-ports openpi-server ee \
  --pipeline.source.checkpoints_dir=~/checkpoints/openpi/experiment_v1/ \
  --pipeline.ee_frame=None
```

Check server: `curl http://localhost:8000/api/v1/models` returns available model IDs.

**Run inference:**
```bash
# Simulation
uv run positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=.remote \
  --policy.address.host=localhost --policy.address.port=8000 \
  --output_dir=~/datasets/inference_logs/exp_v1

# Hardware — the same command against a rig's eval
uv run positronic eval run --eval=.real.droid.pick_place \
  --policy=.remote \
  --policy.address.host=gpu-server --policy.address.port=8000 \
  --output_dir=~/datasets/inference_logs/franka_eval
```

`--eval` names what runs: a whole benchmark, a suite, or one task. [Evaluation](evaluation.md) lists the targets and the flags that shape a sweep — `--eval.trial_count`, `--charge_inference_time`, `--timing`. (`positronic-inference sim` is a shorthand for the same command with `--eval=.sim.positronic.stack_cubes` fixed.)

**Flags name the endpoint.** `--policy.wire` is the transport by name — `websocket`, `websocket_tls`, `websocket_unix`, `grpc` or `grpc_tls`; the `_tls` members dial a TLS front, and `websocket_unix` a Unix socket (below). Each wire then takes its own address, and `--policy.address.*` fills it: `--policy.address.host` and `--policy.address.port` for a network wire (`8000` is every vendor server's websocket default; a TLS front answers on `443`), or `--policy.address=@positronic.cfg.policy.socket_address --policy.address.uds=…` for `websocket_unix`, which names no host and no port. `--policy.address.model` is the checkpoint, and naming none serves the one the server pinned at startup. `--policy.address.query` carries the session params:

```bash
uv run positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=.remote \
  --policy.wire=websocket_tls --policy.address.host=gpu-server --policy.address.port=443 \
  --policy.address.model=checkpoint-20000 --policy.address.query='codec.fps=10&local.pad_start=false'
```

**A Unix socket reaches a server on the same machine.** `--policy.wire=websocket_unix --policy.address=@positronic.cfg.policy.socket_address --policy.address.uds=/run/policy.sock` dials the socket a server bound with `--websocket.served_address=@positronic.offboard.server.socket_at --websocket.served_address.uds=/run/policy.sock`, over no network. `--policy.address.model` and `--policy.address.query` name a checkpoint and session params as they do on any other wire; this wire's address has no host and no port to fill. Use this carrier for a policy process that runs beside the harness and has no network interface of its own.

**Credentials stay off the command line.** A token rides a header instead. It stays off the command line too: `save_run_metadata()` writes `sys.argv` beside the run's episodes. Three policy configs build the header:

- `.authed_remote` — a bearer token read from `AUTH_TOKEN`, which it raises about when that is unset. Every endpoint [`workflows/nebius/serve.sh`](../workflows/nebius/README.md) creates is gated this way, whether the server checks the token itself or a proxy in front of it does.
- `.nebius_remote` — the same header, with the Nebius token fetched for you (see [the Nebius workflow README](../workflows/nebius/README.md#authenticated-inference)).
- `.file_authed_remote` — any header set, read from the JSON object in the file at `--policy.headers.path`.

```bash
uv run positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=.file_authed_remote \
  --policy.wire=websocket_tls --policy.address.host=<endpoint-managed-host> --policy.address.port=443 \
  --policy.headers.path=~/.config/endpoint/headers.json \
  --output_dir=~/datasets/inference_logs/exp_v1
```

**Session parameters** are `--policy.address.query`, a query string: the server applies them as overrides to its pipeline config, so you can tune the served pipeline without restarting the server. Keys are dotted paths into that config and values are JSON literals, forwarded verbatim so they arrive exactly as written (`fps=10`, `pad=false`, `name="s3"`).

The model source (`checkpoints_dir`, `checkpoint`, device...) is fixed at server launch — `source.*` params are rejected; name a checkpoint with `--policy.address.model` instead. Bad params fail at connect with a clear server error. Full rules in the [Offboard README](../positronic/offboard/README.md).

**What crosses the wire is the server's call, not the client's.** A server that wants smaller frames declares `RestrictImageSize` in its rig-side stack (640x640 by default); one behind a proxy with a message-size cap declares `remote(compress_images=True)` and the rig JPEG-encodes frames before sending. A server whose checkpoint speaks a different end-effector frame declares `ChangeEEFrame` with the transform placing that frame relative to the rig's `default`, and the rig converts poses (see [End-effector frames](codecs.md#end-effector-frames)). The client builds whatever the handshake declares, and only that — connecting to a server that declares no stack fails with an error naming the version it runs. What the declared stack must achieve is checked where it matters: the harness refuses to emit an action scheduled further than `MAX_ACTION_SKEW_SEC` from now, which is what a stack that never anchored its chunk to the rig's clock produces.

> **Recording inference I/O:** Pass `--policy.recording_dir=s3://bucket/path` to write a rerun `.rrd` file per episode capturing the raw and server-side observation/action boundaries. Useful for debugging codec behavior and visualizing what the policy actually received.

## Local Inference

Load model directly on robot/simulator machine. Only ACT is supported locally (GR00T and OpenPI use remote inference).

```bash
uv run positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=@positronic.vendors.lerobot_0_3_3.policy.act_absolute \
  --policy.base.checkpoints_dir=~/checkpoints/lerobot/experiment_v1/ \
  --policy.base.checkpoint=10000
```

Use local when latency is critical (<50ms), robot has built-in GPU, or offline operation required. Use remote when GPU server is separate, models are heavy, or multiple robots share one server.

## Who Decides Episode Boundaries

Something has to say when an episode starts and when it finishes. There are two answers, one command each:

**Unattended — `positronic eval run`:** a driver walks the eval's tasks, `--eval.trial_count=10` episodes back-to-back. Each ends when its benchmark reports the task done, or when the task's timeout expires (`--eval.timeout=60`, seconds per episode). Batch evaluation with nobody in the loop.

**Keyboard — `positronic-inference real`:** press `s` to start an episode, `p` to stop and save, `q` to quit. Headless — it renders nothing — and it takes `--next_task`, `--embodiment`, `--policy` and `--output_dir`. `--next_task` names the config that makes each trial, one per press. The default draws a new start pose for every one of them. Set the goal with `--next_task.instruction="..."`. Manual evaluation and debugging on hardware.

Anything richer — a web console, a foot pedal, a rig UI — is a driver of its own rather than a plug-in. A driver is any control system with a `perform_task` caller, and it brings the policy and the output path: each ask carries the session the episode runs on and names where it records. `run_world` builds the world around it — the harness, the recorder, the devices, and every wire between them. `KeyboardOperator` in [`positronic/inference.py`](../positronic/inference.py) is the worked example, in about thirty lines.

## Recording and Replay

Specify `--output_dir` to record runs as Positronic datasets. Recorded data includes robot state, camera feeds, actions, gripper commands, and timing information.

Replay recorded runs: `uv run positronic-server --dataset.path=~/datasets/inference_logs/run1 --port=5001` and open `http://localhost:5001` to review episodes, identify failure modes, and extract clips for dataset augmentation.

## Evaluation Workflow

Run inference with recording, review in Positronic server, score manually (success/partial/failure), repeat for 10-50 trials, calculate success rate and note common failure modes. Compare checkpoints by naming each with `--policy.address.model`. For batch evaluation, use [`utilities/validate_server.py`](../utilities/validate_server.py).

**Iteration:** Evaluate checkpoint → identify failures in server → collect targeted demos for failure modes → append to dataset → retrain → re-evaluate. Convergence typically occurs after 3-5 iterations.

## See Also

- [Training Workflow](training-workflow.md) – Preparing data and training
- [Codecs Guide](codecs.md) – Observation/action encoding
- [Offboard README](../positronic/offboard/README.md) – the session protocol and both wires
- Vendor guides: [OpenPI](../positronic/vendors/openpi/README.md) | [GR00T](../positronic/vendors/gr00t/README.md) | [SmolVLA](../positronic/vendors/lerobot/README.md) | [LeRobot ACT](../positronic/vendors/lerobot_0_3_3/README.md)
