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

# OpenPI (--pipeline.ee_frame states the EE frame the checkpoint speaks; None means the rig's `default`)
cd docker && docker compose run --rm --service-ports openpi-server ee \
  --pipeline.source.checkpoints_dir=~/checkpoints/openpi/experiment_v1/ \
  --pipeline.ee_frame=None
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

**Credentials stay out of the URL.** `--policy.headers='{"Modal-Key": "..."}'` passes auth headers for a fronted endpoint, so the URL itself is safe to paste around. Against a Nebius endpoint use `.authed_remote` or `.nebius_remote`, which build the bearer header for you (see [the Nebius workflow README](../workflows/nebius/README.md#authenticated-inference)).

**What crosses the wire is the server's call, not the client's.** A server that wants smaller frames declares `RestrictImageSize` in its rig-side stack (640x640 by default); one behind a proxy with a message-size cap declares `remote(compress_images=True)` and the rig JPEG-encodes frames before sending. A server whose checkpoint speaks a different end-effector frame declares `ChangeEEFrame` with the transform placing that frame relative to the rig's `default`, and the rig converts poses (see [End-effector frames](codecs.md#end-effector-frames)). The client builds whatever the handshake declares, and only that — connecting to a server that declares no stack fails with an error naming the version it runs. What the declared stack must achieve is checked where it matters: the harness refuses to emit an action scheduled further than `MAX_ACTION_SKEW_SEC` from now, which is what a stack that never anchored its chunk to the rig's clock produces.

> **Recording inference I/O:** Pass `--policy.recording_dir=s3://bucket/path` to write a rerun `.rrd` file per episode capturing the raw and server-side observation/action boundaries. Useful for debugging codec behavior and visualizing what the policy actually received.

## Local Inference

Load model directly on robot/simulator machine. Only ACT is supported locally (GR00T and OpenPI use remote inference).

```bash
uv run positronic-inference sim \
  --policy=@positronic.vendors.lerobot_0_3_3.policy.act_absolute \
  --policy.base.checkpoints_dir=~/checkpoints/lerobot/experiment_v1/ \
  --policy.base.checkpoint=10000
```

Use local when latency is critical (<50ms), robot has built-in GPU, or offline operation required. Use remote when GPU server is separate, models are heavy, or multiple robots share one server.

## Inference Drivers

Positronic provides two interactive drivers for managing inference episodes (see [`positronic/inference.py`](../positronic/inference.py)), plus an unattended mode:

**Unattended (automatic):** The default for `sim` — runs `--eval.trial_count=10` episodes back-to-back, each ending when the task's `timeout` expires (override with `--eval.timeout=60`, seconds per episode); optionally `--show_gui=True` for DearPyGui visualization. Useful for batch evaluation without manual intervention.

**Keyboard driver (manual):** Control inference with keyboard. Press `s` to start episode, `p` to stop and save, `r` to home the robot, `q` to quit. The default for `real`; optionally pass `--driver.show_gui=True` for DearPyGui visualization. Useful for manual evaluation and debugging.

**Eval UI driver:** Dedicated evaluation interface for policy assessment. The default for `phail` — graphical controls and metrics visualization. Useful for systematic policy evaluation with visual feedback.

## Driving a Simulated Rig

The driver and the embodiment are independent axes: `--embodiment=.sim_mujoco` puts a MuJoCo Franka behind any of the drivers above, so an operator surface can be exercised with no robot present.

```bash
# The dearpygui console against a simulated arm, with a recording in place of a served model
uv run positronic-inference phail \
  --embodiment=.sim_mujoco \
  --policy=.replay --policy.dataset_path=~/datasets/inference_logs/run1 \
  --output_dir=~/datasets/sim_console

# The same rig behind the web console instead
uv run positronic-inference phail --driver=.web --driver.task='Pick up the green cube and place it on the red cube.' \
  --embodiment=.sim_mujoco --policy=.replay --policy.dataset_path=~/datasets/inference_logs/run1
```

**An attended run is paced by the wall clock.** A simulated embodiment normally runs under a virtual clock, as fast as the machine allows; attach a driver and the world runs on wall time instead, so a second at the operator's console is a second of the episode whatever the machine's real-time factor. The simulator paces itself by the `control_period` it steps in — `.sim_mujoco` steps at 15 Hz, the rate a Franka rollout is driven at — and falls behind on a box that cannot render that fast, rather than drifting off the recorded timeline. Unattended sweeps keep the virtual clock.

**Rendering is what costs.** The physics of the Franka table scene runs ~100x real time on four CPU cores; the three camera streams are the expense, and shadows, multisampling and specular reflections are most of it under a software GL stack. `.sim_mujoco` therefore builds its scene through `low_render_quality`, which drops all three (`SetRenderQuality`) for a 7x cheaper frame — 49 ms rather than 348 ms per three-camera set at 320x240. Pass the bare loaders (`--embodiment.sim.loaders=@positronic.cfg.simulator.stack_cubes_loaders`) on a box with a GPU, or where those effects are part of what is being evaluated.

**The scene carries across episodes.** Re-randomizing a scene is an eval task's job (`Task.reset`), and an attended run has no task, so finishing an episode homes the arm and leaves the objects where they were: the second episode of a console session starts from the first one's end state. Repeated trials from a fresh scene are the unattended path (`positronic eval run`), which reseeds per trial.

**The replay policy** (`--policy=.replay --policy.dataset_path=<dataset>`, plus `--policy.episode=N`) plays a recorded episode's commands back through the policy interface, so the arm moves with nothing served. Every episode replays the recording from its first waypoint at the cadence it was recorded at; when it runs out the rig holds. The recording must be one the replaying embodiment can execute — a run recorded in the same sim is the faithful case, and the recorded action space is re-issued verbatim (joint targets replay exactly, pose targets go back through the driver's IK, and a recorded reset is reissued as the command it was). Every command channel the recording carries replays under the name the embodiment commands it by, so a multi-arm rig's `robot_command.left` / `.right` and their grips all play. Arm and gripper each keep the timing they were recorded with rather than the other's cadence, and a recording whose action space changed part-way through replays both stretches — provided both are absolute. Delta commands are not replayable at all, since a delta means something only against the state it was issued from, so a recording carrying one is refused outright rather than played in part.

## Running Several Policies in One Run

`--policy=.production` (and the `phail` default `.phail_multiple`, which adds a balancing sampler) routes each episode to one of several named endpoints, so one run compares policies under identical conditions and the operator cannot tell which is driving. An endpoint is a served checkpoint given as its URL, or a mapping declaring what it is:

```bash
uv run positronic-inference phail --embodiment=.sim_mujoco \
  --policy.endpoints='{"arm_a": {"kind": "replay", "dataset": "s3://…/rollouts/", "episode": 0},
                       "arm_b": {"kind": "replay", "dataset": "s3://…/rollouts/", "episode": 1},
                       "arm_c": "wss://host/api/v1/session"}' \
  --output_dir=s3://…/blind/
```

A `remote` endpoint takes a `url` and a `replay` endpoint a `dataset` plus optionally the `episode` within it; the kind defaults to `remote`, so a bare URL string is the short form. The kind is declared rather than read off the locator, because a URL may be written without a scheme (`notebook:8000` is one) and no rule tells that apart from a relative path to a recording. A run needs no GPU and nothing served when every endpoint is a replay, which is what lets a simulated rig exercise the whole comparison.

Passing the whole mapping replaces the endpoints the config carries; the per-key form (`--policy.endpoints.arm_a=…`) adds to them, which is how a run ends up sampling an endpoint nobody asked for.

Give a `dataset` in the whole-mapping form as an absolute path or a URI. A leading dot is configuronic's relative-import sigil inside an override value, so `"./run"` there is read as a config to import and the override raises before the run starts. A relative one is reached by the per-key form, which resolves against the value it replaces rather than against the config:

```bash
  --policy.endpoints='{"a": {"kind": "replay", "dataset": "unset"}}' \
  --policy.endpoints.a.dataset=./run
```

**The names address endpoints on the command line and do not reach the recording.** Each episode records the identity its policy reports — a served endpoint's checkpoint path under `inference.policy.server.checkpoint_path`, a replay's `inference.policy.replay.dataset_path` and `.episode` — which is what tells the episodes of one endpoint from another's after the fact. A policy reporting no checkpoint path of its own is keyed by what it names itself: a replay by its dataset and episode, so reordering the mapping does not move a resumed run's counts from one endpoint to another. Two replay endpoints on the same dataset and episode are refused for the same reason — nothing tells their episodes apart.

## Following a Run From Its Log

The harness logs one line per episode boundary at INFO, so a watcher can follow a run it has no other view of — no console endpoint, no dataset scan:

```
harness: directive start id=<n> task=<task>
harness: directive finish id=<n> outcome=<saved|discarded|aborted>
harness: run finish
```

`id` counts episodes within the run from 0, pairing a start with its finish, and is the same number the episode's telemetry span carries as `episode.index` — so a run log and a `--timing` report name the same episode. `saved` means the harness finalized the episode and handed it to the recorder, `discarded` that the operator dropped it, `aborted` that a failure abandoned it mid-flight. `run finish` is logged only when the harness's own loop returns, so a failure inside the harness is distinguishable from a completed run. It does **not** prove the run was healthy: a background producer or recorder that dies is caught in its own process and reported to the harness as an ordinary stop, so the harness finalizes the live episode and logs `run finish` exactly as it would for a clean end. A watcher that must tell those apart reads the run's stderr or checks the episodes landed; the log alone cannot. The task is last on the line and free-form to its end.

The outcomes report what the harness did, which is all it knows: on a real rig the recorder runs as a background subprocess that writes the artifact after the fact and reports nothing back, so `saved` can precede the episode appearing on disk. A watcher that needs the artifact reads the dataset; the log tells it which episodes to expect and when the run ended.

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
