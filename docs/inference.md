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

## Who Decides Episode Boundaries

Something has to say when an episode starts and when it finishes. `positronic-inference` ships two commands, one per answer (see [`positronic/inference.py`](../positronic/inference.py)):

**Unattended (`sim`):** a driver walks the eval's tasks — `--eval.trial_count=10` episodes back-to-back, each ending when its task's `timeout` expires (override with `--eval.timeout=60`, seconds per episode). Batch evaluation with nobody in the loop.

**Keyboard (`real`):** press `s` to start an episode, `p` to stop and save, `q` to quit. Headless — it renders nothing — and it takes `--next_task`, `--embodiment`, `--policy` and `--output_dir`. `--next_task` names the config that makes each trial, one per press. The default draws a new start pose for every one of them. Set the goal with `--next_task.instruction="..."`. Manual evaluation and debugging on hardware.

### A console of your own

Anything richer — a web console, a foot pedal, a rig UI — is a binary of its own rather than a plug-in: it composes its own `pimm.World` out of the public pieces, and nothing in the library needs to know which surface is driving. The shape, for a **hardware** embodiment:

```python
from contextlib import nullcontext

import pimm
from positronic import wire
from positronic.cli.eval.run import prepare_output_dir
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.policy.harness import Harness

# The setup under the `try` can raise — `prepare_output_dir` syncs a directory and snapshots
# sources into it, `LocalDatasetWriter` scans the one it is given — and the policy is yours to
# close from the moment you first touch it.
try:
    # `None` where the run records nothing, which is why the writer is a nullcontext below.
    output_dir = prepare_output_dir(output_dir)
    harness = Harness(policy, embodiment)
    console = MyConsole()  # asks `perform_task` for a `Task` per episode and emits its terminal on `done`

    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as writer, pimm.World() as world:
        ds_agent = wire.wire_embodiment(world, harness, embodiment, writer, done=console.done)
        world.connect(console.perform_task, harness.perform_task)
        if ds_agent is not None:
            world.connect(harness.ds_command, ds_agent.command)
        world.run([harness, console], [*embodiment.control_systems, ds_agent])
finally:
    policy.close()
```

**A simulated embodiment is not this.** Three things change together, and a copy of the shape above records a sim run against the wall clock: the world takes `virtual_time=True`, `wire_embodiment` takes `TimeMode.MESSAGE`, and the producers are scheduled in the foreground beside the harness rather than as background processes, so one scheduler round is one control period. `_run_world` in [`positronic/cli/eval/run.py`](../positronic/cli/eval/run.py) is the worked version, including why the ordering within a round is what it is.

The world stops when any of its control systems returns, so the console ends the run by returning from its loop. To show the cameras, connect every observation whose name starts with `positronic.keys.IMAGE_PREFIX` into a viewer's `cameras` — `positronic.gui.dpg_ui()` is one, and the naming convention is what identifies a camera on the wire. To let the operator jog the arm between episodes, emit robot commands into `harness.manual_command`, which the harness applies only while idle. To count down an episode's remaining time, read `harness.deadline_ns`: it carries the instant on the world's clock the live episode ends at, so the time left is that value minus your own `clock.now_ns()`. An episode publishes its deadline once the rig is ready — after the task's prepare handlers answer, so the reset costs the episode nothing. It is `None` whenever no deadline stands — no episode running, or one whose task has no `timeout_sec`.

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
