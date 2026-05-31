# Connect Your Model

Run your policy against Positronic's simulation environment, or connect it for real-robot evaluation on [PhAIL](https://phail.ai).

Positronic's inference API connects any model to any robot over a single WebSocket protocol. This guide covers both how it works and the reasoning behind it:

1. Running a reference model to see the system end-to-end
2. The execution model — how the model and a real-time client split the work
3. Observations, actions, and codecs — the wire contract
4. Implementing your own inference server

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://www.docker.com/)
- Clone the repo: `git clone git@github.com:Positronic-Robotics/positronic.git && cd positronic`

## Run a Reference Model

Start an ACT inference server using a public checkpoint trained on the simulated cube stacking task:

```bash
cd docker && docker compose run --rm --service-ports lerobot-0_3_3-server demo
```

The server downloads the checkpoint (~505MB) and starts a WebSocket API on port 8000. The server requires Docker on Linux. Verify it's ready:

```bash
curl http://localhost:8000/api/v1/models
# {"models": ["050000"]}
```

In a separate terminal, run inference in MuJoCo simulation. The inference client runs on Mac or Linux:

```bash
uv run positronic-inference sim \
  --policy=.remote --policy.host=<server-host> --policy.port=8000 \
  --driver.show_gui=True \
  --output_dir=~/datasets/demo_run
```

The MuJoCo window shows the Franka arm executing the policy in real time. The `--output_dir` flag records all episodes (robot state, camera feeds, actions) for later review.

Browse recorded episodes with:

```bash
uv run positronic-server --dataset.path=~/datasets/demo_run --port=5001
# Open http://localhost:5001
```

## The Execution Model: Predictor and Real-Time Client

Before the wire format, the idea behind it. Physical AI inference splits naturally into two halves:

- A **predictor** — your model, behind the WebSocket. Given the current observation, it returns a *chunk* of future actions. It holds no clock and, ideally, no per-session state, so it can run serverless and scale to zero.
- A **real-time client** — the inference loop running next to the robot (or simulator). It owns the clock. It decides *when* each predicted action executes and *how* one chunk hands off to the next.

This split is the point of the API. The hard real-time work — meeting control deadlines, recovering from errors, blending overlapping predictions — stays on the client, close to the hardware. The model only predicts. Everything below follows from that division of labor.

### A command is a trajectory, and a new one overwrites the old

The client doesn't ask for one action at a time. The model returns a *chunk* — a short trajectory of upcoming actions — and the client plays it out while requesting the next one. When the next chunk arrives, it replaces whatever is still in flight: the driver's `TrajectoryPlayer` swaps its whole buffer (`set()`), it does not append or merge. Override means replace. That keeps the wire contract trivial and leaves the policy for *how* to overwrite — wait, blend, ensemble — entirely on the client.

### Timestamps are relative; the client makes them absolute

Each action in a chunk carries a `timestamp`: seconds from the start of the chunk, **not** wall-clock time. The model can't know when its prediction will actually reach the robot — that depends on network and inference latency it can't observe. So it emits *relative* offsets from zero, and the client anchors them to its own clock the instant inference returns:

```
execute action i at:  now + timestamp[i]
```

Because `now` is read *after* the model responds, execution begins at inference-*finish*, and the round-trip latency is absorbed instead of leaving behind a stale, mistimed trajectory. The relative→absolute conversion happens in exactly one place (the client's scheduling wrapper), so the model and codecs stay clock-free.

### Why per-action timestamps, not "fps + count"

Most models today emit chunks at a fixed rate, and the built-in codecs do exactly that — `ActionTimestamp(fps=…)` stamps `0, 1/fps, 2/fps, …`. But the API does not assume it. Every action carries its own offset, so a non-uniform chunk, a resampled one, or one rewritten by a scheduling strategy is equally valid. We deliberately keep "equally-spaced chunk" out of the wire format.

### Scheduling strategies live on the client

How successive chunks combine is a client-side, swappable concern. The same wire contract supports a spectrum of strategies:

- **Predict → execute → predict** (the default, `ChunkedSchedule`): play the current chunk to its end, then request the next. Simple, and the right default.
- **Temporal ensembling** (ACT, [Zhao et al. 2023](https://arxiv.org/abs/2304.13705)): keep several overlapping chunks and average their overlapping actions for smoother motion.
- **Real-time chunking** (RTC, [Black et al. 2025](https://arxiv.org/abs/2506.07339)): predict the next chunk *while* the current one executes, freezing the actions already committed and inpainting the rest — robust to inference latency on dynamic tasks.

All three consume the *same* chunk-of-timestamped-actions described above; they differ only in how the client combines the clock, the incoming chunk, and the trajectory still in flight. Keeping that state on the client is what lets the server stay stateless — even RTC, which needs the previous trajectory, fits by having the client send that trajectory back inside the next observation rather than the server remembering it.

### One loop for real and simulation

The client reaches the world through a clock and a driver, both swappable. Wire in a Franka and the loop runs on hardware; wire in MuJoCo and the identical loop runs in simulation — only the `pimm.Clock` and the driver change. That is why the same `positronic-inference` command and the same server work for both.

## Observations and Actions

This is the concrete wire form of the execution model above. Every timestep, the inference client sends the current robot state to the server and receives a chunk of actions back. All messages use [msgpack](https://msgpack.org/) with numpy array support (see [Serialization](#serialization) below).

### Observations (client to server)

The client sends the full raw robot state as a dict. Keys are flat strings (the dots are literal, not nesting):

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `robot_state.ee_pose` | float32 | (7,) | End-effector pose: `x, y, z, qw, qx, qy, qz` (quaternion is **wxyz**, scalar first) |
| `robot_state.q` | float32 | (7,) | Joint positions (radians) |
| `robot_state.dq` | float32 | (7,) | Joint velocities (radians/s) |
| `grip` | float32 | scalar | Gripper opening |
| `image.<name>` | uint8 | (H, W, 3) | Camera RGB. Stream names come from the run config — sim and PhAIL send `image.exterior` and `image.wrist` (sim also adds `image.agent_view`) |
| `inference_time_ns` | int | scalar | Inference-clock timestamp of this observation (ns) |
| `wall_time_ns` | int | scalar | Wall-clock timestamp (ns) |
| `task` | str | — | Language instruction for the episode |

Your server receives all keys every step. Use what your model needs, ignore the rest. Image stream names are configuration-driven, so key off the names your deployment uses (`image.exterior`, `image.wrist`) rather than assuming fixed names.

### Actions (server to client)

The server returns a list of action dicts (an action chunk):

```python
{"result": [
    {"robot_command": {...}, "target_grip": 0.04, "timestamp": 0.0},
    {"robot_command": {...}, "target_grip": 0.04, "timestamp": 0.066},
    ...
]}
```

Each action carries:

| Field | Type | Description |
|-------|------|-------------|
| `robot_command` | dict | Control command (see table below) |
| `target_grip` | float | Target gripper opening |
| `timestamp` | float | Chunk-relative execution time in seconds — see [The Execution Model](#the-execution-model-predictor-and-real-time-client). The client schedules each action at `now + timestamp` (e.g. `i / action_fps` for the i-th action). A single action dict returned *outside* a list is auto-stamped `0.0`; give every action in a list its own `timestamp`, or the whole chunk collapses onto the same instant and fires at once. |

The `robot_command` field specifies the control mode:

| Command type | Fields | Description |
|--------------|--------|-------------|
| `cartesian_pos` | `pose`: float32 (12,) | Target EE pose: 3 translation + 9 flattened rotation matrix (row-major) |
| `joint_pos` | `positions`: float32 (7,) | Target joint angles (radians) |
| `joint_delta` | `velocities`: float32 (7,) | Joint velocity command |

The codec determines which command type the model produces (see below).

## Codecs: State and Action Representations

Different models expect different input/output formats. Some use end-effector pose, others use joint positions. Some output absolute targets, others output deltas. Positronic uses **codecs** to handle this translation.

A codec sits between the wire protocol and the model:

```
Raw observation (wire) --> codec.encode() --> model input
Model output           --> codec.decode() --> raw action (wire)
```

The wire format (what your server receives and returns) is always the raw robot state described above. If you use Positronic's built-in servers, the codec is configured at server startup:

```bash
# EE pose observation, absolute position actions
docker compose run --rm --service-ports lerobot-0_3_3-server serve \
  --checkpoints_dir=... \
  --codec=@positronic.vendors.lerobot_0_3_3.codecs.ee

# Joint position observation
docker compose run --rm --service-ports lerobot-0_3_3-server serve \
  --checkpoints_dir=... \
  --codec=@positronic.vendors.lerobot_0_3_3.codecs.joints
```

If you implement your own server, you handle this transformation yourself: pick the fields you need from the raw observation, and return actions in the raw format.

### Common representations

| Observation space | What the model sees | When to use |
|-------------------|--------------------|----|
| EE pose (7D) + grip + images | Position and orientation of the end-effector | Most common; sufficient for most manipulation tasks |
| EE pose + joint positions (7D) + grip + images | Both EE and joint state | When joint configuration matters (redundancy resolution, singularity avoidance) |
| Joint positions (7D) + grip + images | Joint angles only | Joint-space policies; no EE computation needed |

| Action space | What the model outputs | When to use |
|--------------|----------------------|-----|
| Absolute EE position (7D) + grip | Target pose the robot should move to | Default; works with position controllers |
| EE delta + grip | Displacement from current pose | Relative policies; smaller action space |
| Joint positions (7D) + grip | Target joint angles | Direct joint control; bypasses IK |

All built-in codecs are documented in the [Codecs Guide](codecs.md) with vendor-specific variants.

## Implement Your Own Server

To connect a custom model, implement a WebSocket server that speaks Positronic's Protocol v1.

### Endpoints

Your server must expose:

| Endpoint | Type | Description |
|----------|------|-------------|
| `GET /api/v1/models` | HTTP | Returns `{"models": ["model_a", "model_b"]}` |
| `WS /api/v1/session` | WebSocket | Inference session with default model |
| `WS /api/v1/session/{model_id}` | WebSocket | Inference session with specific model |

### Session Flow

1. Client connects via WebSocket
2. Server sends status messages while loading (optional but recommended for slow loads):
   ```python
   {"status": "loading", "message": "Loading model..."}
   ```
3. Server sends ready with metadata:
   ```python
   {"status": "ready", "meta": {"type": "my_model", "checkpoint_id": "v1"}}
   ```
4. Inference loop: client sends observation, server returns action, repeat until disconnect

### Using Positronic's Server Base Class

For a fast-loading, in-process model, subclass `InferenceServer` and provide a `Policy`. A `Policy` is a factory: its `new_session(context)` creates a per-episode `Session`, and that `Session` is called once per timestep to return wire-format actions. `Policy.meta` is static (about the model); per-episode info can be exposed via `Session.meta`.

```python
from positronic.offboard.basic_server import InferenceServer
from positronic.policy import Policy, Session

class MySession(Session):
    def __init__(self, model):
        self._model = model

    def __call__(self, obs):
        # obs keys: robot_state.ee_pose, robot_state.q, robot_state.dq, grip,
        #           image.exterior, image.wrist, inference_time_ns, wall_time_ns, task
        # Pick what your model needs:
        images = obs['image.exterior']
        ee = obs['robot_state.ee_pose']

        # Run your model, get a list of predicted poses
        predicted_poses = self._model.predict(images, ee)

        # Return action chunk: list of wire-format commands
        return [
            {'robot_command': {'type': 'cartesian_pos', 'pose': pose}, 'target_grip': 0.04}
            for pose in predicted_poses
        ]

class MyPolicy(Policy):
    def __init__(self, model):
        self._model = model

    def new_session(self, context=None):
        # Called once per episode; do any per-episode setup here, then return a Session.
        return MySession(self._model)

    @property
    def meta(self):
        return {'type': 'my_model'}

# Create server with policy registry
server = InferenceServer(
    policy_registry={'default': lambda: MyPolicy(load_my_model())},
    host='0.0.0.0',
    port=8000,
)
server.serve()
```

Test it:

```bash
uv run positronic-inference sim \
  --policy=.remote --policy.host=localhost --policy.port=8000
```

`InferenceServer` loads the policy synchronously in-process, so it assumes fast (<20 s) loading — otherwise the WebSocket handshake times out before the `ready` message.

### For Slow-Loading or Subprocess Models

The built-in OpenPI and GR00T servers don't use `InferenceServer` — they subclass `VendorServer` (`positronic/offboard/vendor_server.py`), which is the pattern to follow for checkpoints that take minutes to download or run as a separate process. `VendorServer` adds, on top of the same Protocol v1:

- **Progress during loading** — it streams `{"status": "loading", ...}` messages while a checkpoint downloads or a subprocess boots, so the client keepalive doesn't expire.
- **A built-in codec boundary** — you construct it with a `Codec`, and the base wraps your policy via `codec.wrap(policy)`. Your `Policy`/`Session` then works entirely in *model space* (it receives codec-encoded observations and returns model-native actions); the codec translates to and from the wire format described above. This is why OpenPI's session simply returns `[{'action': a} for a in actions]` rather than building `robot_command` dicts itself.
- **Lifecycle hooks** — subclasses implement `resolve_model()`, `create_policy()`, and `get_models()`; the base handles the WebSocket loop, warmup, multi-model switching, optional `recording_dir`, and idle shutdown.

See `positronic/vendors/openpi/server.py` and `positronic/vendors/gr00t/server.py` for complete working references.

### Standalone Implementation

If you prefer not to depend on Positronic for the server, implement the WebSocket protocol directly. The key requirement is msgpack serialization with numpy support (see below).

### Serialization

All messages use msgpack. Numpy arrays are encoded with a custom extension:

```python
# numpy array -> msgpack
{
    b"__ndarray__": True,
    b"data": array.tobytes(),   # raw bytes
    b"dtype": str(array.dtype), # e.g. "<f4"
    b"shape": array.shape       # tuple
}
```

Positronic provides `serialise()` and `deserialise()` in `positronic.utils.serialization` that handle this automatically:

```python
from positronic.utils.serialization import serialise, deserialise

# Server-side WebSocket handler
session = policy.new_session()           # one Session per episode/connection
async for message in websocket.iter_bytes():
    obs = deserialise(message)           # dict with numpy arrays
    actions = session(obs)               # list of action dicts (or None)
    await websocket.send_bytes(serialise({"result": actions}))
```

## See Also

- [Inference Guide](inference.md) – local and remote inference patterns
- [Codecs Guide](codecs.md) – all available codecs by vendor
- [Offboard Protocol](../positronic/offboard/README.md) – full Protocol v1 specification
- [Training Workflow](training-workflow.md) – training with public datasets
