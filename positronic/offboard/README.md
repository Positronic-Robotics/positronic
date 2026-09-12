# Positronic Offboard Inference

This package implements the protocol and utilities for offboard policy inference, allowing robots or simulators to stream observations to a remote server and receive actions.

## Separation of responsibilities: adapter vs codec vs wire client

Three layers touch an observation on its way to a model, and each owns exactly one concern.
When writing a new sim/rig adapter, check this table before adding any transform to it:

| Layer | Owns | Examples |
|---|---|---|
| **Adapter** (per sim/rig, e.g. `simulator/molmo_spaces/adapter.py`) | Rig semantics ONLY: mapping the rig's observation/action vocabulary onto positronic's raw keys | Camera-key mapping, gripper qpos → `[0, 1]` closure, decoded commands → the rig's action format |
| **Codec** (per model family, `policy/codec.py` subclasses) | Model preprocessing: everything the checkpoint's input distribution requires | Resize-with-pad to model resolution, prompt normalization (e.g. DROID lowercasing), state assembly |
| **Wire client** (`InferenceClient` / `RemotePolicy`) | Transport optimization, negotiated — never semantics | Downscaling frames to the server-advertised `image_sizes` (aspect-preserving, never upscaling), optional JPEG compression |

Consequences:

- **An adapter never resizes, pads, normalizes prompts, or otherwise preprocesses for the model.**
  It passes frames and text through at native fidelity. If the same transform appears in an adapter
  and a codec, the adapter's copy is the bug: a drifted duplicate silently changes eval inputs.
- **Bandwidth is not the adapter's problem.** The client already downsizes to what the server says
  it needs: every `Codec` advertises its expected input sizes via the reserved `image_sizes` meta
  key (see `Codec.meta`), the server returns it in the session handshake, and the client fits
  frames to it before sending. This is default-on — an adapter that resizes "to keep the wire
  payload small" is duplicating it.
- **Codecs run on either side of the wire** — the client being the process driving the robot or sim,
  the server being the process holding the model. positronic-native evals compose the codec around
  `RemotePolicy` on the client (`cfg/policy.py` — the wire then carries model-sized encoded inputs,
  and the client-side resize is disabled since `codec.meta` already reports `image_sizes`).
  Thin-client deployments (a sim adapter in a foreign venv talking to a serverless endpoint) host
  the codec on the server — the wire carries raw positronic keys, downsized by the negotiation
  above. Both placements are supported; pick by where the dependencies can live.

## Protocol v1

The unified WebSocket protocol is built to enable ANY hardware to connect to ANY model. All Positronic inference servers (LeRobot, GR00T, OpenPI) implement this protocol, allowing a single `.remote` policy client to work across all vendors.

### Authentication

`PolicyServer(auth_token=...)` gates every route below on `Authorization: Bearer <token>`, answering
`401` on the HTTP route and refusing the WebSocket upgrade before the session opens. `serve` — the
entry point every vendor CLI exposes — takes that token from the `AUTH_TOKEN` environment variable, so
a secret never lands in the process arguments. No token serves open, which is the usual shape on a
trusted LAN; an empty one is a broken secret and refuses to start. `InferenceClient(headers=...)`
carries the header, and `positronic.cfg.policy.authed_remote` fills it in from the same variable.

### Endpoints

#### `GET /api/v1/models`
Returns a list of available model IDs.

**Example Request:**
```bash
curl http://localhost:8000/api/v1/models
```

**Response:**
```json
{
  "models": ["10000", "20000", "30000"]
}
```

Use this to discover which models are available before connecting.

#### `WS /api/v1/session`
Establishes an inference session with the **default** model — the checkpoint pinned at server startup (the configured one, or the latest available at that moment).

#### `WS /api/v1/session/{model_id}`
Establishes an inference session with a **specific** model.

**Example:**
- `ws://localhost:8000/api/v1/session` → Default model
- `ws://localhost:8000/api/v1/session/10000` → Model 10000
- `ws://localhost:8000/api/v1/session/20000` → Model 20000

The id is everything after the prefix, slashes included, so a source may advertise one that is itself a path:
`ws://localhost:8000/api/v1/session/GEAR-Dreams/DreamZero-DROID` serves that HuggingFace checkpoint. Anything else
that would end the path or be decoded away (`?`, `#`, `%`, `:`) must be percent-encoded by whoever writes the URL,
so `s3://bucket/ckpt-1` is requested as `s3%3A//bucket/ckpt-1` and arrives as the original id.

#### Session parameters

Query params on the session URL tune the served policy pipeline for that one session. Each key is a dotted path into the server's pipeline config — any argument at any depth — applied as a config override before the session is built:

```
ws://localhost:8000/api/v1/session?codec.fps=10&local.pad_start=false
```

Rules:

- **Values are JSON literals.** The server parses each value as JSON (`10` → int, `false` → bool, `"hello"` → str); a value that does not parse passes through as a plain string, so a hand-typed `?tag=hello` works. The query travels verbatim — `InferenceClient` forwards whatever the URL already says — so a caller who means the string `true` rather than the boolean writes the quoted literal itself, percent-encoded: `?tag=%22true%22`.
- **Imports are rejected.** Overrides are applied with `Config.override_data`, so a value that configuronic would read as an import — `@module.path.Object`, or a leading-dot path relative to the argument's current value — is refused at any nesting depth, and the error names the offending key. Params can tune the pipeline's arguments, never swap its components. A leading-dot string on an argument that gives imports no base to resolve against (a number, a flag, a plain string) is ordinary data and passes through, so `?tag=./data` works.
- **Duplicate keys are rejected.**
- **Params never name a model.** The path does that, and only the path: `/api/v1/session/20000?codec.fps=10` serves model `20000` with that override. A `?model_id=...` param is an ordinary unknown key and is rejected.
- **The model source is fixed at launch.** Params that would change it (e.g. `?source.checkpoint=...`) are rejected; the only way to get a different model is the path.
- **Only config-launched servers accept params.** All vendor servers qualify; a `PolicyServer` built from an already-instantiated pipeline rejects every param.

Any violation — including an unknown key — fails at connect: the server sends `{"status": "error", "error": ...}` and closes the socket (code 1008) before anything moves, and the Python client raises `RuntimeError`. Overrides apply per session, and the `local_stack` declared in the ready handshake reflects them.

Because the whole session configuration fits in the URL, one string is a complete endpoint description:
`--policy=.remote --policy.url='gpu-host:8000?codec.fps=10'` accepts `host`, `host:port`, and full
`http(s)`/`ws(s)` URLs — optionally with `/api/v1/session/<model_id>` — and forwards the query string verbatim.
Credentials are the exception and stay a separate `headers` argument, so the URL itself is safe to hand around.

### WebSocket Flow

#### 1. Handshake
Upon connection, the server sends a ready packet with metadata:

```json
{
  "status": "ready",
  "meta": {
    "type": "lerobot",
    "host": "localhost",
    "port": 8000,
    "checkpoint_path": "~/checkpoints/lerobot/experiment_v1",
    "checkpoint_id": "10000",
    "image_sizes": [224, 224],
    "action_fps": 15.0,
    "action_horizon_sec": 1.0,
    "local_stack": {"seq": [
      {"name": "chunked_schedule"},
      {"name": "restrict_image_size", "args": {"width": 224, "height": 224}}
    ]},
    "compress_images": false,
    "positronic_version": "0.2.1"
  }
}
```

The client ignores all messages until it sees `status == "ready"` (status updates like `loading`/`waiting` may arrive first).

This metadata tells the client:
- Which checkpoint is loaded
- Server connection details
- Codec metadata (`image_sizes` — the geometry the codec encodes to, `action_fps` and `action_horizon_sec` for timing)
- `local_stack` — the declared local half of the policy pipeline: a spec tree of `{"name", "args"}`
  leaves composed by `"seq"` (the `|` operator) and `"par"` (the `&` operator). `RemotePolicy` builds
  this stack in front of the connection, resolving names only against the closed vocabulary in
  `positronic.policy.spec.WIRE_LAYERS` — an unknown entry fails at connect, before the robot moves.
  Never empty and never absent: a pipeline with nothing left of the marker is refused when the server
  starts, and a handshake declaring nothing is refused by the client. In practice it names at least a
  `chunked_schedule`, which turns the chunk-relative timestamps a codec stamps into times on the rig's
  clock — a stack that fails to leaves the harness rejecting the chunk at the first inference.
- `compress_images` — the `remote` marker's own wire setting: whether the rig JPEG-encodes frames before
  sending, for an endpoint behind a proxy with a message-size cap
- `positronic_version` — the server's positronic version, for diagnosing declaration mismatches

#### 2. Status Updates (Long Model Loading)

Some models may take a long time to load (e.g., OpenPI and GR00T can take 120-300s). The server sends periodic status updates during loading to prevent WebSocket keepalive timeouts:

```json
{
  "status": "loading",
  "message": "Loading checkpoint 10000, please wait..."
}
```

The client should display these status updates to the user. Once loading completes, the server sends the `status: "ready"` packet shown above.

#### 3. Inference Loop

After handshake, the client streams observations and receives actions:

**Client → Server (Observation):**

Keys are flat strings — the dots are literal, not nesting. Arrays travel as numpy, not base64; a rig behind a message-size cap JPEG-encodes its frames instead (see `compress_images` above). `docs/connect-your-model.md` carries the full key table.

```json
{
  "robot_state.ee_pose": [0.5, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
  "robot_state.q": [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8],
  "grip": 0.04,
  "image.wrist": "<uint8 (H, W, 3)>",
  "image.exterior": "<uint8 (H, W, 3)>",
  "obs_time_ns": 1737000000000000000,
  "task": "pick up the red cube"
}
```

**Server → Client (Actions):**

`result` is a **list** of action dicts — one per action in the predicted chunk (or `null` if the model produced no actions). `timestamp` is seconds from the start of the chunk; `robot_command` carries the control command, and a rig with more than one arm names the channel per arm (`robot_command.left`):

```json
{
  "result": [{
    "robot_command": {"type": "cartesian_pos", "pose": [0.51, 0.21, 0.31, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
    "target_grip": 0.02,
    "timestamp": 0.0
  }]
}
```

A command's `type` selects the fields beside it: `cartesian_pos` (`pose`), `joint_pos` (`positions`), `joint_delta` (`velocities`), and `cartesian_delta` (`delta`, `frame`). A pose is translation followed by a row-major 3x3 rotation.

Every command may carry a `mode`, itself a tagged mapping naming the control law to execute under: `{"type": "position_control", "stiffness": [...]}` or `{"type": "impedance", "kq": [...], "kqd": [...], "kx": [...], "kxd": [...]}`. Omit `stiffness` to take the arm's own gains — an empty list is refused. Omit `mode` entirely and the arm runs its native law. `positronic.offboard.protocol` reads that mapping into the typed command the drivers dispatch on, so a server written against another stack sends it as plain data; one built on positronic may instead put a `positronic.drivers.roboarm.command` instance here and let `serialise` encode it.

**Server → Client (Error):**
```json
{
  "error": "Shape mismatch: expected (7,) but got (6,)"
}
```

The loop continues until the client closes the connection or the episode ends.

### Key Benefits

**Unified API:** All vendors implement the same protocol, so swapping models is as simple as changing the server:

```bash
# LeRobot server (SmolVLA — 0.4.x); the subcommand names the codec pipeline
cd docker && docker compose run --rm --service-ports lerobot-server ee \
  --pipeline.source.checkpoints_dir=~/checkpoints/lerobot/exp_v1

# GR00T server (swap hardware code stays the same)
cd docker && docker compose run --rm --service-ports groot-server ee_rot6d_joints \
  --pipeline.source.checkpoints_dir=~/checkpoints/groot/exp_v1

# Client connects the same way
uv run positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=.remote \
  --policy.url=localhost:8000
```

**Model Switching:** Compare multiple models without restarting the server by using specific session endpoints.

**Status Streaming:** Long model loads are handled gracefully with progress updates.

**Server-side recording:** Servers accept an optional `recording_dir`. When set, each WebSocket session writes a rerun `.rrd` file that taps both sides of the codec: `raw` captures the obs/action at the wire boundary, and `inference` captures the encoded observation and raw model output.

**Python Client:** We provide a Python client (`positronic.offboard.client.InferenceClient`) that handles the WebSocket protocol automatically. While the API is currently in alpha and may change, we'll do our best to maintain backward compatibility for the inference client.

## Classes

### `server.PolicyServer`
The one server implementation behind every vendor. It serves a **policy pipeline** (see `positronic.policy.spec`): a layer chain with a `remote` marker, closed by a `ModelSource` terminal. The half right of the marker wraps the model on the server; the half left of it is declared as `local_stack` in the ready handshake for the client to build. The source is the only model loader: `get_models()` backs `/api/v1/models`, `resolve()` maps a requested id (or the default), and `load(model_id, on_progress)` produces the `Policy` — with `on_progress` messages streamed to the connecting client as `loading` status messages.

```python
from positronic.offboard import PolicyServer
from positronic.policy.spec import PolicySource, remote
from positronic.policy.layers import ChunkedSchedule

pipeline = ChunkedSchedule() | remote | PolicySource(my_policy)
PolicyServer(pipeline, host='0.0.0.0', port=8000).serve()
```

`PolicySource` serves one ready in-process policy; vendors instead define a `ModelSource` over a checkpoint directory. Passing a `cfn.Config` that builds the pipeline — as the vendor servers do with their named pipelines — enables [session parameters](#session-parameters); an instantiated pipeline serves exactly as launched. `recording_dir` enables the per-session recording taps described above, and `idle_timeout_min` shuts the server down after that many minutes without activity.

### `server.serve`
The CLI entry point every vendor server exposes. A vendor binds `pipeline` to each of its named pipelines and lists the results as subcommands, so `<vendor>-server <pipeline>` launches one. Only `--host`, `--port`, `--recording_dir` and `--idle_timeout_min` are flags of `serve` itself; everything the served model is — codec, source, checkpoint directory — is reached through the pipeline (`--pipeline.source.checkpoints_dir=...`), which is also where a deployment preset binds it.

### `client.InferenceClient`
A Python client for connecting to an inference server. One URL addresses it, in the same forms
`RemotePolicy` accepts: an omitted port is the scheme's own, 443 for `https`/`wss` and 80 otherwise. The URL
fixes the model and the session params, so serving another model means another client.

```python
from positronic.offboard.client import InferenceClient

# The server's pinned checkpoint, with no session params
client = InferenceClient('localhost:8000')
# A named model, tuned for every session this client opens
# client = InferenceClient('localhost:8000/api/v1/session/model_a?codec.fps=10')

session = client.new_session()
meta = session.metadata
action = session.infer(observation)
```

## Vendor Implementations

Every vendor ships a `ModelSource` plus named pipelines and serves them through the one `PolicyServer`:

- **LeRobot (0.4.x)**: `positronic.vendors.lerobot.server` - Serves SmolVLA/ACT/Diffusion checkpoints (auto-detects policy type)
- **LeRobot (0.3.3)**: `positronic.vendors.lerobot_0_3_3.server` - Serves ACT checkpoints with dynamic loading
- **GR00T**: `positronic.vendors.gr00t.server` - Serves GR00T checkpoints with modality config
- **OpenPI**: `positronic.vendors.openpi.server` - Serves OpenPI checkpoints with config name
- **DreamZero**: `positronic.vendors.dreamzero.server` - Serves DreamZero checkpoints through a torchrun subprocess
- **MolmoAct2**: `positronic.vendors.molmoact2.server` - Serves the pretrained MolmoAct2 DROID model

The server enforces a **Singleton Policy** (only one checkpoint loaded at a time) to manage GPU resources efficiently.

## See Also

- [Training Workflow](../../docs/training-workflow.md) - Starting inference servers
- [Inference Guide](../../docs/inference.md) - Remote policy usage and patterns
- [Model Selection](../../docs/model-selection.md) - Choosing between vendors
