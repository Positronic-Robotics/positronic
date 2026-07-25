# Positronic Offboard Inference

This package implements the protocol and utilities for offboard policy inference, allowing robots or simulators to stream observations to a remote server and receive actions.

## Protocol v1

The unified WebSocket protocol is built to enable ANY hardware to connect to ANY model. All Positronic inference servers (LeRobot, GR00T, OpenPI) implement this protocol, allowing a single `.remote` policy client to work across all vendors.

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
`ws://localhost:8000/api/v1/session/GEAR-Dreams/DreamZero-DROID` serves that HuggingFace checkpoint.

#### Session parameters

Query params on the session URL tune the served policy pipe for that one session. Each key is a dotted path into the server's pipe config — any argument at any depth — applied as a config override before the session is built:

```
ws://localhost:8000/api/v1/session?codec.fps=10&local.pad_start=false
```

Rules:

- **Values are JSON literals.** Each value is parsed as JSON (`10` → int, `false` → bool, `"hello"` → str); a value that does not parse passes through as a plain string, so a hand-typed `?tag=hello` works. Programmatic clients should JSON-encode every value — the Python client does — so types round-trip exactly.
- **Imports are rejected.** configuronic resolves strings starting with `@` or `.` as imports, so any such string anywhere inside a value is rejected. Params can tune the pipe's arguments, never swap its components.
- **Duplicate keys are rejected.**
- **Params never name a model.** The path does that, and only the path: `/api/v1/session/20000?codec.fps=10` serves model `20000` with that override. A `?model_id=...` param is an ordinary unknown key and is rejected.
- **The model source is fixed at launch.** Params that would change it (e.g. `?source.checkpoint=...`) are rejected; the only way to get a different model is the path.
- **Only config-launched servers accept params.** All vendor servers qualify; a `PolicyServer` built from an already-instantiated pipe rejects every param.

Any violation — including an unknown key — fails at connect: the server sends `{"status": "error", "error": ...}` and closes the socket (code 1008) before anything moves, and the Python client raises `RuntimeError`. Overrides apply per session, and the `local_stack` declared in the ready handshake reflects them.

Because the whole session configuration fits in the URL, one string is a complete endpoint description:
`RemotePolicy.from_url('gpu-host:8000?codec.fps=10')` (CLI: `--policy=.remote_url --policy.url='...'`) accepts
`host`, `host:port`, and full `http(s)`/`ws(s)` URLs — optionally with `/api/v1/session/<model_id>` — and forwards
the query string verbatim.

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
    "positronic_version": "0.2.1"
  }
}
```

The client ignores all messages until it sees `status == "ready"` (status updates like `loading`/`waiting` may arrive first).

This metadata tells the client:
- Which checkpoint is loaded
- Server connection details
- Codec metadata (`image_sizes` — the geometry the codec encodes to, `action_fps` and `action_horizon_sec` for timing)
- `local_stack` — the declared local half of the policy pipe: a spec tree of `{"name", "args"}`
  leaves composed by `"seq"` (the `|` operator) and `"par"` (the `&` operator). `RemotePolicy` builds
  this stack in front of the connection, resolving names only against the closed vocabulary in
  `positronic.policy.spec.WIRE_WRAPPERS` — an unknown entry fails at connect, before the robot moves.
  An empty declaration (`{"seq": []}`) means the policy needs no rig-side glue; when the key is
  absent the server declares nothing and the client falls back to the standard `ChunkedSchedule`.
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
```json
{
  "ee_pose": [0.5, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
  "grip": [0.04],
  "wrist_image": "<base64_encoded_image>",
  "exterior_image": "<base64_encoded_image>"
}
```

**Server → Client (Actions):**

`result` is a **list** of action dicts — one per action in the predicted chunk (or `null` if the model produced no actions):

```json
{
  "result": [{
    "action": {
      "target_pose": [0.51, 0.21, 0.31, 0.0, 0.0, 0.0, 1.0],
      "target_grip": [0.02]
    }
  }]
}
```

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
# LeRobot server (SmolVLA — 0.4.x); --pipe selects the named codec pipe
cd docker && docker compose run --rm --service-ports lerobot-server serve \
  --checkpoints_dir=~/checkpoints/lerobot/exp_v1 \
  --pipe=ee

# GR00T server (swap hardware code stays the same); the subcommand names the pipe
cd docker && docker compose run --rm --service-ports groot-server \
  ee_rot6d_joints \
  --checkpoints_dir=~/checkpoints/groot/exp_v1

# Client connects the same way
uv run positronic-inference sim \
  --policy=.remote \
  --policy.host=localhost
```

**Model Switching:** Compare multiple models without restarting the server by using specific session endpoints.

**Status Streaming:** Long model loads are handled gracefully with progress updates.

**Server-side recording:** Servers accept an optional `recording_dir`. When set, each WebSocket session writes a rerun `.rrd` file that taps both sides of the codec: `raw` captures the obs/action at the wire boundary, and `inference` captures the encoded observation and raw model output.

**Python Client:** We provide a Python client (`positronic.offboard.client.InferenceClient`) that handles the WebSocket protocol automatically. While the API is currently in alpha and may change, we'll do our best to maintain backward compatibility for the inference client.

## Classes

### `server.PolicyServer`
The one server implementation behind every vendor. It serves a **policy pipe** (see `positronic.policy.spec`): a wrapper chain with a `remote` marker, closed by a `ModelSource` terminal. The half right of the marker wraps the model on the server; the half left of it is declared as `local_stack` in the ready handshake for the client to build. The source is the only model loader: `get_models()` backs `/api/v1/models`, `resolve()` maps a requested id (or the default), and `load(model_id, on_progress)` produces the `Policy` — with `on_progress` messages streamed to the connecting client as `loading` status frames.

```python
from positronic.offboard import PolicyServer
from positronic.policy.spec import PolicySource, remote
from positronic.policy.wrappers import ChunkedSchedule

pipe = ChunkedSchedule() | remote | PolicySource(my_policy)
PolicyServer(pipe, host='0.0.0.0', port=8000).serve()
```

`PolicySource` serves one ready in-process policy; vendors instead define a `ModelSource` over a checkpoint directory. Passing a `cfn.Config` that builds the pipe — as the vendor servers do with their named `PIPES` — enables [session parameters](#session-parameters); an instantiated pipe serves exactly as launched. `recording_dir` enables the per-session recording taps described above, and `idle_timeout_min` shuts the server down after that many minutes without activity.

### `client.InferenceClient`
A Python client for connecting to an inference server.

```python
from positronic.offboard.client import InferenceClient

client = InferenceClient('localhost', 8000)
# Session params ride on every session URL, JSON-encoded:
# client = InferenceClient('localhost', 8000, params={'codec.fps': 10})

# Connect to default policy
session = client.new_session()
# OR connect to specific policy
# session = client.new_session('model_a')

meta = session.metadata
action = session.infer(observation)
```

## Vendor Implementations

Every vendor ships a `ModelSource` plus named pipes and serves them through the one `PolicyServer`:

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
