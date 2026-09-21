# Positronic Offboard Inference

This package implements the protocol and utilities for offboard policy inference, allowing robots or simulators to stream observations to a remote server and receive actions.

## Protocol v1

The protocol connects any hardware to any model. All Positronic inference servers (LeRobot, GR00T, OpenPI) implement it, so a single `.remote` policy client works across all vendors.

### Wires

The protocol is a sequence of msgpack frames, and two wires carry them. Both carry the same frames in
the same order. The client side of each wire, and the facts both ends share, ship as the
`positronic-wire` distribution ([wire/README.md](../../wire/README.md)); this package holds the
server side.

| Wire | `--policy.wire` | Where it answers |
|---|---|---|
| WebSocket | `websocket`, or `websocket_tls` behind a TLS edge | the server's `port`, beside the HTTP routes |
| WebSocket on a Unix socket | `websocket_unix` | the server's `uds` path, beside the same HTTP routes |
| gRPC | `grpc` | the server's `grpc_port`, sessions alone |
| gRPC over TLS | `grpc_tls` | a TLS edge in front of that same `grpc_port` |

- A client names its wire; nothing reads one off a URL. The WebSocket wire is the default. A server serves
  gRPC only when `grpc_port` names a port.
- `websocket_unix` reaches a server on the same machine, over no network. `serve --uds /run/policy.sock`
  binds that path in place of `host` and `port`, and `--policy.uds` dials it. A caller names the wire and
  then fills that wire's address, so a socket address carries no host and no port at all. Both paths are
  absolute: a relative one is resolved against whatever directory each side was started from, and the
  address refuses it. A socket is same-machine by construction, so there is no TLS member beside it, and
  `--uds` leaves the gRPC wire on host and port.
- A gRPC session is one bidirectional stream on `/positronic.offboard.v1.Inference/Session`.
  No `.proto` file describes the frames.
- The session path, the query and the bearer token cross as the `positronic-session-path`,
  `positronic-session-query` and `authorization` metadata.
- Take the gRPC wire wherever it reaches. Python's WebSocket stack spends about 30 ms per
  846 KiB observation on framing; gRPC spends about 1 ms. Through a managed front the same
  observation round-trips in about 6 ms over gRPC and about 60 ms over the WebSocket.
- gRPC reaches through a managed HTTPS front. The front terminates TLS and must select HTTP/2
  over ALPN. The server binds a plaintext port and holds no certificate.
- On Nebius, declare the gRPC port as an HTTP port and dial it on `grpc_tls`, port 443.
  A `/tcp` port selects no ALPN protocol, and gRPC refuses it. See
  [workflows/nebius/README.md](../../workflows/nebius/README.md#serve-a-checkpoint-as-an-endpoint).

Both wires ping through a silent wait. A front drops a connection it reads nothing from (the managed
front after about 90 s), and the pings keep an inference open through that wait.

`/api/v1/models` is an HTTP route and stays on the server's `port`. `InferenceClient.list_models`
refuses a gRPC wire.

### Authentication

`PolicyServer(auth_token=...)` gates every route below on `Authorization: Bearer <token>`, answering
`401` on the HTTP route, refusing the WebSocket upgrade before the session opens, and answering
`PERMISSION_DENIED` on the gRPC wire. `serve` — the entry point every vendor CLI exposes — takes that
token from the `AUTH_TOKEN` environment variable, so
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

#### `/api/v1/session`
Establishes an inference session with the **default** model — the checkpoint pinned at server startup (the configured one, or the latest available at that moment).

#### `/api/v1/session/{model_id}`
Establishes an inference session with a **specific** model.

**Example:**
- `ws://localhost:8000/api/v1/session` → Default model
- `grpc://localhost:9000/api/v1/session` → Default model, over gRPC
- `ws://localhost:8000/api/v1/session/10000` → Model 10000
- `grpc://localhost:9000/api/v1/session/10000` → Model 10000, over gRPC

Each wire from the table above takes the same path; only the scheme and the port change.

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

Any violation — including an unknown key — fails at connect: the server sends `{"status": "error", "error": ...}` and ends the session before anything moves, and the Python client raises `RuntimeError`. Overrides apply per session, and the `local_stack` declared in the ready handshake reflects them.

The client names each part: `--policy=.remote --policy.wire=websocket --policy.host=gpu-host --policy.port=8000
--policy.model=<model_id> --policy.query='codec.fps=10'`, and forwards the query string verbatim. Credentials stay a
separate `headers` argument.

### Session Flow

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

Some models may take a long time to load (e.g., OpenPI and GR00T can take 120-300s). The client gives the handshake 30 s per message; the server sends status updates during loading, on either wire:

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
cd docker && docker compose run --rm --service-ports groot-server droid \
  --pipeline.source.model_source=~/checkpoints/groot/exp_v1

# Client connects the same way
uv run positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=.remote \
  --policy.host=localhost --policy.port=8000
```

**Model Switching:** Compare multiple models without restarting the server by using specific session endpoints.

**Status Streaming:** Long model loads are handled gracefully with progress updates.

**Server-side recording:** Servers accept an optional `recording_dir`. When set, each session writes a rerun `.rrd` file that taps both sides of the codec: `raw` captures the obs/action at the wire boundary, and `inference` captures the encoded observation and raw model output.

**Python Client:** A Python client (`positronic.offboard.client.InferenceClient`) handles the protocol. The API is in alpha and may change.

## Classes

### `server.PolicyServer`
The one server implementation behind every vendor. It serves a **policy pipeline** (see `positronic.policy.spec`): a layer chain with a `remote` marker, closed by a `ModelSource` terminal. The half right of the marker wraps the model on the server; the half left of it is declared as `local_stack` in the ready handshake for the client to build. The source is the only model loader: `get_models()` backs `/api/v1/models`, `resolve()` maps a requested id (or the default), and `load(model_id, on_progress)` produces the `Policy` — with `on_progress` messages streamed to the connecting client as `loading` status messages.

```python
from positronic.offboard.server import PolicyServer
from positronic.offboard.websocket_wire import WebsocketWire
from positronic.policy.spec import PolicySource, remote
from positronic.policy.layers import ChunkedSchedule

pipeline = ChunkedSchedule() | remote | PolicySource(my_policy)
server = PolicyServer(pipeline)
server.serve([WebsocketWire('0.0.0.0', 8000, server.api)])
```

`serve` takes the wires that sessions arrive on. Each wire binds its own address, reads its own route
for the model a session asks for, and checks its own session headers. Add `grpc_wire.GrpcWire(host,
port)` to the list to serve gRPC beside the WebSocket. An HTTP wire takes `server.api`, the model
catalogue, and answers it wherever it carries sessions. A wire names where it bound in its
`served_address` property: `ServedHostPort` for a host and a port — a wire asked for port 0 binds any
free one, and `ws.served_address.port` is the port it took — or `ServedUnixSocket` for a socket path,
which has no port at all.

`PolicySource` serves one ready in-process policy; vendors instead define a `ModelSource` over a checkpoint directory. Passing a `cfn.Config` that builds the pipeline — as the vendor servers do with their named pipelines — enables [session parameters](#session-parameters); an instantiated pipeline serves exactly as launched. `recording_dir` enables the per-session recording taps described above, and `idle_timeout_min` ends the server after that many minutes without activity.

### `server.serve`
The CLI entry point every vendor server exposes. A vendor binds `pipeline` to each of its named pipelines and lists the results as subcommands, so `<vendor>-server <pipeline>` launches one. Only `--host`, `--port`, `--uds`, `--grpc_port`, `--recording_dir` and `--idle_timeout_min` are flags of `serve` itself; everything the served model is — codec, source, checkpoint — is reached through the pipeline, which is also where a deployment preset binds it. Select GR00T checkpoints with `--pipeline.source.model_source=...`; LeRobot and OpenPI use `--pipeline.source.checkpoints_dir=...`.

### `client.InferenceClient`
A Python client for connecting to an inference server. It takes the wire and the session address as values
(`positronic_wire.registry.CLIENT_WIRES` lists every wire by name); the address fixes the model and the
session params, so serving another model means another client.

```python
from positronic.offboard.client import InferenceClient
from positronic_wire import registry
from positronic_wire.wire import SessionAddress, session_path

# The server's pinned checkpoint, with no session params
client = InferenceClient(registry.client_wire('websocket'), SessionAddress('localhost', 8000, session_path(), ''))
# A named model, tuned for every session this client opens
# client = InferenceClient(registry.client_wire('websocket'), SessionAddress('localhost', 8000, session_path('model_a'), 'codec.fps=10'))
# The same session on the gRPC wire, on a LAN and behind a TLS edge
# client = InferenceClient(registry.client_wire('grpc'), SessionAddress('localhost', 9000, session_path('model_a'), ''))
# client = InferenceClient(registry.client_wire('grpc_tls'), SessionAddress('gpu-host', 443, session_path('model_a'), ''))

session = client.new_session()
meta = session.metadata
action = session.infer(observation)
```

`new_session` retries a cold backend until `connect_deadline`, and raises `TimeoutError` when it stays
cold. A refusal that no retry clears raises `wire.ConnectRefused`, whose `refusal` says what the server
answered: `FORBIDDEN` for a refused credential, `FINAL` for a permanent refusal. `new_session` raises no
exception of the WebSocket or gRPC library.

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
