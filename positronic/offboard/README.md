# Positronic Offboard Inference

This package implements the protocol and utilities for offboard policy inference, allowing robots or simulators to stream observations to a remote server and receive actions.

## Protocol v1

The protocol connects clients to callable models. ACT uses a server-declared processor stack on the
client, with codecs configured separately on either side of the connection.

TODO: Migrate the remaining vendor configurations to callable models and explicit `Pipeline` arguments.

### Wires

The protocol is a sequence of msgpack frames, and two wires carry them. Both carry the same frames in
the same order.

| Wire | URL | Port |
|---|---|---|
| WebSocket | `ws://host:8000/api/v1/session[/<model_id>]` | the server's `port`, beside the HTTP routes |
| gRPC | `grpc://host:9000/api/v1/session[/<model_id>]` | the server's `grpc_port`, sessions alone |
| gRPC over TLS | `grpcs://host:443/api/v1/session[/<model_id>]` | a TLS edge in front of that same `grpc_port` |

- The WebSocket wire is the default. A server serves gRPC only when `grpc_port` names a port.
- A gRPC session is one bidirectional stream on `/positronic.offboard.v1.Inference/Session`.
  No `.proto` file describes the frames.
- The session path, the query and the bearer token cross as the `positronic-session-path`,
  `positronic-session-query` and `authorization` metadata.
- Take the gRPC wire wherever it reaches. Python's WebSocket stack spends about 30 ms per
  846 KiB observation on framing; gRPC spends about 1 ms. Through a managed front the same
  observation round-trips in about 6 ms over gRPC and about 60 ms over the WebSocket.
- gRPC reaches through a managed HTTPS front. The front terminates TLS and must select HTTP/2
  over ALPN. The server binds a plaintext port and holds no certificate.
- On Nebius, declare the gRPC port as an HTTP port and dial it as `grpcs://<host>:443`.
  A `/tcp` port selects no ALPN protocol, and gRPC refuses it. See
  [workflows/nebius/README.md](../../workflows/nebius/README.md#serve-a-checkpoint-as-an-endpoint).

Both wires ping through a silent wait. A front drops a connection it reads nothing from (the managed
front after about 90 s), and the pings keep an inference open through that wait.

`/api/v1/models` is an HTTP route and stays on the server's `port`. `InferenceClient.list_models`
refuses a `grpc://` URL.

### Authentication

`PolicyServer(auth_token=...)` gates every route below on `Authorization: Bearer <token>`, answering
`401` on the HTTP route, refusing the WebSocket upgrade before the session opens, and answering
`PERMISSION_DENIED` on the gRPC wire. `serve` — the entry point every vendor CLI exposes — takes that
token from the `AUTH_TOKEN` environment variable, so
a secret never lands in the process arguments. No token serves open, which is the usual shape on a
trusted LAN; an empty one is a broken secret and refuses to start. `InferenceClient.from_url(headers=...)`
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
ws://localhost:8000/api/v1/session?fps=10&horizon_sec=1.0
```

Rules:

- **Values are JSON literals.** The server parses each value as JSON (`10` → int, `false` → bool, `"hello"` → str); a value that does not parse passes through as a plain string, so a hand-typed `?tag=hello` works. The query travels verbatim — `InferenceClient` forwards whatever the URL already says — so a caller who means the string `true` rather than the boolean writes the quoted literal itself, percent-encoded: `?tag=%22true%22`.
- **Imports are rejected.** Overrides are applied with `Config.override_data`, so a value that configuronic would read as an import — `@module.path.Object`, or a leading-dot path relative to the argument's current value — is refused at any nesting depth, and the error names the offending key. Params can tune the pipeline's arguments, never swap its components. A leading-dot string on an argument that gives imports no base to resolve against (a number, a flag, a plain string) is ordinary data and passes through, so `?tag=./data` works.
- **Duplicate keys are rejected.**
- **Params never name a model.** The path does that, and only the path: `/api/v1/session/20000?fps=10` serves model `20000` with that override. A `?model_id=...` param is an ordinary unknown key and is rejected.
- **The model source is fixed at launch.** Params that would change it (e.g. `?source.checkpoint=...`) are rejected; the only way to get a different model is the path.
- **Only config-launched servers accept params.** All vendor servers qualify; a `PolicyServer` built from an already-instantiated pipeline rejects every param.

Any violation — including an unknown key — fails at connect: the server sends `{"status": "error", "error": ...}` and ends the session before anything moves, and the Python client raises `RuntimeError`. Overrides apply per session, and the `local_stack` declared in the ready handshake reflects them.

One string is a complete endpoint description, because the whole session configuration fits in the URL:
`--policy=.remote --policy.url='gpu-host:8000?fps=10'` accepts `host`, `host:port`, and full
`http(s)`/`ws(s)`/`grpc(s)` URLs — optionally with `/api/v1/session/<model_id>` — and forwards the query string verbatim.
Credentials are the exception and stay a separate `headers` argument, so the URL itself is safe to hand around.

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
      {"name": "stop_on_fault"},
      {"name": "chunked_schedule", "args": {"fps": 15.0, "horizon_sec": 1.0}}
    ]},
    "local_codec": {"name": "restrict_image_size", "args": {"width": 224, "height": 224}},
    "compress_images": false,
    "positronic_version": "0.2.1"
  }
}
```

The client ignores all messages until it sees `status == "ready"` (status updates like `loading`/`waiting` may arrive first).

This metadata tells the client:
- Which checkpoint is loaded
- Server connection details
- Codec geometry (`image_sizes`) and scheduler cadence (`action_fps`, `action_horizon_sec`).
- `local_stack` — processor definitions composed by `"seq"`, with the first outermost.
  `RemotePolicy.run` starts these generators and supplies an ordinary remote inference callable.
  `ChunkedSchedule` submits that callable, turns its ordered commands into timed steps, and limits
  the chunk's execution horizon. The harness emits each step's commands immediately.
- `local_codec` — optional data conversions around the remote callable. These run inside submitted
  work, including image resizing. Codec specs support `"seq"` and `"par"` composition.
  Processor and codec names are resolved only through `WIRE_PROCESSORS` and `WIRE_CODECS` in
  `positronic.policy.spec`; an unknown name fails before the policy emits commands.
- `compress_images` — whether the rig JPEG-encodes frames before
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
  "task": "pick up the red cube"
}
```

**Server → Client (Actions):**

For ACT, `result` is a **list** of command dicts, one per action in the predicted chunk. The client
scheduler supplies timing; commands carry no timestamps or end-of-chunk sentinel.
`robot_command` carries the control command, and a rig with more than one arm names the channel per arm
(`robot_command.left`):

```json
{
  "result": [{
    "robot_command": {"type": "cartesian_pos", "pose": [0.51, 0.21, 0.31, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
    "target_grip": 0.02
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

**Server-side recording:** Servers accept an optional `recording_dir`. When set, each session writes a rerun `.rrd` file that taps both sides of the codec: `raw` captures the obs/action at the wire boundary, and `inference` captures the encoded observation and raw model output.

**Python Client:** A Python client (`positronic.offboard.client.InferenceClient`) handles the protocol. The API is in alpha and may change.

## Classes

### `server.PolicyServer`
Serves a `Pipeline` with explicit `source`, `local`, `codec`, and `local_codec` arguments.
`ModelSource.get_models()` backs the catalogue, `resolve()` selects a checkpoint, and `load()` returns
a callable `Model` that owns the loaded resources. The server resets that model at session start.
Server codecs wrap its call; the client receives the processor and local-codec specs in the handshake.

```python
from positronic.offboard.server import PolicyServer
from positronic.offboard.websocket_wire import WebsocketWire
from positronic.policy import Sequential
from positronic.policy.codec import RestrictImageSize
from positronic.policy.spec import Pipeline
from positronic.policy.layers import ChunkedSchedule, StopOnFault

pipeline = Pipeline(
    source=my_model_source,
    local=Sequential(StopOnFault(), ChunkedSchedule(fps=15, horizon_sec=1.0)),
    local_codec=RestrictImageSize(224, 224),
    codec=my_model_codec,
)
server = PolicyServer(pipeline)
server.serve([WebsocketWire('0.0.0.0', 8000, server.api)])
```

`serve` takes the wires that sessions arrive on. Each wire binds its own port, reads its own route for
the model a session asks for, and checks its own session headers. Add `grpc_wire.GrpcWire(host, port)`
to the list to serve gRPC beside the WebSocket. An HTTP wire takes `server.api`, the model catalogue,
and answers it on the port it carries sessions on. A wire asked for port 0 binds any free one and
names it in its `endpoint` property, so `ws.endpoint.port` is the port the wire took.

Passing a `cfn.Config` that builds the pipeline enables [session parameters](#session-parameters);
an instantiated pipeline serves exactly as launched. `idle_timeout_min` ends the server after that
many minutes without activity. Boundary recording through `recording_dir` is not implemented for
callable models or processor runs; passing it raises. Harness episode recording remains available.

### `server.serve`
The CLI entry point every vendor server exposes. A vendor binds `pipeline` to each of its named pipelines and lists the results as subcommands, so `<vendor>-server <pipeline>` launches one. Only `--host`, `--port`, `--grpc_port`, `--recording_dir` and `--idle_timeout_min` are flags of `serve` itself; everything the served model is — codec, source, checkpoint directory — is reached through the pipeline (`--pipeline.source.checkpoints_dir=...`), which is also where a deployment preset binds it.

### `client.InferenceClient`
A Python client for connecting to an inference server. `from_url` reads it off one URL, in the same forms
`RemotePolicy` accepts: an omitted port is the scheme's own, 443 for a TLS scheme and 80 otherwise. The URL
fixes the wire, the model and the session params, so serving another model means another client. The
constructor takes the wire and the session address as values; `wires.CLIENT_WIRES` lists every wire.

```python
from positronic.offboard.client import InferenceClient

# The server's pinned checkpoint, with no session params
client = InferenceClient.from_url('localhost:8000')
# A named model, tuned for every session this client opens
# client = InferenceClient.from_url('localhost:8000/api/v1/session/model_a?fps=10')
# The same session on the gRPC wire, on a LAN and behind a TLS edge
# client = InferenceClient.from_url('grpc://localhost:9000/api/v1/session/model_a')
# client = InferenceClient.from_url('grpcs://gpu-host:443/api/v1/session/model_a')

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
