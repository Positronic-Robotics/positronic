# Positronic Offboard Inference

This package implements the protocol and utilities for offboard policy inference, allowing robots or simulators to stream observations to a remote server and receive actions.

## Protocol v1

The protocol connects clients to callable models. Each deployment declares a client stack of
processors and codecs, with an optional codec around the server call.

### Wires

The protocol is a sequence of msgpack frames, and two wires carry them. Both carry the same frames in
the same order. The client side of each wire, and the facts both ends share, ship as the
`positronic-wire` distribution ([wire/README.md](../../wire/README.md)); this package holds the
server side.

| Wire | `--policy.wire` | Where it answers |
|---|---|---|
| WebSocket | `websocket`, or `websocket_tls` behind a TLS edge | the websocket wire's port, beside the HTTP routes |
| WebSocket on a Unix socket | `websocket_unix` | the websocket wire's socket path, beside the same HTTP routes |
| gRPC | `grpc` | the gRPC wire's own port, sessions alone |
| gRPC over TLS | `grpc_tls` | a TLS edge in front of that same port |

- A client names its wire; nothing reads one off a URL. The WebSocket wire is the default, and a server
  serves gRPC only where `--grpc` names that wire.
- Each server wire carries the address it binds, and `serve` binds what it is given:
  `--websocket.served_address.port=9000` moves the WebSocket wire, and
  `--websocket.served_address=@positronic.offboard.server.socket_at --websocket.served_address.uds=/run/policy.sock` binds it
  to a Unix socket instead. `--grpc=@positronic.offboard.server.grpc --grpc.served_address.port=9001` serves the gRPC wire beside it,
  on an address of its own.
- `websocket_unix` reaches a server on the same machine, over no network, and
  `--policy.address=@positronic.cfg.policy.socket_address --policy.address.uds=…` dials it.
  A caller names the wire and then fills that wire's address, so a socket address carries no host and no
  port at all — on either end. Both paths are absolute: a relative one is resolved against whatever
  directory each side was started from, and the address refuses it. A socket is same-machine by
  construction, so there is no TLS member beside it.
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

`/api/v1/models` is an HTTP route. It answers on the address the WebSocket wire binds: a port, or a
Unix socket. The gRPC port carries sessions alone.

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
Returns the ID of the one checkpoint the server serves. The route answers only after the model loads,
so a probe can read it to find a ready server.

**Example Request:**
```bash
curl http://localhost:8000/api/v1/models
```

**Response:**
```json
{
  "models": ["30000"]
}
```

#### `/api/v1/session`
Establishes an inference session with the server's model. The server loads one checkpoint at startup:
the configured one, or the latest available at that moment. To serve another checkpoint, start another
server.

**Example:**
- `ws://localhost:8000/api/v1/session` → over the WebSocket wire
- `localhost:9000/api/v1/session` → over gRPC

Each wire from the table above carries the same route, and each names the server its own way:
`websocket` and `websocket_tls` a host and a port with a scheme, `grpc` and `grpc_tls` a target, and
`websocket_unix` a socket path and no authority at all. A route below `/api/v1/session` opens no
session.

#### Session parameters

Query params on the session URL tune the served policy pipeline for that one session. Each key is a dotted path into the server's pipeline config — any argument at any depth — applied as a config override before the session is built:

```
ws://localhost:8000/api/v1/session?fps=10&horizon_sec=1.0
```

Rules:

- **Values are JSON literals.** The server parses each value as JSON (`10` → int, `false` → bool, `"hello"` → str); a value that does not parse passes through as a plain string, so a hand-typed `?tag=hello` works. The query travels verbatim — `InferenceClient` forwards whatever the URL already says — so a caller who means the string `true` rather than the boolean writes the quoted literal itself, percent-encoded: `?tag=%22true%22`.
- **Imports are rejected.** Overrides are applied with `Config.override_data`, so a value that configuronic would read as an import — `@module.path.Object`, or a leading-dot path relative to the argument's current value — is refused at any nesting depth, and the error names the offending key. Params can tune the pipeline's arguments, never swap its components. A leading-dot string on an argument that gives imports no base to resolve against (a number, a flag, a plain string) is ordinary data and passes through, so `?tag=./data` works.
- **Duplicate keys are rejected.**
- **The model source is fixed at launch.** Params that would change it (e.g. `?source.checkpoint=...`) are rejected.
- **Only config-launched servers accept params.** All vendor servers qualify; a `PolicyServer` built from an already-instantiated pipeline rejects every param.

Any violation — including an unknown key — fails at connect: the server sends `{"status": "error", "error": ...}` and ends the session before anything moves, and the Python client raises `RuntimeError`. Overrides apply per session, and the `local_stack` declared in the ready handshake reflects them.

The client names each part: `--policy=.remote --policy.wire=websocket --policy.address.host=gpu-host --policy.address.port=8000
--policy.address.query='fps=10'`, and forwards the query string verbatim. Credentials stay a
separate `headers` argument.

### Session Flow

#### 1. Handshake
Upon connection, the server sends a ready packet with metadata:

```json
{
  "status": "ready",
  "protocol_version": 2,
  "session_id": "<server-issued ID>",
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
      {"name": "stop_on_fault", "version": 2},
      {"name": "chunked_schedule", "version": 2, "args": {"fps": 15.0, "horizon_sec": 1.0}},
      {"name": "restrict_image_size", "version": 1, "args": {"width": 224, "height": 224}}
    ]},
    "compress_images": false,
    "positronic_version": "0.2.1"
  }
}
```

The client ignores all messages until it sees `status == "ready"` (status updates like `loading`/`waiting` may arrive first).

The session ID belongs to this connection and is separate from model metadata. `RemotePolicy.run()`
owns the session and includes its ID in each inference request.

This metadata tells the client:
- Which checkpoint is loaded
- Server connection details
- Codec geometry (`image_sizes`) and scheduler cadence (`action_fps`, `action_horizon_sec`).
- `local_stack` — processors and codecs composed by `"seq"`, with the first outermost.
  `RemotePolicy.run` starts these generators and supplies an ordinary remote inference callable.
  `ChunkedSchedule` submits that callable, turns its ordered commands into timed steps, and limits
  the chunk's execution horizon. The harness emits each step's commands immediately.
  A codec outside the scheduler runs on every policy call and decodes each emitted command set,
  preserving the step's wake-up time. A codec inside the scheduler runs with submitted inference
  and decodes whole chunks. Codec specs also support `"par"` composition.
  Processor and codec names and versions resolve only through `COMPONENTS` in
  `positronic.policy.spec`; an unsupported declaration fails before the policy emits commands.
- `compress_images` — whether the rig JPEG-encodes frames before
  sending, for an endpoint behind a proxy with a message-size cap.
  The client sets the quality with `RemotePolicy(jpeg_quality=...)`, 90 by default, and
  records it in the policy metadata.
- `positronic_version` — the server's positronic version, for diagnosing declaration mismatches

#### Compatibility and deprecation

Protocol and component versions are independent positive integers. Missing versions mean **v1**.
The client reads the handshake before sending requests; it never probes an old server with new
messages. Protocol v1 sends observations directly and ends a session by disconnecting. Protocol v2
uses the session-ID envelope and explicit end-session acknowledgement described below. The URL's
`/api/v1` prefix identifies the route; it does not select the inference-message protocol.

Each component leaf declares its own `version`. `seq` and `par` belong to the protocol grammar.
The client selects the exact registered implementation; it never substitutes a newer version or
guesses from constructor arguments. Unsupported versions fail with supported-version information.
`positronic_version` identifies the server build for diagnostics, not compatibility selection.

The client runs a protocol v1 server's stack on the current processors. A new server uses
protocol v2 and the current components.

Published versions have three states in the protocol and component registries:

- **Supported:** an implementation is available without a warning.
- **Deprecated:** the implementation remains available, with an announcement date, earliest removal
  date, and migration instructions. Selection emits a visible warning, once per component version
  in a stack. The notice period is measured in calendar time.
- **Removed:** a later client release explicitly removes the implementation and retains the notice
  and removal date so the error explains how to migrate. Removal cannot precede the announced date.

Dates never disable an installed client. Reaching the earliest removal date leaves a deprecated
version working until an upgrade installs a release that explicitly removes it. No v1 version is
currently deprecated.

To evolve a component, add a `Version(factory)` under a new integer in `COMPONENTS` and set the
emitting class's `WIRE_VERSION`. Keep the old factory and its input/output behavior. Breaking changes
to message shapes or composition grammar instead need a new protocol version. Mark a version
deprecated by attaching `Deprecation(announced_on, remove_after, replacement)` to its registry entry;
removal replaces the factory with `None` and records `removed_on`. Both registries use the same
validation in `positronic.utils.versions`. Compatibility tests cover wire messages and behavior,
including chunk timing and fault cancellation.

#### 2. Status Updates

The server loads its checkpoint before any wire binds, so every session opens on a loaded model. A
load can take minutes (OpenPI and GR00T take 120-300 s), and its progress goes to the server log.

The protocol keeps the `loading` and `waiting` statuses. The client gives the handshake 30 s per
message, and logs each status it receives before `ready`:

```json
{
  "status": "loading",
  "message": "Loading checkpoint 10000, please wait..."
}
```

#### 3. Inference Loop

After handshake, the client streams observations and receives actions:

**Client → Server (Observation):**

Keys are flat strings — the dots are literal, not nesting. Arrays travel as numpy, not base64; a rig behind a message-size cap JPEG-encodes its frames instead (see `compress_images` above). `docs/connect-your-model.md` carries the full key table.

```json
{
  "session_id": "<server-issued ID>",
  "observation": {
    "robot_state.ee_pose": [0.5, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
    "robot_state.q": [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8],
    "grip": 0.04,
    "image.wrist": "<uint8 (H, W, 3)>",
    "image.exterior": "<uint8 (H, W, 3)>",
    "task": "pick up the red cube"
  }
}
```

**Server → Client (Actions):**

For the vendor deployments, `result` is a **list** of command dicts, one per action in the predicted chunk. The client
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

The server passes the encoded observation to `model(obs, session_id=...)`; codecs receive only the
observation. An ID from another connection returns an error and closes the requesting session
before invoking the model. The other session stays open; no replacement session is created automatically.

#### 4. End the session

After inference finishes, closing the policy run sends:

```json
{"session_id": "<server-issued ID>", "end_session": true}
```

The server calls `model.end_session(id)` and acknowledges with the same message. The client then
closes the connection. The loaded model stays available for other sessions. ACT retains no episode
state, so its `end_session` does nothing. A disconnected client also triggers session cleanup after
any running inference finishes. Reconnecting creates a new session ID.

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
  --policy.address.host=localhost --policy.address.port=8000
```

**One checkpoint per server:** A server serves the checkpoint it was launched with. To compare
checkpoints, start one server for each.

**Python Client:** A Python client (`positronic.offboard.client.InferenceClient`) handles the protocol. The API is in alpha and may change.

## Classes

### `server.PolicyServer`
Serves a `PolicyDeployment` with explicit `source`, `local`, and `codec` arguments.
`ModelSource.checkpoint_id()` names the checkpoint the source serves, and `load()` returns a callable
`Model` that owns the loaded resources. The server calls both once, at startup.
Server codecs wrap its call; the client receives one stack spec containing its processors and codecs.

```python
from positronic.offboard.server import PolicyServer
from positronic.offboard.server_wire import ServedHostPort
from positronic.offboard.websocket_wire import WebsocketWire
from positronic.policy import Sequential
from positronic.policy.codec import RestrictImageSize
from positronic.offboard.spec import PolicyDeployment
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable

pipeline = PolicyDeployment(
    source=my_model_source,
    local=Sequential(
        PauseOnUnavailable(), ChunkedSchedule(fps=15, horizon_sec=1.0), RestrictImageSize(224, 224)
    ),
    codec=my_model_codec,
)
server = PolicyServer(pipeline)
server.serve([WebsocketWire(ServedHostPort('0.0.0.0', 8000))])
```

`serve` takes the wires that sessions arrive on. Each wire carries the address it binds, and checks
its own session headers:

```python
from positronic.offboard import grpc_wire, server_wire, websocket_wire

wires = [
    websocket_wire.WebsocketWire(server_wire.ServedHostPort('0.0.0.0', 8000)),
    # or on this machine only: websocket_wire.WebsocketWire(websocket_wire.ServedUnixSocket(Path('/run/p.sock')))
    grpc_wire.GrpcWire(server_wire.ServedHostPort('0.0.0.0', 8001)),
]
server.serve(wires)
```

`serve` hands every wire the HTTP routes it owns; a wire whose transport carries HTTP answers them
beside its sessions, and the gRPC wire, whose port carries sessions alone, does not. A wire names
where it bound in its `served_address` property: `ServedHostPort` for a host and a port — a wire
asked for port 0 binds any free one, and `ws.served_address.port` is the port it took — or
`ServedUnixSocket` for a socket path, which has no port at all.

Passing a `cfn.Config` that builds the pipeline enables [session parameters](#session-parameters);
an instantiated pipeline serves exactly as launched. `idle_timeout_min` ends the server after that
many minutes without activity.

### `server.serve`
The CLI entry point every vendor server exposes. A vendor binds `pipeline` to each of its named pipelines and lists the results as subcommands, so `<vendor>-server <pipeline>` launches one. Only `--websocket`, `--grpc` and `--idle_timeout_min` are flags of `serve` itself — each wire carries the address it binds, so `--websocket.served_address.port=9000` moves one and `--grpc=@positronic.offboard.server.grpc` adds the other; everything the served model is — codec, source, checkpoint — is reached through the pipeline, which is also where a deployment preset binds it. Select GR00T checkpoints with `--pipeline.source.model_source=...`; LeRobot and OpenPI use `--pipeline.source.checkpoints_dir=...`.

### `client.InferenceClient`
A Python client for connecting to an inference server. It takes the wire and the address that wire
dials (`positronic_wire.registry.CLIENT_WIRES` lists every wire by name); the address fixes the server
and the session params. Each wire names its own address type, and the client refuses one built for
another wire.

```python
from pathlib import Path

from positronic.offboard.client import InferenceClient
from positronic_wire import registry
from positronic_wire.wire import SESSION_PATH, HostPortAddress, UnixSocketAddress

# The server's model, with no session params
client = InferenceClient(registry.client_wire('websocket'), HostPortAddress('localhost', 8000, SESSION_PATH, ''))
# The same model, tuned for every session this client opens
# client = InferenceClient(registry.client_wire('websocket'), HostPortAddress('localhost', 8000, SESSION_PATH, 'fps=10'))
# The same session on the gRPC wire, on a LAN and behind a TLS edge
# client = InferenceClient(registry.client_wire('grpc'), HostPortAddress('localhost', 9000, SESSION_PATH, ''))
# client = InferenceClient(registry.client_wire('grpc_tls'), HostPortAddress('gpu-host', 443, SESSION_PATH, ''))
# A server on this machine: the socket wire's address names the socket, and no host and no port
# client = InferenceClient(registry.client_wire('websocket_unix'), UnixSocketAddress(Path('/run/policy.sock'), SESSION_PATH, ''))

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
- **LeRobot (0.3.3)**: `positronic.vendors.lerobot_0_3_3.server` - Serves ACT checkpoints
- **GR00T**: `positronic.vendors.gr00t.server` - Serves GR00T checkpoints with modality config
- **OpenPI**: `positronic.vendors.openpi.server` - Serves OpenPI checkpoints with config name
- **DreamZero**: `positronic.vendors.dreamzero.server` - Serves DreamZero checkpoints through a torchrun subprocess
- **MolmoAct2**: `positronic.vendors.molmoact2.server` - Serves the pretrained MolmoAct2 DROID model

## See Also

- [Training Workflow](../../docs/training-workflow.md) - Starting inference servers
- [Inference Guide](../../docs/inference.md) - Remote policy usage and patterns
- [Model Selection](../../docs/model-selection.md) - Choosing between vendors
