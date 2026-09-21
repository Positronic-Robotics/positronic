# positronic-wire

The client side of the two transports a Positronic inference session runs over, and the facts both
ends of a wire share. It is one distribution, installable on its own, with `grpcio` and
`websockets` as its only dependencies.

> **Alpha, under rapid development.** Names and behaviour change without notice, and nothing here is
> covered by a backwards-compatibility guarantee. Pin the exact version you tested against.

```bash
uv add "positronic-wire==0.2.0"
uv add "positronic-wire @ git+https://github.com/Positronic-Robotics/positronic@<tag or commit>#subdirectory=wire"
```

The package is not on PyPI yet. Until it is, use the second line, pinned to a tag or a commit.

`positronic_wire` never imports `positronic`. `positronic` depends on it: the server side of each
wire, in `positronic.offboard`, imports the facts from here, and `InferenceClient` dials through
the wires here.

## What a wire is

A wire is one transport that carries the `positronic.offboard.protocol` frames as opaque bytes. It
has two ends. The client end dials a session and probes a server; the server end accepts sessions
and refuses an unauthorized peer. Everything specific to a transport — the library, its exception
types, its status codes, its metadata keys, the path a probe asks for, the URL scheme it writes —
lives inside that transport's wire classes. Code outside a wire speaks to every transport through
one interface and reads one answer. A wire over TLS is a member of its own, so no caller holds a
`secure` flag, and nothing anywhere reads a transport off a URL scheme: a caller names the wire.

## The package boundary

| Module | Holds |
|---|---|
| `positronic_wire.wire` | The routes (`API_PATH`, `SESSION_PATH`, `MODELS_PATH`) and `session_path(model)`, `MAX_MESSAGE_BYTES`, `SessionAddress(host, port, path, query, uds)`, `Endpoint`, `Refusal`, `ConnectRefused`, `PeerDisconnected`, and the abstract `ClientWire` and `ClientConnection` |
| `positronic_wire.websocket` | `WebsocketClientWire`, `WebsocketTlsClientWire`, `WebsocketUnixClientWire`, `WebsocketClientConnection` |
| `positronic_wire.grpc` | `GrpcClientWire`, `GrpcTlsClientWire`, `GrpcClientConnection`, and the call both ends agree on: `SERVICE`, `METHOD_PATH`, `PROBE_PATH`, `SESSION_PATH_HEADER`, `SESSION_QUERY_HEADER`, `MESSAGE_SIZE_OPTIONS`, `PING_EVERY_MS` |
| `positronic_wire.registry` | `CLIENT_WIRES`, every member by its `NAME`, and `client_wire(name)` |

`positronic.offboard` keeps the server side: `server_wire.Wire` and `server_wire.ServerConnection`,
`websocket_wire.WebsocketWire`, `grpc_wire.GrpcWire`, the session protocol, `InferenceClient` and
`PolicyServer`. The server side depends on `fastapi`, `uvicorn` and `grpc.aio`, which no client
needs.

## The client interface

`NAME` is what a caller selects a wire by: `websocket`, `websocket_tls`, `websocket_unix`, `grpc`,
`grpc_tls`. `DEFAULT_PORT` is the port a URL leaves out. The verbs follow.

- `session_url(address)` — the session as this wire names it, for a log and for an error. The
  websocket members write `ws://` or `wss://`, `websocket_unix` writes `ws+unix://`, and the gRPC
  members write `host:port`. What a wire dials is its own: `websocket` and `websocket_tls` dial this
  spelling, and the others dial a socket or a target instead.
- `api_url(address)` — the server's HTTP API beside this wire (`http://` or `https://`), or `None`
  where the wire's port carries sessions alone.
- `api_socket(address)` — the Unix socket `api_url` answers on, or `None` where it answers over the
  network. Only `websocket_unix` names a socket, so a caller reaches the API through the wire and
  never reads `uds` itself.
- `dial(address, headers, open_timeout)` — a client's end of one session. It raises
  `ConnectRefused` when the session does not open, whatever refused it. The `refusal` on the
  exception says what the caller does next: `COLD` retries, `FORBIDDEN` retries a few times,
  `FINAL` surfaces at once.
- `probe(address, headers, open_timeout)` — whether a server answers at the address, without opening
  a session. It carries the same `headers` as `dial`, so an edge that authenticates on them lets the
  probe through to the server behind it: the probe wakes what a session would reach. `None` when a
  server answers; a `Refusal` says why none did, in the terms `dial` uses, `FORBIDDEN` among them for
  a credential the edge refused. The websocket wire asks the host's root for an upgrade, which the
  server refuses with 403 and nothing else answers 403 there. The gRPC wire calls `PROBE_PATH`,
  which a server that is up answers `UNIMPLEMENTED`.

`registry.client_wire(name)` is the one lookup, and it refuses a name no wire carries. A
`SessionAddress` is `host`, `port`, `path` (`session_path(model)`), `query` and `uds`, as written; a
caller that records an endpoint records those five and the wire's name, never a URL.

`uds` is the fifth field, and only `websocket_unix` reads it: an absolute path to a
Unix socket a server on the same machine bound, dialled instead of the network. `host` still stands for
the server in the handshake sent over that socket, and no port is claimed there. A socket is
same-machine by construction, so no TLS member sits beside it. An `OSError` the socket raises is `COLD`
only where the path could still become a socket — it is absent, or a refusal comes from a socket a
server is restarting on; a misspelt path, a path holding something that is not a socket and a refused
permission are `FINAL`, because no retry reaches them.

## What each consumer pays

| Consumer | Needs | Installs |
|---|---|---|
| A rig client, and `positronic` itself | Every verb, the session protocol, the policy stack | `positronic`, which pins `positronic-wire` exactly |
| A coordinator that probes an endpoint and warms it | `registry.client_wire`, `probe`, `PROBE_PATH`, the routes | `positronic-wire` alone: `grpcio`, `websockets` and nothing else |
| A service that validates an endpoint record | `registry.CLIENT_WIRES` | `positronic-wire` alone |
| A server | The server side | `positronic` |

A consumer whose lockfile already carries `grpcio` (through a cloud SDK) and `websockets` (through
`uvicorn`) adds no third-party package when it adds this one.

## What the wire keeps out of its consumers

A transport-specific dial disappears from a consumer once it speaks to the wire. A consumer that
opens a `websockets.connect` of its own to wake an endpoint has to skip every gRPC endpoint, because
the same call cannot reach one; `probe` reaches both. A consumer that classifies a failed dial by
matching library exception names over the raised type's bases has to list `grpc.RpcError`,
`websockets.*` and `httpx.*`; `dial` raises one `ConnectRefused` for all of them, so `isinstance`
answers.

A copied fact disappears too. The gRPC probe path, the session route and the models route are each
one symbol here, and the set of wires is `registry.CLIENT_WIRES`. A literal spelled a second time
in another repository drifts the day either side edits it, and nothing reports the drift; an
imported symbol cannot.

A scheme table disappears with them. A record that names an endpoint carries the wire's name, the
host, the port, the model and the query as five fields — and the socket path as a sixth, where the
wire is `websocket_unix` — so no reader of the record derives a transport, a TLS setting or a
default port from the spelling of a URL.

The port a server serves a wire on, and the server flag that names it, stay literals where they are
spelled: they are a deployment's configuration, not wire facts. A test in the consumer that installs
`positronic` pins the flag name against the server's own signature.

## Versioning and skew

`positronic` pins `positronic-wire==<version>` exactly, as it pins `positronic-platform-client`.
`utilities/check_workspace_version_bump.py` refuses a change under `wire/` that reuses the version,
and one that leaves the root's pin behind. The release workflow publishes `wire/` to the index
before `positronic`, with `skip-existing`, so an unchanged wire is a no-op and a changed one is on
the index before the release that depends on it.

A consumer outside this repository pins the wire the way it pins the client: by the index once the
wire is published there, and until then by git subdirectory at a commit
(`positronic-wire @ git+…positronic@<sha>#subdirectory=wire`). A consumer that also installs
`positronic` at a commit names the same commit for both, so the wire the git install provides is
the version `positronic` requires.

Two deployments of one consumer can hold different wire versions: a rig runs a pinned console on
its own schedule while a coordinator runs its checkout. The wire keeps that skew readable instead
of guarding it with a version floor. Each side reports the names its registry lists, and the side
that hands an endpoint record across refuses a wire name the other side's registry lacks. A
transport added to the wire is then one entry in `registry.CLIENT_WIRES`, and no consumer has to
raise a constant naming the version that first dialled it.

## Adopting the wire in a consumer

A consumer moves onto the wire in this order, each step green on its own:

1. Depend on `positronic-wire`. Import the routes, the probe path and the registry; delete the
   local copies, the tables keyed by scheme, and every read of a URL scheme. An endpoint record
   names its wire, host, port, model and query. Report the registry's names in the deploy handshake
   and retire any per-transport version floor.
2. Replace the transport-specific dial with `probe`, and the exception-name match with
   `isinstance(raised, (wire.ConnectRefused, wire.PeerDisconnected, TimeoutError))`.
3. Move the consumer's own transports — a partner's wire, a socket-path address — into their own
   `ClientWire` subclasses, registered by name beside the wire's own.
4. Deploy the pinned side at the commit that carries the wire, then the checkout side.

## What is not shared

- The server side of each wire, which serves through `fastapi`, `uvicorn` and `grpc.aio`.
- The session protocol and the policy stack, which shape an observation and need `numpy` and the
  codecs. A consumer that warms an endpoint with a real observation runs that in an environment
  carrying `positronic`.
- A transport this package does not implement. A consumer that owns one writes it as a `ClientWire`
  of its own.
