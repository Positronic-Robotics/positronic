# positronic-wire

The client side of the two transports a Positronic inference session runs over, and the facts both
ends of a wire share. It is one distribution, installable on its own, with `grpcio` and
`websockets` as its only dependencies.

> **Alpha, under rapid development.** Names and behaviour change without notice, and nothing here is
> covered by a backwards-compatibility guarantee. Pin the exact version you tested against.

```bash
uv add "positronic-wire==0.4.0"
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
| `positronic_wire.wire` | The routes (`API_PATH`, `SESSION_PATH`, `MODELS_PATH`) and `session_path(model)`, `MAX_MESSAGE_BYTES`, the addresses `HostPortAddress(host, port, path, query)` and `UnixSocketAddress(uds, path, query)` under the abstract `SessionAddress`, `netloc`, `Refusal`, `ConnectRefused`, `PeerDisconnected`, and the abstract `ClientWire` and `ClientConnection` |
| `positronic_wire.websocket` | `WebsocketClientWire`, `WebsocketTlsClientWire`, `WebsocketUnixClientWire`, `WebsocketClientConnection` |
| `positronic_wire.grpc` | `GrpcClientWire`, `GrpcTlsClientWire`, `GrpcClientConnection`, and the call both ends agree on: `SERVICE`, `METHOD_PATH`, `PROBE_PATH`, `SESSION_PATH_HEADER`, `SESSION_QUERY_HEADER`, `MESSAGE_SIZE_OPTIONS`, `PING_EVERY_MS` |
| `positronic_wire.registry` | `CLIENT_WIRES`, every member by its `NAME`, and `client_wire(name)` |

`positronic.offboard` keeps the server side: `server_wire.Wire` and `server_wire.ServerConnection`,
`websocket_wire.WebsocketWire`, `grpc_wire.GrpcWire`, the session protocol, `InferenceClient` and
`PolicyServer`. The server side depends on `fastapi`, `uvicorn` and `grpc.aio`, which no client
needs.

## The client interface

A caller selects a wire by `NAME`: `websocket`, `websocket_tls`, `websocket_unix`, `grpc`, `grpc_tls`,
`roboarena`. A wire dials an address of its `ADDRESS` type, and `DEFAULT_PORT` names the port a URL
leaves out on the members that carry one.

- `session_url(address)` — the session as this wire names it, for a log and for an error. The
  websocket members write `ws://` or `wss://`, `websocket_unix` writes `ws+unix://`, and the gRPC
  members write `host:port`, and `roboarena` writes the root it dials. What a wire dials is its own:
  `websocket`, `websocket_tls` and `roboarena` dial this spelling, and the others dial a socket or a
  target instead.
- `list_models(address, headers, open_timeout)` — the models the server serves, read on the
  transport that carries this wire's sessions: over HTTP for the network members, and over the
  socket itself for `websocket_unix`. It carries the same `headers` as `dial`, so an edge that
  authenticates on them lets the read through, and refuses in `dial`'s own vocabulary. The gRPC
  members raise `ValueError`: their port carries sessions alone. A server that answers the catalogue
  serves an HTTP-capable wire beside the gRPC one. `roboarena` raises too, because a partner's
  endpoint is the one model it serves. No caller builds a URL or a transport.
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
  which a server that is up answers `UNIMPLEMENTED`. The roboarena wire opens the root and reads the
  frame the server announces itself with, which is the only readiness the protocol carries.

`registry.client_wire(name)` is the one lookup, and it refuses a name no wire carries.

**Each wire declares the address it dials, and takes no other.** `ClientWire.ADDRESS` names that
type, and every verb above takes it. `websocket`, `websocket_tls`, `grpc` and `grpc_tls` take a
`HostPortAddress` — `host`, `port`, `path` (`session_path(model)`), `query`, as written.
`websocket_unix` takes a `UnixSocketAddress` — `uds`, `path`, `query`. `roboarena` takes a
`RoboarenaAddress` — `host` and `port`. A record that names an endpoint carries the wire's name and
that wire's fields, never a URL, and no address carries a field a wire ignores.

`uds` is an absolute path to a Unix socket a server on the same machine bound, dialled instead of
the network. The address refuses a relative path when it is built, because a relative one names a
different socket to each caller. It names no host and no port, because a socket has neither: the
handshake carries `localhost` as a stand-in the server never resolves. A socket is same-machine by
construction, so no TLS member sits beside it. An absent path is `COLD`: the client cannot tell a
misspelt path from a socket nobody has bound yet, so it retries either to its deadline. A refusal
from a socket a server is restarting on is `COLD` too. A path holding something that is not a
socket, and a refused permission, are `FINAL`, because no retry reaches them. A handshake that timed
out or was reset reached the socket, so the server rather than the path was not ready, and it reads
`COLD` as it does on a port.

`roboarena` is a partner's own protocol: msgpack frames on a websocket at the bare root of a port the
partner publishes. The server closes any other path, and it routes on a key inside each frame, so the
address names no route and no query. It publishes no default port either, so every address states one.
The server announces its configuration as the first frame of every connection: `dial` leaves that frame
for the caller's codec, and `probe` reads it and closes. A port that accepts a connection and announces
nothing is a backend still starting, so it reads `COLD`. The protocol names no URL scheme and the port is
plain, so no TLS member sits beside this wire; a partner who terminates TLS in front of it refuses a
handshake in the terms the websocket members already read.

Typing carries the split: a wire handed the other wire's address is a type error at the call site.
`registry.client_wire(name)` answers by name and cannot, so `InferenceClient` checks `ADDRESS` once,
before it dials, and names both in the refusal.

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
   local copies, the tables keyed by scheme, and every read of a URL scheme. **An endpoint record
   names its wire, then that wire's address** — host, port, model and query for a network wire, the
   socket path for `websocket_unix`. Report the registry's names in the deploy handshake and retire
   any per-transport version floor.
2. Replace the transport-specific dial with `probe`, and the exception-name match with
   `isinstance(raised, (wire.ConnectRefused, wire.PeerDisconnected, TimeoutError))`.
3. Move the consumer's own transports into their own `ClientWire` subclasses, each declaring the
   address it dials, registered by name beside the wire's own.
4. Deploy the pinned side at the commit that carries the wire, then the checkout side.

## What is not shared

- The server side of each wire, which serves through `fastapi`, `uvicorn` and `grpc.aio`.
- The session protocol and the policy stack, which shape an observation and need `numpy` and the
  codecs. A consumer that warms an endpoint with a real observation runs that in an environment
  carrying `positronic`.
- A transport this package does not implement. A consumer that owns one writes it as a `ClientWire`
  of its own.
