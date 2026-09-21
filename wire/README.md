# positronic-wire

The client side of the two transports a Positronic inference session runs over, and the facts both
ends of a wire share. It is one distribution, installable on its own, with `grpcio` and
`websockets` as its only dependencies.

> **Alpha, under rapid development.** Names and behaviour change without notice, and nothing here is
> covered by a backwards-compatibility guarantee. Pin the exact version you tested against.

```bash
uv add "positronic-wire==0.1.0"
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
types, its status codes, its metadata keys, the path a probe asks for, the URL schemes that select
it — lives inside that transport's two wire classes. Code outside a wire speaks to every transport
through one interface and reads one answer.

## The package boundary

| Module | Holds |
|---|---|
| `positronic_wire.wire` | The routes (`API_PATH`, `SESSION_PATH`, `MODELS_PATH`), `MAX_MESSAGE_BYTES`, `SessionAddress`, `Endpoint`, `Scheme`, `Refusal`, `ConnectRefused`, `PeerDisconnected`, and the abstract `ClientWire` and `ClientConnection` |
| `positronic_wire.websocket` | `WebsocketClientWire`, `WebsocketClientConnection` |
| `positronic_wire.grpc` | `GrpcClientWire`, `GrpcClientConnection`, and the call both ends agree on: `SERVICE`, `METHOD_PATH`, `PROBE_PATH`, `SESSION_PATH_HEADER`, `SESSION_QUERY_HEADER`, `MESSAGE_SIZE_OPTIONS`, `PING_EVERY_MS` |
| `positronic_wire.wires` | `CLIENT_WIRES`, `BY_SCHEME`, and `from_url(url) -> (ClientWire, SessionAddress)` |

`positronic.offboard` keeps the server side: `server_wire.Wire` and `server_wire.ServerConnection`,
`websocket_wire.WebsocketWire`, `grpc_wire.GrpcWire`, the session protocol, `InferenceClient` and
`PolicyServer`. The server side depends on `fastapi`, `uvicorn` and `grpc.aio`, which no client
needs.

## The client interface

`ClientWire` has four verbs:

- `schemes()` — every URL scheme that selects this wire, each saying whether it names TLS.
- `api_url(address)` — the server's HTTP API beside this wire, or `None` where the wire's port
  carries sessions alone.
- `dial(address, headers, open_timeout)` — a client's end of one session. It raises
  `ConnectRefused` when the session does not open, whatever refused it. The `refusal` on the
  exception says what the caller does next: `COLD` retries, `FORBIDDEN` retries a few times,
  `FINAL` surfaces at once.
- `probe(address, open_timeout)` — whether a server answers at the address, without opening a
  session. `None` when one does; a `Refusal` says why none did, in the terms `dial` uses. The
  websocket wire asks the host's root for an upgrade, which no server grants, and reads any status
  as an answer. The gRPC wire waits for the channel to become ready, or for `PROBE_PATH` to answer
  `UNIMPLEMENTED`.

`wires.from_url(url)` selects the wire a URL names and parses the session address. A consumer that
holds a URL calls it once and then speaks to the wire it got back.

## What each consumer pays

| Consumer | Needs | Installs |
|---|---|---|
| A rig client, and `positronic` itself | Every verb, the session protocol, the policy stack | `positronic`, which pins `positronic-wire` exactly |
| A coordinator that probes an endpoint and warms it | `from_url`, `probe`, `PROBE_PATH`, the scheme table, the routes | `positronic-wire` alone: `grpcio`, `websockets` and nothing else |
| A service that validates an endpoint URL | `BY_SCHEME` | `positronic-wire` alone |
| A server | The server side | `positronic` |

A consumer whose lockfile already carries `grpcio` (through a cloud SDK) and `websockets` (through
`uvicorn`) adds no third-party package when it adds this one.

## What the wire keeps out of its consumers

A transport-specific dial disappears from a consumer once it speaks to the wire. A consumer that opens a `websockets.connect` of its own to wake an
endpoint has to skip every gRPC endpoint, because the same call cannot reach one; `probe` reaches
both. A consumer that classifies a failed dial by matching library exception names over the
raised type's bases has to list `grpc.RpcError`, `websockets.*` and `httpx.*`; `dial` raises one
`ConnectRefused` for all of them, so `isinstance` answers.

A copied fact disappears too. The gRPC probe path, the session route, the models route, the scheme lists and the
scheme-to-TLS map are each one symbol here. A consumer imports the symbol or derives its table
from `BY_SCHEME` (`{s.text for s in wire.schemes()}` over `CLIENT_WIRES`). A literal spelled a
second time in another repository drifts the day either side edits it, and nothing reports the
drift; an imported symbol cannot.

The port a server serves a wire on, and the server flag that names it, stay literals where they are
spelled: they are a deployment's configuration, not wire facts. A
test in the consumer that installs `positronic` pins the flag name against the server's own
signature.

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
of guarding it with a version floor. Each side reports the schemes its wire lists, and the side
that hands a URL across refuses one whose scheme the other side does not list. A transport added
to the wire is then one entry in `CLIENT_WIRES`, and no consumer has to raise a constant naming
the version that first dialled it.

## Adopting the wire in a consumer

A consumer moves onto the wire in this order, each step green on its own:

1. Depend on `positronic-wire`. Import the routes, the probe path and the scheme table; delete the
   local copies and the tables keyed by scheme. Report the wire's schemes in the deploy handshake
   and retire any per-transport version floor.
2. Replace the transport-specific dial with `probe`, and the exception-name match with
   `isinstance(raised, (wire.ConnectRefused, wire.PeerDisconnected, TimeoutError))`.
3. Move the consumer's own transports — a partner's wire, a socket-path address — into their own
   `ClientWire` subclasses, or keep them in the consumer's own scheme set unioned with the wire's.
4. Deploy the pinned side at the commit that carries the wire, then the checkout side.

## What is not shared

- The server side of each wire, which serves through `fastapi`, `uvicorn` and `grpc.aio`.
- The session protocol and the policy stack, which shape an observation and need `numpy` and the
  codecs. A consumer that warms an endpoint with a real observation runs that in an environment
  carrying `positronic`.
- A transport this package does not implement. A consumer that owns one writes it as a `ClientWire`
  of its own.
