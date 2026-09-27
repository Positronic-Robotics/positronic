# Measuring where a round trip's time goes

A served round trip divides into what the server reports spending and what it does not. The server's
`served_ms` covers the observation's decode, the queue and the model. Everything else — the link, and
whether the receiver drains it — is outside every figure the server sends.

Each step below answers on its own, and every step that needs no model runs before the model is
loaded, so a session cut short still holds its results.

## What each probe answers

| Probe | Answers |
|---|---|
| `link_probe facts` | the MTU, the network namespace and the socket buffers of wherever it runs |
| `link_probe sink` | when the first read returned, when the last one did, and every read between |
| `link_probe source` | how long this end's own write took to return, and the send buffer it wrote into |
| `link_probe watch` | the receive queue over the transfer |
| `serving_cost --server_address` | one round trip, divided by the phases the server reports |

`link_probe` carries no policy and no model. `serving_cost` needs a served handshake, so it runs last.

## What you need

- One GPU host with Docker, which the client reaches directly on ports 8000, 9000, 9100, 9101 and 9102.
- A shell on that host, and a shell **in the receiver's network namespace**. A container has its own,
  and the host's readings do not describe it. `docker exec -it link-probe-server bash` opens one, and
  steps 3 and 5 use two at a time. Where the server's image cannot run the probe, join a second
  container to its namespace instead: `docker run --network container:link-probe-server <image> python
  -m positronic.offboard.link_probe watch --port=8000`.
- A recorded episode of the rig the server serves, for `serving_cost`.

Set these once in each shell:

```bash
SERVER=<host the server runs on>      # as the client reaches it
CLIENT=<address of the client>        # as the server sees it
EPISODE=<dataset path>                 # one recorded episode
PROBE="uv run --locked python -m positronic.offboard.link_probe"
```

## 1. Bring up the box and the container

Start one of the servers that [docs/inference.md](../../docs/inference.md) names, in a terminal of its own
on the host. Keep its subcommand and model flags, and add the container name, the probe ports and a gRPC
wire on 9000:

```bash
cd docker && docker compose run --rm --service-ports --name link-probe-server \
    -p 9000:9000 -p 9100:9100 -p 9102:9102 <server> <subcommand> <model flags> \
    --grpc=@positronic.offboard.server.grpc --grpc.served_address.port=9000
```

`--service-ports` publishes the websocket on 8000. The container gets no `AUTH_TOKEN`, so the server
serves open and the client sends no token.

The server runs in the foreground. It holds its terminal until step 7 stops it, and its log prints there.
The model load starts here and takes tens of minutes. Do not wait for it. Open the other shells in new
terminals, and run steps 2 and 3 while it loads.

## 2. Read the namespace, on both sides

```bash
$PROBE facts --peer=$SERVER          # on the client
$PROBE facts --peer=$CLIENT          # in the container
```

Record the MTU of the interface each side sends through, both namespaces, and `tcp_rmem`. A tunnel
carries less than an ethernet link, and a container's namespace is not its host's. A connection's
receive buffer starts at the middle value of `tcp_rmem`, and the kernel grows it up to the top value. A
payload larger than the starting buffer waits in the sender's queue until the receiver reads.

## 3. Measure the read, before the model is up

Start a sink in the container and a second one on the host, on different ports. The same client drives
both, so the two differ only by the path into the container. Each sink serves until you stop it with
Ctrl-C, so give each one a shell of its own. The watcher samples for 120 seconds: start it in a second
container shell, then run both sources from the client before it ends.

```bash
# container shell 1
$PROBE sink --port=9100
# host shell
$PROBE sink --port=9101
# container shell 2, beside the sink under test
$PROBE watch --port=9100 --interval_ms=20 --seconds=120 --out=recvq-container.jsonl

# from the client, while the watcher runs, against each sink
$PROBE source --host=$SERVER --port=9100 --kib=750 --transfers=10 --out=into-container.json
$PROBE source --host=$SERVER --port=9101 --kib=750 --transfers=10 --out=into-host.json
```

Read `sndbuf_bytes` first. Where it holds the payload whole, `sendall` returns before the receiver
reads, so `write_ms` cannot show a late reader and `report_ms` carries it. Otherwise read the pair this
way:

- **`write_ms` high, `recv_q` high** — the receiver is not draining. The bytes arrived and sat.
- **`write_ms` high, `recv_q` near zero** — the bytes are not arriving. The path is the cost.
- **`read_span_ms` far below `write_ms`** — the reader started late and then caught up at full speed.

Then sweep the payload, which says whether the cost scales with bytes or is a fixed stall:

```bash
for KIB in 128 750 2026; do $PROBE source --host=$SERVER --port=9100 --kib=$KIB --transfers=10; done
```

And raise a reader that competes for the interpreter, as a busy model does to the loop that
reads for it:

```bash
# container shell 2; stop it with Ctrl-C after the source ends, because its threads load every core
$PROBE sink --port=9102 --busy_threads=$(nproc)
# from the client
$PROBE source --host=$SERVER --port=9102 --kib=750 --transfers=10
```

## 4. Load the model, once

Wait for the handshake. `serving_cost` opens a session, so it reports the load is finished by running
at all.

## 5. Divide a served round trip, on both wires

Same episode, same server, one wire each. The two wires report the same cost in different halves: a
websocket `send` returns once the bytes are written, and a gRPC `send` returns before that, so an
undrained receiver shows up in `send_ms` on one and in `recv_ms` on the other.

```bash
# in the receiver's namespace: one watcher per wire, both in the background, over both runs
$PROBE watch --port=8000 --interval_ms=20 --seconds=300 --out=recvq-served-ws.jsonl &
$PROBE watch --port=9000 --interval_ms=20 --seconds=300 --out=recvq-served-grpc.jsonl &

# from the client, while the watchers run
uv run --locked python -m positronic.offboard.serving_cost \
    --dataset.path=$EPISODE --server_wire=websocket \
    --server_address=@positronic.cfg.policy.network_address --server_address.host=$SERVER \
    --server_address.port=8000 --requests=20 --out=served-ws.json
uv run --locked python -m positronic.offboard.serving_cost \
    --dataset.path=$EPISODE --server_wire=grpc \
    --server_address=@positronic.cfg.policy.network_address --server_address.host=$SERVER \
    --server_address.port=9000 --requests=20 --out=served-grpc.json

# in the receiver's namespace, after both runs: let the watchers finish and write their files
wait
```

Each run prints the stack it rebuilt from the handshake. Read it: the rig sends that.

Read `wire_kib` before anything else. The observation came to that on the wire, and if it is not
close to what the rig sends in production, the episode is not standing in for the rig and no figure
under it compares to one. `round_trip_ms - pack_ms - served_ms` is the link and the receiver, and
`send_ms` against `recv_ms` says which half of the wire holds it.

## 6. Test the countermeasures the readings point at

Run these two whatever the readings say.

**A sink during a real inference.** Run step 3's sink in the container. Repeat one of step 5's
`serving_cost` runs, and from a second terminal on the client drive the sink while it runs:

```bash
$PROBE source --host=$SERVER --port=9100 --kib=750 --transfers=10 --out=into-container-serving.json
```

A sink that stalls too says the whole box is busy; a sink that reads at full speed while the server
does not says the cost is inside the server process.

**A larger starting receive buffer.** A connection's receive buffer starts at the middle value of
`tcp_rmem`, 131072 bytes by default. A 750 KiB payload does not fit there, so most of it waits in the
sender's queue until the receiver reads. Raise the middle value to 1 MiB, which holds the payload. A
container shell cannot write it, because Docker mounts `/proc/sys` read-only there. Each network
namespace has its own `tcp_rmem`, so a privileged container in the server's namespace writes it. The
command reads the current values and writes the minimum and the top value back unchanged:

```bash
# on the host
docker run --rm --privileged --network container:link-probe-server alpine sh -c \
    'set -- $(cat /proc/sys/net/ipv4/tcp_rmem); sysctl -w net.ipv4.tcp_rmem="$1 1048576 $3"'
```

`net.core.rmem_max` does not apply, because these receivers never set `SO_RCVBUF`. A connection takes
its starting buffer from its listening socket, which reads `tcp_rmem` when it opens. Restart the sinks in
the container, then repeat step 3. A larger buffer changes the measurement only before a connection first
reads, or on a link where it grows more slowly than the payload arrives.

Compare `write_ms + report_ms`, because a larger receive buffer can move the wait from one to the other.
A sum that falls to what the link needs proves the cost is the receiver's scheduling, and that
buffering absorbs it. A sum that does not move rules the buffer out.

Run each of these only where the readings above point at it:

| Reading | Countermeasure | What it would prove |
|---|---|---|
| the server process stalls, the sink beside it does not | serve the wire on its own event loop, or its own process | the read competes with the inference for one interpreter |
| the cost scales with payload size | send fewer frames, or a smaller bound, per `serving_cost`'s flags | the cost is bytes, not scheduling |
| the container is slower than its host | give the container the host's network namespace | the container's network path is the cost |

## 7. Tear down, and verify it is gone

```bash
# on the host: stop the host sink from step 3 with Ctrl-C, then
docker stop link-probe-server                      # --rm then removes the container
docker ps -a --filter name=link-probe-server       # prints no container
```

A stopped container is not a deleted box. If the host was created for this session, delete it with its
provider, then list the provider's resources and read the count, because a deleted box can leave its
disk.

## What a local rehearsal cannot tell you

Every probe here runs on one box over loopback, which is worth doing before the window and proves the
commands work. It does not reproduce the reading: loopback buffers absorb a 750 KiB payload whole, so a
reader that stalls never shows up in `write_ms`. A reader that starts late shows up in `report_ms`, and
one that stalls between its reads shows up in `read_span_ms`. A link with a smaller bandwidth-delay
product pushes the same stall back to the sender, and only a reading taken at both ends divides it.
