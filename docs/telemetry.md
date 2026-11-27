# Eval telemetry

Wall-clock telemetry for `positronic eval run`. A sim eval runs on a virtual clock, so the virtual
time a rollout advances says nothing about the real compute it cost. This telemetry captures that operational
signal — the wall-clock split of each phase (reset, env step, inference, record IO), the machine's CPU / memory
/ GPU load, and the inference-latency distribution — so a sizing or performance pass can read what a rollout
actually spent.

It is **observability, not data** (see `architecture.md`, "Telemetry is observability, not data"). The dataset
records the robot's world under the run's (possibly virtual) clock; telemetry describes the machinery around it
in wall time. So it lives in **sidecar files next to the dataset, never inside it**, and anything derived from
them is an offline reduce over the raw files rather than a second recording.

The boundary holds in the other direction too: nothing under `positronic/dataset` imports telemetry, its
vocabulary, or `opentelemetry`, and `positronic/dataset/tests/test_telemetry_boundary.py` fails the build if
one starts to. Code that needs timing there takes it as an opaque injected context factory — `DsWriterAgent`'s
`telemetry_span`, inert by default — so the dataset core stays clock-agnostic and names none of this.

`positronic/telemetry.py` owns the mechanism, and it is domain-blind: spans, anchors and sidecar files, with
no notion of an episode, a pass or an inference call. An **anchor** is a long-running span its owner holds
open; the spans an instrumented call site emits parent to the innermost one, which is how a phase span
emitted by one control system lands under the rollout another control system is running (OTel's ambient
context does not survive the scheduler's generator hops). The anchor stack is per-context, so work the loop
dispatches to a thread — an inference call it keeps playing through — records under the anchors that stood when
it was dispatched rather than under whatever is anchored by the time it returns. Ownership follows the phase:
the harness opens, stamps and closes the `episode` span, the eval CLI the `eval.pass` span.

The contract literals split by who writes the bytes they name. Every span name and attribute key in the tables
below is written by eval-domain code through the span helpers, which pass them opaquely and never match on
them, so they are defined once in `positronic/telemetry_keys.py` — a module the mechanism does not import,
which is what lets it stay domain-blind. `positronic/telemetry.py` owns the names of what it writes itself:
the machine-load sample's fields, the sidecar suffixes and the telemetry subdirectory.

## Collecting

```
positronic eval run --eval <eval> --policy <policy> --output_dir <dir> --timing
```

`--timing` needs `--output_dir` (the sidecars are written under it) and an **all-simulated** sweep: everything
under the bound tracer enters the report, so a sweep containing a real embodiment is rejected up front rather
than allowed to pollute it. (A real eval runs its producers as separate processes, each with its own sidecar and
no pass span — run real embodiments in a separate untimed invocation.)

A run that records episodes writes the harness sidecar without being asked: `prepare_output_dir` points
`POSITRONIC_ENV_TELEMETRY_DIR` at `<output_dir>/telemetry/`, so the spans land where `pos3.sync` mirrors them
and upload with the episodes. `--timing` adds the `eval.pass` span and the machine-load stream on top. A run
with no `--output_dir` records nothing, and the span helpers compile to no-ops.

### Levels

Each span has a level, and a process records the spans at or below its active level:

| Level | Records | Rate on a real robot |
|---|---|---|
| `call` | the pass, each episode, reset and policy call, with their spans: `eval.pass`, `episode`, `reset`, `policy.submit`, `policy.prepare`, `policy.encode` of a call, `policy.infer`, `wire.send`, `wire.recv`, the `record.io` that opens and closes the episode's writer | a few spans per policy call |
| `step` | also the spans of each control step: `harness.step`, one `<processor>` span per processor per step, `record.io` of each append at a step, `step.record_wait_ms` of the samples between steps, `driver.command`, the client `env.step` and `materialize` | hundreds of spans per second |

`POSITRONIC_TELEMETRY_LEVEL` names the level, `call` or `step`, and each process that a run spawns inherits it.
Unset, `--timing` records at `step`, because the report reads the step spans. The sidecar that a recording run
writes without being asked records at `call`: the export thread encodes each batch of spans while it holds the
interpreter lock, so the step spans add latency to the control loop. A profiling run sets
`POSITRONIC_TELEMETRY_LEVEL=step`. A name that is not a level stops the run.

A span above the level returns one shared inert span: the helper builds no attributes and queues nothing for
export. Instrumented code gives a span that opens once per control step, or more often, the level `telemetry.Level.STEP`, the
second positional argument of each helper;
every other span takes the default, `CALL`. A span at `call` that opens inside an unrecorded step span parents to
the innermost anchor, for example the episode. The env server of a simulation writes its own file with its own
writer, and it records every span at both levels.

The recording stack is not a default dependency: it ships in the `telemetry` extra
(`uv sync --extra telemetry`, or `pip install positronic[telemetry]`), which the dev dependency group pulls in,
so a checkout has it. `--timing` raises without it and names the extra. A recording run without it warns and
records no spans, because a run delivers episodes and the sidecar is not one. The RoboLab image ships the repo
and resolves its dependencies at run time, so a timed eval inside it names the extra on the invocation itself:
`uv run --extra telemetry positronic eval run … --timing`.

## File layout

Everything lands under `<output_dir>/telemetry/`, one set of files per process:

| File | Written by | Contents |
|---|---|---|
| `harness.spans.jsonl` | the eval process | OTLP/JSON-lines spans: the pass, each episode, and its phase spans |
| `harness.stats.jsonl` | the eval process | one machine-load sample per line |
| `env.spans.jsonl` | a launched env server (e.g. RoboLab) | the server's own step decomposition, in its own file |
| `<process>.spans.jsonl` | each background process of a rig World (a camera, arm or gripper driver) | the spans that process opens, in its own file |

A launched env server writes its own files because it runs in its own interpreter (RoboLab's Isaac venv, where
positronic cannot be imported); it is handed the telemetry dir and run id through the environment its launcher
forwards, and writes the same OTLP/JSON-lines shape with a stdlib-only writer. Nothing rides over the wire.

On a rig, the camera, arm and gripper drivers run as background processes of the World. The World opens
`telemetry.bind_from_env` in each of them at start, under the process's name (`ZedCamera.run`, for example), so
each one records into its own file and none takes the harness's file. The World gives two processes of one class
two names: the second is `ZedCamera.run.2`. The World runs inside `telemetry.one_run_id`, so the harness and
every background process stamp the same run id. The recorder encodes on a thread of the harness process, so its
spans stay in `harness.spans.jsonl`.

### Span schema (`*.spans.jsonl`)

OTLP/JSON, one document per line (`resourceSpans → scopeSpans → spans`). Ids are hex; `startTimeUnixNano` /
`endTimeUnixNano` are epoch-nanosecond strings. Each document's resource block carries the writing process's
identity (`run.id`, `process.name`, `process.pid`, `host.name`). `read_spans` parses it back, tolerating a
truncated final line.

The span names are the contract:

| Span | Parent | Attributes | Measures |
|---|---|---|---|
| `eval.pass` | root | `run.id`, policy name | one eval sweep (its wall is `W_pass`) |
| `episode` | `eval.pass` | `episode.index` + the trial's flat `params` keys; `inference.policy.*`, the scalar fields of the policy's recorded metadata, at start and again before the policy cleanup; at end `episode.steps`, `episode.virtual_s`, `episode.steps_late_ge_<N>ms`, `episode.encodes`, `episode.encodes_gt_<N>ms`, `episode.waypoints.hold_sum_ms`, `episode.waypoints.hold_max_ms` | one rollout's wall |
| `reset` | `episode` | — | scene reset (the harness readying the rig + the producer publishing what it drew) |
| `env.step` | `episode` | — | the client-observed env step (materialisation included) |
| `materialize` | `env.step` | — | client-side observation assembly (shared-memory image alloc + camera copies) |
| `record.io` | `episode` | — | the recorder's serialize + append, at a step and at the open and close of the recording. A sample between two steps records no span: the next `harness.step` carries its time as `step.record_wait_ms` |
| `harness.step` | `episode` | `step.late_ms`, `step.observe_ms`, `step.policy_ms`, `step.emit_ms`, `step.read_ms.<signal>`, `step.convert_ms.<signal>`, `step.frame_wait_ms.<signal>`, `step.frame_driver_ms.<signal>`, `step.gc_wait_ms`, `step.gc_ms`, `step.gc_generation`, `step.record_wait_ms` | one harness step: it reads the observations, calls the policy and emits the commands |
| `<processor>` (the class name in snake case) | `harness.step`, or the processor that calls it | — | one call of one processor in the policy stack |
| `policy.submit` | the processor span that submits the call | `obs.age_ms`; `charged_ms` in a sim that charges server time | one call on a worker thread |
| `policy.infer` | a processor span | `served.<name>` from the answer's `timing`; `wire.uplink_ms`, `wire.downlink_ms`, `wire.network_ms`; `wire.sent_alone` | one real inference round-trip, client-side image compression excluded (a scheduler replay gets none) |
| `wire.send` | `policy.infer` | `wire.bytes`; `wire.new_connection`; on the first send of a connection, `wire.endpoint` (the URL that the client dialed) and `wire.connect_ms` (the time of that dial) | the send of one request |
| `wire.recv` | `policy.infer` | — | the wait for one answer, from the end of its send |
| `driver.command` | root (an arm driver's file) | `command.sync`; `command.age_ms` on a streamed setpoint | one command that an arm driver takes, from its read to its apply |
| `env.step` (server) | root (env file) | — | the env server's own in-step wall |
| env-owned children (e.g. `physics`, `render`) | server `env.step` | env decides | the sim's native phase decomposition |

### Reading arm commands

An arm driver records one `driver.command` span in its own file for each command that it takes. The span starts
when a tick of the driver reads the command. It ends when the first write to the hardware that carries the command
returns: the franka controller call, the kinova current command, the so101 bus write, the yam joint command, or the
CRON velocity command or planned move. The driver takes a `CommandReceipt` from `Moves.take()` right after the read,
passes it with the target, and calls `receipt.applied()` after the write. A second call records nothing.

- `command.sync` is true for a call whose asker waits for the arrival, and false for a streamed setpoint.
- `command.age_ms` is the time from the emit of a streamed setpoint to its read, on the host's monotonic clock.
  A call carries no emit time, so its span has no age.

The span of a call ends when the move starts, not when the arm arrives. A call that waits through an arm error is
timed from its first read. A command that the driver drops, for example a target with no finite joints, records no
span. A command whose first write fails, a target that the driver replaces before its first write, and a move that a
stop or a refusal ends before its first write also record no span.

### Reading harness steps

A `harness.step` span holds the durations of one step in milliseconds, and one collection generation:

- `step.late_ms`: the time from the step's due time to its start. Other loops in the process and a late wake
  from sleep add to it. An answer can start a step before its due time; that step has no value.
- `step.observe_ms`: the time to read all observations. `step.read_ms.<signal>` is one receiver's read.
  `step.convert_ms.<signal>` is the serializer and the copy of a new message, and is absent when no new
  message arrived.
- `step.frame_wait_ms.<signal>` and `step.frame_driver_ms.<signal>`: the two parts of the age of a camera frame
  when the step reads it. Their sum is the age of the frame. `step.frame_wait_ms` is the time from the driver's
  hand-off (the frame's `emitted` time) to the read, on the clock of the policy runtime. The step records it again
  each time it reads the same frame, so it grows until a new frame arrives. `step.frame_driver_ms` is the driver's
  time from capture to hand-off, on the camera's own clock: the frame's `camera.handoff` stamp less its
  `camera.capture` stamp. The ZED and Luxonis drivers stamp both. The OpenCV and V4L2 drivers stamp no capture
  time, so their frames have no driver part. The ZED hand-off stamp reads the host clock. When that clock steps
  back, the stamp stays above the last one until the clock passes it, and the driver part is wrong for that time.
- `step.policy_ms`: the policy call. Its processor spans are the children of the step.
- `step.emit_ms`: the time to emit the commands.
- `step.gc_wait_ms`: the garbage-collection time of the process from the end of the previous step to the start of
  this step. A collection after the due time adds to `step.late_ms`. For the first step of an episode, the window
  starts at the start of the rollout, so it covers the opening of the recording.
- `step.gc_ms`: the garbage-collection time of the process inside the step.
- `step.gc_generation`: the highest generation that a collection in either window reached, from 0 to 2.
- `step.record_wait_ms`: the time that the samples between the previous step and this step spent in their appends.
  The harness samples the observations at its recording rate between two steps, and each sample appends the new
  messages to the recording. These samples record no `record.io` span, because on a rig they give most of the
  spans of the harness. The value is present only when a sample appended a message in that time. The samples after
  the last step of an episode have no next step, so their time is not recorded.

The three collection values are present only when a collection ran in one of the two windows. A collection in any
thread stops the harness thread too, so each value counts all threads. The harness counts collections only while
telemetry records.

A step can find a required observation missing, for example while a camera starts. That step calls no policy,
so its span has only `step.late_ms`, the observe values, the collection values and `step.record_wait_ms`.

The span's own duration less the observe, policy and emit times is the other work in the step. On a real rig the
producers run in other processes, and each writes its own sidecar. Their cost reaches a step only as CPU
contention.

This prints the p10, p50 and p90 of each value over all steps that called the policy in a spans file:

```
uv run python -c "import sys,numpy as np;from positronic import telemetry as t, telemetry_keys as k;S=[s.attrs|{'step_ms':(s.end_ns-s.start_ns)/1e6} for s in t.read_spans(sys.argv[1]) if s.name==k.SPAN_HARNESS_STEP and k.ATTR_STEP_POLICY_MS in s.attrs];[print(f'{k:28s} n={len(v):5d}  p10/p50/p90 ms',*(f'{x:7.3f}' for x in np.percentile(v,[10,50,90]))) for k in sorted({k for a in S for k in a}) for v in [[a[k] for a in S if k in a]]]" <output_dir>/telemetry/harness.spans.jsonl
```

### Reading holds and observation age

The `episode` span carries `episode.waypoints.hold_sum_ms` and `episode.waypoints.hold_max_ms` when a
`TrajectoryPlayer` plays the trajectories of the policy. A hold is the time from the end of the plan to the first
command of the next chunk. During a hold the arm keeps its last target. The wait for the first chunk does not
count, and a hold that the episode end cuts does not count. The hold share of a rollout is
`hold_sum_ms / 1000 / episode.virtual_s`.

The `charged_ms` of a `policy.submit` span is the episode time that the call costs when the trial's
`inference_charge` is `server`: the time that the server reports it computed for the call, or the wall time from
submission to completion for a server that reports none. The span's duration also holds the wait at the server, and
`charged_ms` does not.

The `obs.age_ms` of a `policy.submit` span is the time from the emission of a camera frame to the start of the
call. The frame is the newest frame of each camera that the harness reads, and with several cameras the oldest of
them. A remote deployment puts each of these cameras on the wire, and the server drops the cameras that its model does
not use. So a slow camera that the model does not use still sets the age. A camera that a local codec drops before
inference also sets the age.
The history frames of a `TemporalStack` do not count, because their age is set by the stack offsets. The harness reads the two times on one clock, the clock of the policy
runtime. On a real rig it is the monotonic clock of the host. In a simulation it is the world clock, so the age is
simulated time and does not include the wall time of the simulator. The age uses the capture time in
`Runtime.observed_ns` when the call is submitted. The harness sets it for each step. A processor that sends an older
sample sets the capture time of that sample: `Periodic` on an answer wake-up, and a `TemporalStack` of camera frames
whose newest offset is in the past. An observation with no camera gives no value.

### Reading late steps and slow encodes

The `episode` span counts the late steps and the slow encodes, so a query of the episodes alone can test a
latency target. `episode.steps_late_ge_<N>ms` counts the harness steps that start N ms or more after their due time
(the `step.late_ms` of a `harness.step` span), for N in 1, 2, 5, 10, 20 and 50. A step that reads no complete
observation counts too, and `episode.steps` does not count it. `episode.encodes` counts the `policy.encode` spans of
the policy calls that the rollout submits, and `episode.encodes_gt_<N>ms` counts the spans of them that take more than
N ms, for N in 5, 10, 15, 30 and 60. Each episode span carries each count, 0 included. The counts record at the `call`
level, so a run gets them without the step spans. They count only the encode spans that the active level records.

### Reading the network time of a call

A server answers each call with a `timing` block. Its durations are in milliseconds on the server's clock, and
`policy.infer` copies each one as `served.<name>`. `served_ms` runs from the arrival of the whole request to the
end of the call. `encode_ms` is the encode of the answer's images. Two instants are in Unix nanoseconds on the
server's clock: `received_ns`, when the server read the whole request, and `sent_ns`, when the answer starts to
serialise. The encode of the answer's images comes before `sent_ns`.

With the two instants, the client puts three values on `policy.infer`:

- `wire.uplink_ms`: from the start of the send to `received_ns`. It includes the server's read of the request.
- `wire.downlink_ms`: from `sent_ns` to the receipt of the answer. It includes the serialise and the send of the
  answer.
- `wire.network_ms`: the sum of the two.

Each one-way value subtracts a client time from a server time, so it holds the offset between the two clocks. A
clock that runs 20 ms ahead on the server adds 20 ms to the uplink and removes 20 ms from the downlink. Neither the
sum nor `served.sent_ns - served.received_ns`, the server's time from its read to its send, holds an offset. A
server that sends no instants gives no `wire.*` values, and its `served.*` values stay as they are.

A server reads a request only after it answers the request before it. So a call that the client sends while another
call is in flight waits on the server before `received_ns`, and its one-way values would count that wait as network
time. Such a call gives no one-way values.

`wire.sent_alone` is `true` on a call that the client sent with no earlier call in flight, and `false` on a call sent
behind another. A call that never reached its send carries no mark.

### Stats schema (`*.stats.jsonl`)

One JSON sample per line, sampled free-running at 1 Hz (default) on wall time — no span context, no episode
boundary, so no sample is lost at a phase edge. A sample is a flat object of host and process readings plus a
list holding one object per GPU and a list holding one object per process. Its keys are the `STAT_*`, `GPU_*`
and `PROC_*` constants in `positronic/telemetry.py`
— read them there; the sampler writes them and the reduce imports them.

The process readings cover this eval's whole process tree (harness + env server + Isaac children): the tree sums
(`STAT_CPU_PROC_PCT`, `STAT_RSS_PROC_B`), and a list with one `PROC_*` entry (name, pid, CPU percent, RSS) per
process. The caller names a pid through `process_names`; a pid it does not name takes the OS process name. A
process's CPU percent in its first sample covers no interval, so it is not a reading. On a Jetson, the one GPU
entry takes its load from sysfs (an Orin's path by default, `tegra_gpu_load_path` elsewhere) and carries no memory or
power reading. The GPU list is empty on a box with no NVML and no Jetson GPU, and shorter than the
recorded device count when a device refuses a query mid-run (MIG, a transiently lost GPU) — so that count tells a
reader whether a sample saw the whole box. Per-process GPU memory is null where it cannot be attributed (a PID
namespace without `--pid=host`, or a driver that will not report it), and per-process GPU utilisation is always
null: it is not reliably attributable under MPS / co-location.

## Reporting

```
positronic eval timing-report <run_dir> [--gpu_policy_log <nvidia-smi dmon log>]
```

Reduces the sidecars under `<run_dir>/telemetry/` into a pass report and writes `timing_summary.json` beside
the input (`<run_dir>` may be an `s3://` URI). It reports:

- the **wall split** — each phase's share of the report's wall window (reset / env.step / policy wait / record
  IO / overhead / between-episodes), summing to 1. Record IO is the `record.io` spans plus the `step.record_wait_ms`
  of the steps; the policy-wait share also carries the sizing figure derived
  from it, how many such evals one policy server could keep fed;
- the **env-step split** — physics / render / server-other (the env server's own decomposition), plus the wire
  and materialisation shares of the client step; absent for a native sim, which reports no server decomposition;
- **inference latency** — call count and p50 / p95;
- the **latency table** — the count, the episode count and the p50 / p90 / p99 of each part of a step and of a
  policy call, over the spans that start inside an episode of the same run. Only a value named `_ms` makes a row, so
  `step.gc_generation` makes none. `wire.uplink_ms` and `wire.downlink_ms` hold the clock offset, so they make no
  row either; `wire.network_ms` does. A row named with `+` and `-` is that
  sum of spans. `wire.send + wire.recv - served.served_ms - served.encode_ms` holds the transfer both ways and no
  clock offset. A call with no `served.encode_ms` subtracts 0 for it. The row still holds the serialise of the
  answer on the server. It counts only the calls with `wire.sent_alone` `true`, so a spans file from a client that
  writes no such mark gives no such row. The `-` rows of a call leave out a call that sent more than once: it holds the
  reconnect between its attempts, and its `served.*` values are of the last attempt only. A spans file with no `harness.step` spans gives the call rows only;
- the **real-time factor** — recorded virtual duration over span wall;
- the sim box's **GPU load** — mean utilisation, peak VRAM, and this eval's peak process VRAM, from the stats
  stream. The two peaks are box-wide totals, so they need samples that saw every device and read unavailable
  when none did. Utilisation averages every per-device reading taken, so it is biased towards the devices that
  answered; it carries `devices_seen` / `box_devices` with it, and the report prints `util 60% over 1 of 2
  GPUs` whenever the mean covers part of a box. A box whose devices all refuse their queries still reports as
  a GPU box, with the metrics unavailable — only a box holding no device at all carries no GPU line.

Shares print as percentages; `timing_summary.json` keeps them as fractions.

Every share, and the real-time factor, is a fraction of one wall window, and `window` names which: `W_pass`,
the `eval.pass` span, whenever that span closed. A run killed or preempted mid-pass never writes it, so the
reduce falls back to `W_episodes` — the wall from its first complete episode to its last, one window per run
where a directory holds several. That window excludes whatever ran either side of those episodes, so a share
of it is not a share of a pass; the report says which one it used in its own body as well as on the console.
A directory holding neither a closed pass span nor an episode has nothing to reduce, and the reduce refuses.

`--gpu_policy_log` folds in the policy endpoint (a different box) from an `nvidia-smi dmon -s um` log; it reads
the `sm` / `fb` column positions from the log header and fails loudly if the `fb` (framebuffer) column is
missing, rather than dropping every row and reporting no peak VRAM for the policy box.
