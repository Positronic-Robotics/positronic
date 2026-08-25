# Eval telemetry

Opt-in wall-clock telemetry for `positronic eval run`. A sim eval runs on a virtual clock, so the virtual
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
than allowed to pollute it. (A real eval runs its recorder and producers as separate processes with no shared
tracer, so there is nothing to time there anyway — run real embodiments in a separate untimed invocation.)

A normal eval (no `--timing`) pays nothing: the span helpers compile to no-ops and no sidecar is written. That
is also why the recording stack is not a default dependency — `--timing` needs the `telemetry` extra
(`uv sync --extra telemetry`, or `pip install positronic[telemetry]`), and says so if it is missing. An install
without it carries only the OTel API the no-op helpers sit on. The RoboLab image ships the repo and resolves
its dependencies at run time, so a timed eval inside it names the extra on the invocation itself:
`uv run --extra telemetry positronic eval run … --timing`.

## File layout

Everything lands under `<output_dir>/telemetry/`, one set of files per process:

| File | Written by | Contents |
|---|---|---|
| `harness.spans.jsonl` | the eval process | OTLP/JSON-lines spans: the pass, each episode, and its phase spans |
| `harness.stats.jsonl` | the eval process | one machine-load sample per line |
| `env.spans.jsonl` | a launched env server (e.g. RoboLab) | the server's own step decomposition, in its own file |

A launched env server writes its own files because it runs in its own interpreter (RoboLab's Isaac venv, where
positronic cannot be imported); it is handed the telemetry dir and run id through the environment its launcher
forwards, and writes the same OTLP/JSON-lines shape with a stdlib-only writer. Nothing rides over the wire.

### Span schema (`*.spans.jsonl`)

OTLP/JSON, one document per line (`resourceSpans → scopeSpans → spans`). Ids are hex; `startTimeUnixNano` /
`endTimeUnixNano` are epoch-nanosecond strings. Each document's resource block carries the writing process's
identity (`run.id`, `process.name`, `process.pid`, `host.name`). `read_spans` parses it back, tolerating a
truncated final line.

The span names are the contract:

| Span | Parent | Attributes | Measures |
|---|---|---|---|
| `eval.pass` | root | `run.id`, policy name | one eval sweep (its wall is `W_pass`) |
| `episode` | `eval.pass` | `episode.index` + the trial's flat `params` keys; at end `episode.steps`, `episode.virtual_s` | one rollout's wall |
| `reset` | `episode` | — | scene reset (the harness readying the rig + the producer publishing what it drew) |
| `env.step` | `episode` | — | the client-observed env step (materialisation included) |
| `materialize` | `env.step` | — | client-side observation assembly (shared-memory image alloc + camera copies) |
| `record.io` | `episode` | — | the recorder's serialize + append |
| `policy.infer` | `episode` | — | one real inference round-trip, client-side image compression excluded (a scheduler replay gets none) |
| `env.step` (server) | root (env file) | — | the env server's own in-step wall |
| env-owned children (e.g. `physics`, `render`) | server `env.step` | env decides | the sim's native phase decomposition |

### Stats schema (`*.stats.jsonl`)

One JSON sample per line, sampled free-running at 1 Hz (default) on wall time — no span context, no episode
boundary, so no sample is lost at a phase edge. A sample is a flat object of host and process readings plus a
list holding one object per GPU. Its keys are the `STAT_*` and `GPU_*` constants in `positronic/telemetry.py`
— read them there; the sampler writes them and the reduce imports them.

The process readings cover this eval's whole process tree (harness + env server + Isaac children). The GPU
list is empty on a box with no NVML, and shorter than the recorded device count when a device refuses a query
mid-run (MIG, a transiently lost GPU) — so that count is what tells a reader whether a sample saw the whole
box. Per-process GPU memory is null where it cannot be attributed (a PID namespace without `--pid=host`, or a
driver that will not report it), and per-process GPU utilisation is always null: it is not reliably
attributable under MPS / co-location.

## Reporting

```
positronic eval timing-report <run_dir> [--gpu_policy_log <nvidia-smi dmon log>]
```

Reduces the sidecars under `<run_dir>/telemetry/` into a pass report and writes `timing_summary.json` beside
the input (`<run_dir>` may be an `s3://` URI). It reports:

- the **wall split** — each phase's share of the report's wall window (reset / env.step / policy wait / record
  IO / overhead / between-episodes), summing to 1; the policy-wait share also carries the sizing figure derived
  from it, how many such evals one policy server could keep fed;
- the **env-step split** — physics / render / server-other (the env server's own decomposition), plus the wire
  and materialisation shares of the client step; absent for a native sim, which reports no server decomposition;
- **inference latency** — call count and p50 / p95;
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
