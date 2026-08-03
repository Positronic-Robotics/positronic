# Eval telemetry

Opt-in wall-clock telemetry for `positronic eval run`. A sim eval runs on a virtual clock, so the virtual
time a rollout advances says nothing about the real compute it cost. This telemetry captures that operational
signal — the wall-clock split of each phase (reset, env step, inference, record IO), the machine's CPU / memory
/ GPU load, and the inference-latency distribution — so a sizing or performance pass can read what a rollout
actually spent.

It is **observability, not data** (see `architecture.md`, "Telemetry is observability, not data"). The dataset
records the robot's world under the run's (possibly virtual) clock; telemetry describes the machinery around it
in wall time. So it lives in **sidecar files next to the dataset, never inside it** — the dataset core imports
no telemetry and stays clock-agnostic.

`positronic/telemetry.py` owns the mechanism, and it is domain-blind: spans, anchors and sidecar files, with
no notion of an episode, a pass or an inference call. An **anchor** is a long-running span its owner holds
open; the spans an instrumented call site emits parent to the innermost one, which is how a phase span
emitted by one control system lands under the rollout another control system is running (OTel's ambient
context does not survive the scheduler's generator hops). Ownership follows the phase: the harness opens,
stamps and closes the `episode` span, the eval CLI the `eval.pass` span. The vocabulary both sides and the
reduce agree on — every span name and attribute key in the tables below — is defined once in
`positronic/telemetry_keys.py`.

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
| `episode` | `eval.pass` | `episode.index` + flat trial-context keys; at end `episode.steps`, `episode.virtual_s`; `episode.aborted` on an abort | one rollout's wall |
| `reset` | `episode` | — | scene reset (harness arming + producer frame-0 materialisation) |
| `env.step` | `episode` | — | the client-observed env step (materialisation included) |
| `materialize` | `env.step` | — | client-side observation assembly (shared-memory image alloc + camera copies) |
| `record.io` | `episode` | — | the recorder's serialize + append |
| `policy.infer` | `episode` | — | one real inference round-trip (a scheduler replay gets none) |
| `env.step` (server) | root (env file) | — | the env server's own in-step wall |
| env-owned children (e.g. `physics`, `render`) | server `env.step` | env decides | the sim's native phase decomposition |

### Stats schema (`*.stats.jsonl`)

One JSON sample per line, sampled free-running at 1 Hz (default) on wall time — no span context, no episode
boundary, so no sample is lost at a phase edge:

```json
{"t_ns": 0, "cpu_sys_pct": 0.0, "iowait_pct": 0.0, "mem_sys_used_b": 0,
 "cpu_proc_pct": 0.0, "rss_proc_b": 0,
 "gpus": [{"i": 0, "util_pct": 0.0, "mem_used_b": 0, "mem_total_b": 0, "power_w": 0.0,
           "proc_mem_b": 0, "proc_util_pct": null}]}
```

`cpu_proc_pct` / `rss_proc_b` / `proc_mem_b` are this eval's whole process tree (harness + env server + Isaac
children). `gpus` is empty on a box with no NVML; `proc_mem_b` is null where per-process GPU memory is
unavailable (a PID namespace without `--pid=host`); `proc_util_pct` is left null because per-process GPU
utilisation is not reliably attributable under MPS / co-location.

## Reporting

```
positronic eval timing-report <run_dir> [--gpu_policy_log <nvidia-smi dmon log>]
```

Reduces the sidecars under `<run_dir>/telemetry/` into a pass report and writes `timing_summary.json` beside
the input (`<run_dir>` may be an `s3://` URI). It reports:

- the **wall split** — each phase's share of `W_pass` (reset / env.step / policy wait / record IO / overhead /
  between-episodes), summing to 1; the policy-wait share also carries the sizing figure derived from it, how
  many such evals one policy server could keep fed;
- the **env-step split** — physics / render / server-other (the env server's own decomposition), plus the wire
  and materialisation shares of the client step; absent for a native sim, which reports no server decomposition;
- **inference latency** — call count and p50 / p95;
- the **real-time factor** — recorded virtual duration over span wall;
- the sim box's **GPU load** — mean utilisation, peak VRAM, and this eval's peak process VRAM, from the stats
  stream.

Shares print as percentages; `timing_summary.json` keeps them as fractions.

`--gpu_policy_log` folds in the policy endpoint (a different box) from an `nvidia-smi dmon -s um` log; it reads
the `sm` / `fb` column positions from the log header and fails loudly if the `fb` (framebuffer) column is
missing rather than under-reporting peak VRAM as 0.

## Future work

Not implemented here — recorded so the design is legible:

- **Perfetto / Chrome-trace view.** The spans (as `X` duration events) and the stats (as `C` counter events)
  open on one timeline in `ui.perfetto.dev` for interactive inspection. The report reads the raw JSONL directly
  and needs no converter, so this is a convenience view, deferred.
- **H100 / server-side serving telemetry.** The policy-serving images could emit their own sidecar: startup
  spans (entrypoint → weights loaded → listening → first request), CUDA-event stage timing, per-request peak
  VRAM, and Cristian's-algorithm min-RTT clock probes to reconcile the policy box's wall clock with the eval
  box's. A follow-up; this is eval-box only.
- **Closed-weight-model telemetry.** A client-facing recipe for models served behind a vendor API (no
  server-side hooks) — a separate proposal.
