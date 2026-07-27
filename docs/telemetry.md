# Eval telemetry

Opt-in wall-clock telemetry for `positronic eval run`. A sim eval runs on a virtual clock, so the virtual
time a rollout advances says nothing about the real compute it cost. This telemetry captures that operational
signal — the wall-clock split of each phase (reset, env step, inference, record IO), the machine's CPU / memory
/ GPU load, and the inference-latency distribution — so a sizing or performance pass can read what a rollout
actually spent.

It is **observability, not data** (see `architecture.md`, "Telemetry is observability, not data"). The dataset
records the robot's world under the run's (possibly virtual) clock; telemetry describes the machinery around it
in wall time. So it lives in **sidecar files next to the dataset, never inside it** — the dataset core imports
no telemetry and stays clock-agnostic. It is owned by `positronic/telemetry.py`.

## Collecting

```
positronic eval run --eval <eval> --policy <policy> --output_dir <dir> --timing
```

`--timing` needs `--output_dir` (the sidecars are written under it) and applies to **simulated** evals: a real
embodiment runs the recorder and producers as separate processes with no shared tracer, so nothing there is
timed. A mixed sweep still times its simulated evals; a sweep with nothing to time is rejected up front.

A normal eval (no `--timing`) pays nothing: the span helpers compile to no-ops and no sidecar is written.

## File layout

Everything lands under `<output_dir>/telemetry/`, one set of files per process:

| File | Written by | Contents |
|---|---|---|
| `harness.meta.json` | the eval process | process identity (schema, run id, host, pid, start times, python, platform) |
| `harness.spans.jsonl` | the eval process | OTLP/JSON-lines spans: the pass, each episode, and its per-tick phases |
| `harness.stats.jsonl` | the eval process | one machine-load sample per line |
| `env.meta.json` / `env.spans.jsonl` | a launched env server (e.g. RoboLab) | the server's own step decomposition, in its own file |

A launched env server writes its own files because it runs in its own interpreter (RoboLab's Isaac venv, where
positronic cannot be imported); it is handed the telemetry dir and run id through the environment its launcher
forwards, and writes the same OTLP/JSON-lines shape with a stdlib-only writer. Nothing rides over the wire.

### Span schema (`*.spans.jsonl`)

OTLP/JSON, one document per line (`resourceSpans → scopeSpans → spans`). Ids are hex; `startTimeUnixNano` /
`endTimeUnixNano` are epoch-nanosecond strings. `read_spans` parses it back, tolerating a truncated final line.

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
  between-episodes), summing to 1;
- the **env-step split** — physics / render / server-other (the env server's own decomposition), plus the wire
  and materialisation shares of the client step; absent for a native sim, which reports no server decomposition;
- **inference latency** — call count and p50 / p95;
- the **real-time factor** — recorded virtual duration over span wall;
- the sim box's **GPU load** — mean utilisation, peak VRAM, and this eval's peak process VRAM, from the stats
  stream.

`--gpu_policy_log` folds in the policy endpoint (a different box) from an `nvidia-smi dmon -s um` log; it reads
the `sm` / `fb` column positions from the log header and fails loudly if the `fb` (framebuffer) column is
missing rather than under-reporting peak VRAM as 0.

## Future work

Not implemented here — recorded so the design is legible:

- **Perfetto / Chrome-trace view.** The spans (as `X` duration events) and the stats (as `C` counter events)
  open on one timeline in `ui.perfetto.dev` for interactive inspection; separate process files merge via each
  meta file's start offset. The report reads the raw JSONL directly and needs no converter, so this is a
  convenience view, deferred.
- **H100 / server-side serving telemetry.** The policy-serving images could emit their own sidecar: startup
  spans (entrypoint → weights loaded → listening → first request), CUDA-event stage timing, per-request peak
  VRAM, and Cristian's-algorithm min-RTT clock probes to reconcile the policy box's wall clock with the eval
  box's. A follow-up; this is eval-box only.
- **Closed-weight-model telemetry.** A client-facing recipe for models served behind a vendor API (no
  server-side hooks) — a separate proposal.
