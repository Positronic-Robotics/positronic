# PhAIL Eval — Design Doc

_Living design doc for the PhAIL eval toolkit. Part 1 is the durable **vision** (the abstractions and principles). Part 2 is the **chain** — a sequence of right-sized steps (each a reviewable PR) that build the vision one move at a time. Part 3 is **reference** (research context, code pointers, non-goals, open questions). Part 4 is **design rationale** (the non-obvious whys). Written so a fresh reader picks up cold — it describes the present plan, not its history._

_Worktree: `worktree-phail-eval` (in positronic repo at `.claude/worktrees/phail-eval/`)._

---

# Part 1 — Vision

## 1.1 What we're building

A **VLA-eval toolkit for researchers**, shipped as part of Positronic. The product surface is a single CLI: `positronic eval run <eval> …` produces a **Positronic Dataset** (existing format, `positronic.dataset.local_dataset.LocalDatasetWriter`); `positronic eval analyze …` scores those datasets later. The toolkit runs entirely locally with no account.

PhAIL the leaderboard is a *consequence* of this toolkit running on real-robot evals — not the product itself. Leaderboard submission is opt-in and post-MVP.

### Why this exists
- ICRA 2026 survey (349 manipulation papers): the VLA field has an evaluation crisis — every paper picks its own task suite, hardware, trial counts. Cross-lab comparison is effectively impossible.
- vla-eval (Allen AI) is solving *sim* eval with a Docker-per-benchmark + WebSocket architecture — relevant prior art on the sim side.
- Positronic already has the methodology primitives the field lacks: codec (training/inference parity), trajectory-as-command with absolute timestamps, scheduler composition, latency-honest sim (the `inference_latency` knob on a world-owned virtual clock), the pos3 dataset format.
- Researchers spend 50%+ of their time on eval infrastructure. A drop-in toolkit that produces credible, shareable, comparable artifacts is the wedge.

### Where it fits in Positronic's surface
- `positronic eval` (this work): the researcher-facing CLI — a curated catalog + a clean submission shape on top of the existing harness.
- `positronic server` (existing): FastAPI dataset viewer at `localhost:8400`. PhAIL artifacts render here; no new viewer needed.
- `positronic inference` (existing `positronic-inference`): the internal harness CLI. It and `eval`/`server` converge under one `positronic` parent command.

## 1.2 Personas

- **Academic researcher** publishing a VLA paper — needs credible baseline comparisons + reproducible numbers reviewers can audit.
- **Foundation-model lab** (Physical Intelligence, NVIDIA, smaller VLA labs) — needs fast feedback during training + occasional credible real-robot validation.

Both care more about real than sim, both have a checkpoint and want comparable shareable numbers with minimum pain, both prefer CLI + code over GUIs. **Implication:** agentic-first. The CLI is the contract; artifacts are machine-readable; a human dashboard is optional and sits on the same data.

## 1.3 The model

The whole design is four roles with sharp boundaries: **embodiment**, **task**, **submission**, and the shared **Harness + driver + World**. An *eval* is `embodiment + task`; a *submission* is the researcher's policy; the Harness/driver/CLI/World are shared machinery.

### Catalog: name vs identity

**Name grammar: `<universe>.<primary-axis>.<task>`.** `universe` (`sim`/`real`) is whether `pimm.World` runs on **virtual time** (`World(virtual_time=True)` — the scheduler advances a virtual clock) or **wall time** (real hardware). The middle segment is whatever most defines the eval in that universe — vendor/suite for sim (`sim.positronic.stack_cubes`, `sim.libero.<name>`), embodiment for real (`real.droid.pick_place.wooden_spoons`).

The **name** is the minimal unambiguous handle; the comparability **identity** is the full tuple — universe × vendor × task × embodiment × success-criterion × trial-protocol — and is always recorded in the artifact. Embodiment is elided from the name while unambiguous and promoted only when a second appears (`stack_cubes` implies the Franka today). `cloud` is **not** a universe — it's where a rollout executes (local GPU vs hosted queue), orthogonal to sim/real.

The catalog is a **flat dict keyed by dotted name** (`{'sim.positronic.stack_cubes': cfg, …}`); `eval list <prefix>` is a thin traversal. In-package for now; entry-point plugins later.

### Embodiment (the eval API)

An embodiment is **only what the policy drives** — the robot, never the task. It is produced by a **factory function** (the eval's entry in the catalog), which builds the device control systems once and returns the signal-dict contract the rest of the system consumes:

- **`observations`** — named signal sources (arm state, gripper, camera frames), each paired with a **serializer** that turns the device value into the canonical observation entries fed to the policy. The serializer owns the key names: a robot-state source serializes to `robot_state.q/.dq/.ee_pose/.status`; a camera to its image array; the gripper to `grip`.
- **`commands`** — for each action key the policy emits, the device receiver it routes to and the **home value** for that channel (`robot_command → (arm, Reset())`, `target_grip → (gripper, 0.0)`). Commands are timestamped trajectories (`list[(timestamp_ns, value)]`).
- **`privileged`** — ground-truth signal sources to record but never feed the policy (sim's full state; a real scale).
- **`descriptor`** — a short **string** identifying the embodiment (`mujoco.franka`, `real.droid`), handed to the policy on every call.

**The Harness is name-free.** It hardcodes no canonical key. It applies the embodiment's observation serializers to assemble the obs dict, and demuxes the policy's action chunks onto the command channels **uniformly** — every channel treated identically, with no arm-vs-gripper knowledge. Adding or swapping a channel (a second arm, a different gripper) is a change in the factory, not the Harness.

**Backed by 1 or N device CSs — not fused.** What satisfies the contract is an implementation detail:
- **Real** composes separate device CSs (Franka arm + Robotiq gripper + cameras), so a device swaps independently — change the arm, keep the gripper and cameras.
- **An external sim (LIBERO)** is naturally one CS — its `env.step()` does sim + robot + gripper + render atomically.
- **Native MuJoCo** stays composed device CSs over a shared `sim`.

The contract lives at the **signal boundary**, not at "one CS vs many." **Fusing kills swappability.** Multiplicity (more than one of a role) is the only thing that needs channel-keyed dicts, deferred until the first real bimanual embodiment.

**One factory, three consumers.** The same embodiment factory feeds the eval runner, teleop data collection, and replay — device construction lives in one place instead of being copy-pasted across them. `wire()` connects the command producer (the Harness, or the teleop/replay controller) to the embodiment's signals generically.

**Gym-style envs fold in via a general adapter.** Many sim benchmarks (LIBERO, RoboCasa, ManiSkill, …) expose a gym `reset()/step(action)` API. A reusable adapter turns any such Env into an embodiment satisfying this contract — the integrator supplies the mappings: canonical command → env action, env obs → canonical observations, env state → privileged signals, seed → init-state. LIBERO is the *first instance* of that utility, not a one-off.

The embodiment also:
- **Holds the last _commanded_ setpoint when the trajectory queue is dry.** No-op ≡ re-assert the last *commanded* (not measured) target. An empty command trajectory tells a channel's driver to hold: native MuJoCo via sticky `ctrl`, a real arm via its low-level controller, a wrapped external sim by re-issuing the last commanded target — which forces absolute-mode OSC.
- **Owns scene reset and the seed sequence** (see "reset, seeding & reproducibility"). It does **not** own the clock — the World does.
- **Exposes its privileged / ground-truth state** for the task to record — sim full state, a real scale reading. The policy never sees it.

The **descriptor is a bare string** because nothing consumes structured fields yet — it is purely the per-call routing hook for a future multi-embodiment policy. The control frame is deliberately **not** in it: a robot accepts commands in different frames, so the frame travels with the command/codec, not the embodiment identity.

### Task (the scenario)

The task is the scenario layered on an embodiment — the catalog's `<task>` segment. It owns:

- **The instruction (prompt)** — the language goal sent to the policy. The *one* task element that is policy-facing; it rides the per-episode `RUN` context and may vary per episode (pick-towels vs pick-spoons variants).
- **The privileged signals to record** — which ground-truth to capture from the embodiment's privileged state. Recorded, **never fed to the policy**.
- **The success criteria** — a *calculation over recorded data*, applied in analysis.
- **Reset & reproducibility** — the task accepts a `seed` and resets the starting state from it: **fully reproducible in sim** (same seed → same scene), **best-effort on real** (physical/human reset, with everything recorded so reruns are as close as possible).
- **Optionally, a live termination predicate** (see lifecycle).

Registering a new eval is, in essence, *exposing the embodiment's privileged state and declaring the task that records it.*

### Outcome & scoring (analysis-time)

Success is **not** computed at rollout time and is **not** part of the embodiment. The rollout records the **raw privileged state as-is** — for a sim task the *entire simulator state* (`save_state`, every physics spec), not pre-reduced observer scores — so *any* criterion is computable later; **success-rate and the TTS distribution are functions over those recordings, computed in analysis** — re-runnable and tunable without re-running rollouts (one expensive rollout, many cheap criteria experiments). So stacking-success is a pure analysis function over recorded poses; LIBERO's goal predicate is re-derivable from recorded env state; real scales record raw weight.

The only thing computed *live* is an **optional termination predicate**, and only when an eval auto-stops (real scales ending an episode). TTS is a **distribution, not a scalar** — multiple success events per episode (pick-and-place) give it for free, because scoring reads a time series, not a terminal flag.

### Submission

A submission is a single **Policy** = the researcher's server code + a local wrapper. A `Policy` is a **factory for per-episode `Session`s** (`new_session(context) → Session`; `session(obs) → list[dict] | None`, where `None` = "keep executing the current chunk"). It is **embodiment-agnostic to the framework**: one `Policy` serves whatever embodiments it supports by creating independent `Session`s and **routing on the per-call descriptor string internally** (a multi-embodiment policy handles several robots; a single-robot policy handles one) — which is exactly why the policy is stateless and the descriptor is passed every call.

The wrapper is a **`PolicyWrapper` pipeline** composed with `|` (left = outermost) and applied via `pipeline.wrap(RemotePolicy(...))`, resolved via cfn `@module.symbol` — e.g. `ErrorRecovery | ChunkedSchedule | ObservationCodec | AbsolutePositionAction`. **Scheduling and error recovery are wrappers** (`ChunkedSchedule`/`ErrorRecovery`) → submission-owned; the **codec** (`ObservationCodec` + an action codec) sits inside the pipeline, *below* the Harness boundary, so the Harness always sees **canonical** observations in and **canonical** commands out. The codec defines *both* sides — inference (`encode`/`decode`) and training (`training_encoder`, which maps recorded signals to features via the dataset `transforms` layer) — so train↔infer parity is the codec's job. There is **no new wrapper DSL**; researchers ship Python and override via cfn.

```
positronic eval run sim.positronic.stack_cubes \
  --policy=@positronic.cfg.policy.remote --policy.host=my-model.com --policy.port=8443 \
  --policy.headers='{"Authorization":"Bearer xxx"}' \
  --wrap=@my_research.wrappers.research_pipeline \
  --trial_count=50 --output_dir=./runs/exp42
```

### Episode lifecycle (Harness + driver)

The Harness wears two hats, both independent of the embodiment's channel count:
- **A name-free signal router** — assemble the obs dict by applying the embodiment's observation serializers, call `session(obs)` (which returns `list[dict] | None` — `None` means "keep executing the current chunk", set by the scheduling wrapper), and demux the returned actions onto the embodiment's command channels **uniformly** (every channel emits its due waypoints; an empty trajectory holds that channel). Scheduling, error recovery, and relative→absolute time anchoring live in the wrapper/session layer. The Harness's only time effect is `inference_latency` — a **sim-only** simulation of model latency (sleep the virtual clock by the inference cost, then post-shift the chunk); in real there is nothing to simulate. There is **no per-channel and no per-command special-casing** in the router — no "is this a recovery emit" check.
- **The lifecycle & recording owner** — `RUN`/`STOP`/`FINISH`/`HOME` directives drive the dataset writer (start/suspend/finalize/abort), scene reset, and homing.

**Stateless policy.** The Harness puts the embodiment **descriptor** string and the **instruction** into the per-call input dict on *every* session call, even though they're constant within an episode. Nothing relies on the policy having stashed state at session creation. (It may still hold internal compute state — chunk buffers, ensembling — but the *contract* is: inputs are always complete.)

**Execution knobs ride the `RUN` directive.** The driver assembles `RUN{instruction + context, timeout, inference_latency}`; the Harness *enforces* the timeout (self-terminating at `start + timeout`, since the World owns the clock) but holds no per-episode execution config of its own. `timeout` and `inference_latency` are CLI/Harness knobs, **not eval identity** — recorded in meta so any non-default value stays honest.

**The driver just sequences trials:** loop `trial_count` (a CLI input), emit `RUN`, advance when the episode ends.

**Termination is a typed signal.** An episode ends with a **reason** — a typed enum (`success`, `fail`, `stalled`, `safety`, `system`, `out_of_time`), like the codebase's `DirectiveType`/`RobotStatus`, recorded as its string value — not a free-form string. Sources map to reasons: Harness timeout → `out_of_time`; human UI → the chosen button; task auto-detect → `success`/`fail`; safety system → `safety`. The Harness can `match` on it exhaustively to pick a disposition + home path — clean save (`success`/`fail`/`out_of_time`), flag (`safety`), or drop (`system`/cancel). The reason is the episode's single *terminal* outcome, distinct from the possibly-many success *events* on the recorded signal stream.

**Human-in-the-loop is the two-phase case.** For real evals the operator presses stop (or the timeout fires) → `STOP` suspends without finalizing → the operator reviews and annotates (notes, success/total items, markups) → `FINISH(reason + annotations)` finalizes, or `Cancel → HOME` aborts. Automatic evals collapse this into one step with a machine-supplied reason. Same mechanism; only who supplies the reason + annotations differs.

### Time & reset (the World owns the clock)

**The World owns time; no control system is a clock.** The scheduler (`World.interleave`) holds a priority queue of control systems keyed by their next-wake time and advances a single **virtual clock to the exact next scheduled event** — so time paces to whatever the systems actually need, with no fixed discretization a separate "clock" system would impose. `World(virtual_time=True)` runs this virtual clock (sim); `World(virtual_time=False)` uses wall time (real hardware).

`MujocoSim` is an ordinary control system: when scheduled it steps physics forward by the elapsed virtual time and yields `Sleep(timestep)`. It reports no time of its own.

Three execution profiles fall out of one mechanism:
- **eval-sim** — virtual clock, **free-run**: the scheduler advances as fast as compute allows. Fastest rollouts; model latency is simulated explicitly via `inference_latency` (which sleeps the virtual clock).
- **VR data-collection** — virtual clock, **paced to wall**: a human teleoperates live, so sim-time is pinned to wall-time.
- **real** — **wall** clock: real I/O paces naturally.

Pacing (free-run vs paced-to-wall) is a runner/World mode, not bespoke per-entrypoint code.

**Reset is a pure state discontinuity, never a time discontinuity.** Because world-time is independent of any engine's internal time, `reset()` just re-randomizes the scene (`mj_resetData` + loaders); the World's virtual clock keeps marching monotonically. Episodes are split by the lifecycle (`new_episode`), not by a time reset. **One persistent World** — never recreated, which matters most for real hardware (you can't tear down and rebuild a robot between trials).

### Reset, seeding & reproducibility

The eval owns the reset; the **seed value is a CLI input** (`base`, random by default). Given the seed, **sim is fully reproducible** (same seed → same scene); **real is best-effort** (record everything). Reproducibility means "rerun close enough, everything recorded," not bit-equality. The embodiment **self-seeds deterministically**: constructed with `base`, it derives `base + n` on its `n`-th reset, re-randomizes, and **self-reports the seed used** into episode meta. Loaders stay `seed=None` so they consume the `np_seed`-wrapped global stream — one outer seed makes the whole scene deterministic. The seed **cannot ride the `RUN` context**, because re-randomization for trial *i* happens at trial *i-1*'s `FINISH`, before trial *i*'s context exists — so the **scene-owning embodiment** owning the seed sequence is the clean path. A LIBERO wrapper maps `seed % num_init_states` onto its enumerated saved init-states.

### Serialization (type-owned, two boundaries)

Serialization is keyed by **domain type**, not declared per signal and not dispatched on raw `np.array` (a 7-vector is a pose-quat or 7 joints — ambiguous). Domain types (`Transform3D`, `JointPositions`, `State`) disambiguate. Each type owns its (de)serialization at **two boundaries**, and both just *ask the type*:

- **Dataset / `Signal`** — type ↔ on-disk (parquet / video / object blob).
- **Offboard / network** — type ↔ wire (the `InferenceServer`/`Client` WebSocket).

This single principle removes the `Kind` enum, the `uint8-HxWx3` storage heuristic, per-signal serializer declaration, **and** `TrajectoryOverrideSerializer` (a chunk is simply a type the `Signal` knows how to store). The physical backends (parquet/video/object writers) remain as implementation the *type* selects. Domain types are **array-backed for pimm transport** — the `MujocoFrankaState(State, NumpySMAdapter)` pattern, where `np.array` is the IPC substrate and the type wraps it. `Signal[T]` is already generic; storage just needs to carry the domain type through. This is the train→inference parity story made structural: the *same* typed value crosses both boundaries.

**Recording == canonical policy I/O.** Because the codec sits inside the submission, the Harness boundary is always canonical, so the same declared serializer feeds the policy *and* records the signal — submission-independent and comparable across models. The observation serializer that builds the obs dict and the serializer that records the signal currently differ (the obs side keeps the raw `status` enum that `ErrorRecovery` matches on; recording emits `.error`); reconciling them onto one declared, type-owned serializer collapses the duplication. Policy **outputs** (command chunks) are recorded too — faithfully, once chunks are an object-valued signal (`TrajectoryOverrideSerializer` is lossy: it flattens chunks, drops the *predicted* trajectory, and cannot represent overlapping schedulers like RTC/temporal-ensembling). A separate **rerun-tap** path (`recording.py`) logs canonical policy I/O as structure-of-arrays for inference *debugging* — distinct from the dataset artifact, and a ready reference for object-chunk encoding.

**Privileged/task signals stay separate** — never policy I/O, recorded through their own path (the observer→`add_signal` mechanism).

### Ownership split (the one-glance summary)

- **Eval (the factory) owns:** the embodiment (device CSs + `observations`/`commands`/`privileged` + `descriptor`) and the task (instruction, privileged signals + success criteria, initial-state distribution + seed, optional termination predicate).
- **Submission owns:** the Policy — server code + local wrapper (codec + scheduler), embodiment-agnostic (routes on the per-call descriptor string internally).
- **CLI command owns:** trial count, seed (base), `timeout`, `inference_latency`, `output_dir`, `--policy`/`--wrap`.
- **Harness owns:** applying the embodiment's serializers, the uniform demux, episode lifecycle, recording, timeout enforcement — all name-free.
- **World owns:** time (virtual or wall) and scheduling.
- **Driver owns:** trial sequencing.

## 1.4 Principles (quick reference)

1. Product = the eval toolkit; the PhAIL leaderboard is a consequence (opt-in, post-MVP).
2. Agentic-first: the CLI is the contract; artifacts are machine-readable; the viewer is optional.
3. Local-first, cloud-optional; v1 is local-only, including local hardware.
4. Two CLI steps: `run` (produce datasets) and `analyze` (score them, post-MVP). Leaderboard submission is its own flag.
5. An eval = embodiment + task, produced by a **factory** (the eval API). Sim engines are embodiments (any gym-style Env folds in via a general adapter); success is a task concern.
6. Embodiment = a signal-dict contract (`observations`/`commands`/`privileged`/`descriptor`) returned by a factory, backed by 1 or N device CSs — **not fused**. One factory feeds the eval runner, data-collection, and replay.
7. The Harness is **name-free**: the embodiment provides the observation serializers (which own the canonical key names) and the command spec; the Harness applies them and demuxes uniformly — no arm-vs-gripper knowledge, no per-command special-casing.
8. The policy is **stateless** and **embodiment-agnostic**: a descriptor **string** + the instruction are passed every call; one submission routes across embodiments internally.
9. Outcome is **scored in analysis**, not at rollout. The rollout records raw privileged state as-is (the entire sim `save_state` for a sim task). TTS is a distribution.
10. Termination carries a **typed reason**; human-in-the-loop is the two-phase `STOP → review → FINISH(reason + annotations)`.
11. `inference_latency` and `timeout` are Harness/CLI knobs delivered on the `RUN` directive — not eval identity; recorded in meta. `inference_latency` is **sim-only** (real latency is real).
12. Seed: random default, convention `base + trial_index`; the **scene-owning embodiment** self-seeds and self-reports. Fully reproducible in sim, best-effort on real (not bit-equality).
13. **The World owns time.** `World(virtual_time)` picks a virtual clock (sim, advanced by the scheduler to the next event) or wall time (real); pacing (free-run vs paced-to-wall for a live human) is a runner mode. One persistent World; reset is a state discontinuity, never a time rewind; never recreate.
14. Serialization is **type-owned at two boundaries** (dataset + network); no `Kind` enum; recording == canonical policy I/O.
15. External sims run **under our Harness (Mode B)** — latency-honest, comparable to our evals, not bit-comparable to published baselines. Honesty is the point.
16. No new `Eval`/Benchmark ABC and no wrapper DSL — an eval is a registered factory `cfn.Config`; the only interfaces are the embodiment signal-dict contract and the `Policy`/`Session` pair.
17. One `positronic` parent command; `eval`/`inference`/`server` are groups under it (the configuronic nested command tree).

---

# Part 2 — The chain

A sequence of right-sized steps (each a reviewable PR). Foundational refactors land first (the World clock), then the embodiment/eval API and a name-free Harness, then the researcher CLI on top, then lifecycle knobs, seeding, and the real path, and finally the dataset/offboard serialization and the external-sim/bimanual generalizations. Each step is behavior-preserving or purely additive where possible.

## ✅ Done — configuronic nested command-tree (`cfn.cli`, v0.5.0)
**Repo:** configuronic (`~/Documents/dev/configuronic`, ours).
`cfn.cli` accepts an arbitrarily-nested dict of `Config`s: positional args walk the tree by key to a `Config` leaf, then `--kwargs` override and instantiate it; `--help` lists children at a group node and required args at a leaf. The flat `{'cmd': cfg}` form is the depth-1 special case. This is what `positronic <group> <verb> <name>` needs (`eval → run → <flat dotted catalog key>` → eval `Config` leaf), with `inference`/`server` as sibling groups. Released as `configuronic` 0.5.0; positronic pins `configuronic>=0.5.0`.
**Note:** leaf args must be `--kwargs`; a trailing *positional* after a `Config` leaf binds to fire's `help`, so `eval list <prefix>` uses `--prefix=<…>`.

## ✅ Done — The World owns time (pimm core, #413)
**Repo:** `pimm/`. The clock now lives in the scheduler, not in any control system. `World(virtual_time=True/False)` selects a virtual clock (sim — `interleave` reasons in integer nanoseconds and advances it to the exact next scheduled wake) or wall time (real); there is no `clock` constructor arg. `Sleep` requires a positive duration and `Yield()` expresses zero-duration cooperation. `MujocoSim` is a plain `ControlSystem` that yields `Sleep(timestep)` as the pacer and no longer subclasses `Clock`; world-time is decoupled from `data.time`, so `reset()` is a pure state discontinuity with no time-rewind. `World.run(main, background)` drives the loops (sleeping under wall time, fast-pumping under virtual time); pacing (free-run vs paced-to-wall vs wall) is a runner mode. Behavior-preserving: recorded timestamps unchanged; golden + sim-integration + harness tests green.

Step 2 is the architectural keystone, so it lands as **two reviewable PRs**: **2a** reshapes the Harness's obs assembly into a uniform channel loop and moves sim scoring downstream (additive, low-risk groundwork); **2b** externalizes the contract into the embodiment factory and makes the Harness truly name-free. The downstream chain (steps 3–13) is unchanged.

## ✅ Done — 2a: Name-free Harness *observations* + privileged full-state recording (#414)
Reshaped obs assembly into a uniform channel loop and moved sim scoring downstream, without yet externalizing the contract. What landed:
- `_build_obs` iterates an internal `_observations` dict (`name → (receiver, serializer)`) instead of hand-rolling the keys. The per-type serializers and the `name+suffix` unfold (`expand_suffixed`) were **extracted into `positronic/dataset/serializers.py`** (the `Serializers` namespace), shared by the writer and the Harness. The obs side keeps the raw `status` enum that `ErrorRecovery` matches on (`robot_state_obs`), distinct from the recording serializer (`robot_state`, which drops `RESETTING` / emits `.error`) — the two reconcile in step 7.
- A **`descriptor` string** on the Harness, put into the obs dict on **every** call — `mujoco.franka` for sim, `''` on real. No structured fields, no control frame (Why #9). Documented in the raw-obs wire contract (`connect-your-model.md`).
- `FullSimState` (a `MujocoSimObserver` over `save_state`); `stack_cubes` records it as the privileged signal via the `observers=` kwarg and **drops** the `StackingSuccess`/`BodyDistance` observers — scoring moved downstream.
**Outcome:** behavior-preserving for policy I/O (golden byte-identical); the only recorded-signal change is privileged (full state in, live observer scores out). The command side (`_step` demux, `_home`, `is_recover_only`) and `wire()` stayed name-coupled — that is 2b.

## 2b — Embodiment factory + `wire()` dedup (inference path)  *(the architectural keystone)*
**Goal:** externalize the obs/command/privileged/descriptor contract into a **factory**, making the Harness truly name-free, proven on `stack_cubes` (still runnable via the existing `positronic-inference sim` path).
**Changes:**
- Introduce the **embodiment factory** (`positronic/embodiment.py`) — a `franka(robot_arm, gripper, cameras, …)` builder shared by the sim, real, and golden paths (they share the arm/gripper CS interface). It returns `observations` (driver source + the 2a serializers — note obs carries *two* serializers, to-policy `robot_state_obs` vs to-record `robot_state`, until step 7 unifies them), `commands` (action-key → device receiver + record serializer), `home` (`{command: value}` — `robot_command → Reset()`, `target_grip → 0.0`), `privileged` (ground-truth sources, recorded-only), the `descriptor` **string**, and `static_meta`.
- The Harness receives an `Embodiment` instead of constructing channels: generic `observations` (`ReceiverDict`) / `commands` (`EmitterDict`). The demux is **republish-all** — `None` skips; `[]` cancels every channel; otherwise each command channel emits the waypoints this chunk carries for it (an omitted channel → `[]` → its signal is overwritten and the driver holds). This **deletes** the `is_recover_only` sniff, the grip-cancel special case, and the recover latency-bypass. `_home` emits the embodiment's home action; the descriptor passes every call. Privileged is declared in the embodiment (recorded-only), replacing `inference.py`'s `observers=` loop.
- `wire()` consumes the embodiment for the **inference** path (`inference.py` `main`/`main_sim`). Keep device CSs separate (no fusion); single channel today; canonical keys preserved (the codec/parity contract depends on them — now living in the embodiment's serializers, not the Harness).
**Defers:** the **teleop + replay dedup** — `DataCollectionController`/`Replay` keep their current named `wire()` wiring (the "one factory, three consumers" reuse lands with the teleop generalization, **after** bimanual; input devices like VR stay plain control systems, *not* embodiments — an embodiment is the actuated body). Multi-channel routing (step 11); the obs/recording serializer reconciliation (step 7); the typed-signal *storage* refactor (step 8).
**Bar:** behavior-preserving for policy I/O on the inference path. The **one** intentional behavior change: a recover chunk now incurs `inference_latency` in sim (the bypass is gone) — republish-all itself is byte-identical, so the **golden is regenerated** for that single timing shift. Harness + sim-integration tests green.

## 3 — `positronic eval run/list` + catalog
**Goal:** the researcher CLI, on the refactored embodiment + the command tree.
**Changes:**
- One `positronic` parent command; `eval` group with `run` and `list`.
- Flat dotted-key **catalog** registering the evals: `sim.positronic.stack_cubes`, `sim.positronic.multi_tote`. `run <name>` builds the eval (via its factory) + a trial-sequencing driver; `--policy`/`--wrap`/`--trial_count`/`--output_dir` are cfn overrides on the selected eval config.
- Output via existing `LocalDatasetWriter`; viewable in `positronic server`.

## 4 — RUN-directive knobs + `inference_latency` rename + typed termination reason
**Goal:** lifecycle knobs and termination become explicit and recorded.
**Current state it builds on:** `simulate_inference` is a Harness *constructor* param; timeout is *external* (a driver emits `FINISH`). Both move onto the directive.
**Changes:**
- Move `timeout` and `inference_latency` off the constructor onto the `RUN` directive; the Harness self-terminates at `start + timeout`. The driver becomes a pure trial-sequencer.
- Rename `simulate_inference` → **`inference_latency`** (`False`=ignore / `True`=measured wall time / `float`=fixed delay); it is sim-only.
- Typed termination **reason** — an **enum** (`success`/`fail`/`stalled`/`safety`/`system`/`out_of_time`), matched exhaustively by the Harness for disposition + home (save/flag/drop), recorded as its string value. Generalizes the free-form `FINISH` payload into a typed field.
**Defers:** auto-detect termination producers (arrive with real scales / LIBERO `check_success`).

## 5 — Sim seeding + eval-identity meta  *(end of sim MVP)*
**Goal:** reproducible, self-describing sim rollouts.
**Changes:**
- The embodiment **self-seeds** (`base + n` on its `n`-th reset; loaders stay `seed=None` under an `np_seed(base+n)` wrapper) and **self-reports the seed** into episode meta. Same seed → same scene (fully reproducible).
- Rollout stamps the **identity static-meta block** (`eval.name`/`eval.universe`/`eval.embodiment`/`eval.task` + instruction) from the catalog entry, plus the success-criterion id; **dump the resolved eval cfn config** ("record all we can"). The policy/codec/wrapper/session composition already surfaces — `_build_episode_meta` merges `policy.meta | session.meta` under `inference.policy.*` — so this step adds the eval-identity keys + seed.

## 6 — Real (`real.droid.pick_place`) + human-in-the-loop  *(in parallel with 5)*
**Goal:** retire the parallel real impl early — don't keep two implementations for long.
**Depends on:** 2 + 4 (the embodiment factory + directive knobs + typed reason). Independent of 5 (real has no scene seed; physical/human reset; real already has real latency).
**Changes:**
- Register the real tote **pick-and-place** eval (`EvalUI`'s `UNIFIED_TASK`, variants `.wooden_spoons` etc.) via its own embodiment factory — *not* `stack_cubes`, which real doesn't have.
- Two-phase **`STOP` → review/annotate → `FINISH(reason + annotations)`** via `EvalUI`; record hardware-identity meta (camera serials, robot id).

## 7 — Unify recording on the declared serializer
**Goal:** one serializer for "feed the policy" and "record the signal."
**Changes:**
- Reconcile the embodiment's observation serializer with the recording serializer (the obs side keeps the raw `status` enum for `ErrorRecovery`; recording emits `.error`). Recording becomes exactly the **canonical policy I/O** (submission-independent because the codec lives below the Harness boundary). Privileged/task signals stay on their own path; the rerun-tap debug path (`recording.py`) stays separate.

## 8 — Dataset: type owns serialization at the `Signal` boundary
**Repo:** `positronic/dataset/`.
**Goal:** remove `Kind`; record policy outputs faithfully.
**Changes:**
- Push (de)serialization onto **domain types** (array-backed for transport, the `MujocoFrankaState` pattern); `Signal[T]` carries the domain type through storage. **Remove the `Kind` enum** and the `uint8-HxWx3` heuristic — the type selects its backend (parquet/video/**object**).
- A command **chunk** becomes an object-valued signal → **retire `TrajectoryOverrideSerializer`** (lossy: drops predicted trajectories, can't represent overlapping schedulers). Unblocks faithful policy-output recording + RTC/temporal-ensembling. (`recording.py`'s `action_chunk_arrays` is a ready structure-of-arrays reference.)

## 9 — Offboard: type owns serialization at the network boundary
**Repo:** `positronic/offboard/`.
**Goal:** the same type-keyed codecs at the `InferenceServer`/`Client` wire boundary, completing train→infer parity (the same typed value crosses dataset *and* network).

## 10 — Gym-Env → embodiment adapter (LIBERO the first instance, Mode B)
**Goal:** a reusable utility that turns any gym-style `reset()/step()` Env into an embodiment — re-hosted in our latency-honest Harness; brings robosuite in.
**Changes:**
- A general **gym adapter**: one fused embodiment CS that the World's virtual clock drives via `env.step()` (synchronous, no real-time assumption). Per tick it samples the due waypoint from the buffered trajectory (the `TrajectoryPlayer` role pulled inside), maps **canonical command → env action**, steps, and maps **env obs → canonical observations** + **env state → privileged signals**. The integrator supplies those four mappings + **seed → init-state**.
- **LIBERO** as the first instance: **absolute-mode OSC** (`control_delta=False`) — forced by the no-op = hold-last-*commanded* invariant; `Transform3D` → pos + axis-angle; `seed % num_init_states`; `check_success()` records env ground-truth (and may be a live termination signal).
- Latency honesty holds at control-tick (20 Hz) granularity. **Risks (process, not architecture):** robosuite/LIBERO MuJoCo version vs positronic's; camera obs at control rate unless rendered off-cycle.

## 11 — Multi-channel embodiment (bimanual) on robosuite TwoArm
**Goal:** the multiplicity generalization deferred since step 2.
**Note:** positronic has **no native bimanual** (only `franka`/`kinova`/`so101`, all single-arm; `trossen_aloha` is aspirational). Use **robosuite's TwoArm envs**, which come with the gym adapter / robosuite integration from step 10.
**Changes:** channel-keyed `observations`/`commands` (`left_arm.*`/`right_arm.*`); status aggregation across channels; multi-channel codec; generalize the single-arm `RobotState` bundle.

## 12 — Driver uniformity: gripper as a command stream  *(independent; any time after step 2)*
**Goal:** make every command channel the same *type*, removing the last assumption in the demux.
**Why:** the arm channel speaks `Command` objects while the gripper speaks raw floats, so a uniform `Hold`/cancel can't flow down the grip channel today, and the demux leans on "empty trajectory = hold" (`TrajectoryPlayer.set([])` clears the buffer). Promote the gripper to a command stream (`SetGrip`/`Hold`), make `set([])` a genuine no-op (keep playing), and let an explicit cancel/`Hold` flow uniformly. Then the Harness always emits every channel without the hold-on-empty assumption, and asymmetric-rate channels (a future codec emitting arm-without-grip) are safe.

## 13 — Teleop & replay on the embodiment + general device→command mapping  *(after bimanual)*
**Goal:** retire the second device-construction copy (`DataCollectionController`/`Replay` adopt the embodiment's generic channels — the "one factory, three consumers" dedup deferred in 2b) and generalize teleop input beyond a single 3D pose.
**Depends on:** 2b (the factory) + 11 (multi-channel commands, so bimanual teleop has left/right arm channels to bind to).
**Changes:**
- Move the teleop/replay `wire()` producers onto the embodiment's generic `commands`/`observations`; drop the interim named-port wiring 2b left them on (after this, `wire()` carries one interface, not two).
- A **device→command mapping** — a configured list of mappings, each binding named input-device signals (a controller/phone pose + clutch + analog) to embodiment command channels and owning its own tracker/clutch state. One mapping today; N for VR-bimanual (`right→right_arm`, `left→left_arm`) or several phones. **Input devices stay plain control systems, not embodiments** (the embodiment is the body that acts); their signals are wired into the obs/recording pool. This is the teleop counterpart to the action codec.
- Lifecycle stays orthogonal — record-buttons → a small directive driver → Harness `RUN`/`STOP`; control signals → the mapping → commands — so start/stop is never mixed into the control path (the snag that makes today's monolithic `DataCollectionController` hard to generalize). Teleop-as-Policy is possible but the ~10× loop-rate gap and the empty wrapper stack argue for "share the embodiment, not the Harness."
**Adjacent:** leader-follower (one arm's *measured* state driving another's commands) is a natural mapping variant on the same multi-channel foundation.

## Beyond this chain (future work — not detailed here)
Tracked separately, intentionally out of this doc: `positronic eval analyze` (success criteria + TTS over recordings → CSV/summaries); the cloud queue (account, hosted storage, real-robot submission); leaderboard submission (its own flag) and submission-trust / anti-cheat.

**Architectural direction (deferred — noted, not pulled).** *The policy as a control system* — emitting its action-plan trajectory on a signal channel instead of returning it synchronously. The action output is already signal-shaped (latest-wins, complete each call — the same shape as the 2b command channels), so this is the honest completion: inference latency would become the policy CS's natural `Sleep` (latency-honesty *structural*, no `inference_latency` Harness hack), and the world would stop blocking on the network RTT. **Why not now:** the synchronous `session(obs) → actions` is far more debuggable, and the one motivation that would pull on it — overlapping schedulers (RTC, temporal ensembling) — is instead absorbed as **alternative `ChunkedSchedule` implementations** inside the sync model, so nothing forces the async rewrite. The trade only makes sense if a concrete need (a measured blocking-stall on real control) later appears. Teleop, by contrast, must *never* go through this path regardless — its low-latency budget can't absorb the extra signal hops.

---

# Part 3 — Reference

## 3.1 Research context (read if picking up cold)

Workspace docs (in `/Users/vertix/Documents/dev/workspace/projects/phail/`):
- `benchmarking-landscape.md` — ICRA 2026 manipulation field + vla-eval deep-dive + Positronic comparison. **Read first.**
- `icra-targets.md` — per-paper directory of ICRA 2026 outreach targets (10 deep-dives w/ hardware/tasks/baselines).
- `strategy.md` — positioning, leaderboard mechanics, real-robot value prop.
- `marketing-strategy.md`, `content-strategy.md` — go-to-market.

External references:
- vla-eval repo: `github.com/allenai/vla-evaluation-harness` (Apache 2.0); paper arXiv:2603.13966.
- STAR-Gen (closest competitor concept): arXiv:2503.01238.
- phail-paper (TTS methodology): `…/phail/phail-paper.md` — TTS is a distribution with potentially multiple successes per episode.

## 3.2 Code pointers (current architecture)

- `pimm/world.py` — `World` (the scheduler `interleave`, the clock, `start`/`connect`/`pair`), `SystemClock`. Step 1 moves the virtual clock into `interleave` and adds `World(virtual_time)`.
- `pimm/core.py` — `Clock`, `Sleep`/`Pass`, `ControlSystem`, `SignalEmitter`/`SignalReceiver`, `ReceiverDict`/`EmitterDict`.
- `pimm/shared_memory.py` — `SMCompliant`/`NumpySMAdapter` (the array-backed-typed-object transport pattern).
- `positronic/inference.py` — CLI entry (`positronic-inference sim/real/phail/sim_pnp`), `main_sim`/`main`, `TimedDriver`, the flat `cfn.cli({…})` dispatch.
- `positronic/policy/base.py` — `Policy` (**factory**: `new_session(context) → Session`), `Session` (`__call__(obs) → list[dict] | None`), `PolicyWrapper`/`_Pipeline` (`|` composition), `DelegatingPolicy/Session`, `SampledPolicy`.
- `positronic/policy/codec.py` — `Codec` (`|`/`&`, `encode`/`decode`/`training_encoder`), `lerobot_*` feature helpers. The representation layer (inside the submission, below the Harness boundary).
- `positronic/policy/observation.py` — `ObservationCodec` (same keys train+infer). `positronic/policy/action.py` — `AbsolutePositionAction`/`AbsoluteJointsAction`/`IKJointsAction`/`RelativePositionAction`/`JointDeltaAction` (decode to canonical `{'robot_command','target_grip'}`).
- `positronic/policy/harness.py` — `Harness` (`_build_obs` → `session(obs)` → demux), `Directive`, `ChunkedSchedule`/`ErrorRecovery`/`default_wrappers` (scheduling/recovery as wrappers), `on_episode_complete`, `simulate_inference` (→ `inference_latency`, step 4). Step 2 makes it name-free.
- `positronic/policy/recording.py` — `Recorder` + `tap()` (rerun `.rrd` inference-debug taps); `action_chunk_arrays` (chunk → structure-of-arrays, step 8 reference).
- `positronic/policy/remote.py` — `RemotePolicy` (`host, port, resize, model_id, headers, secure`).
- `positronic/offboard/{basic_server,vendor_server,client}.py` — `InferenceServer`/`InferenceClient` (FastAPI + WebSocket). The network serialization boundary (step 9).
- `positronic/dataset/ds_writer_agent.py` — `DsWriterAgent`, `Serializers` (per-type codecs), `TrajectoryOverrideSerializer` (the hack step 8 retires), `StatefulSerializer`. `_append` expands a dict serializer result into `name+suffix` signals.
- `positronic/dataset/signal.py` — `Signal[T]`, `SignalMeta`, `Kind` (removed in step 8). `local_dataset.py` — `LocalDatasetWriter`, `DiskEpisodeWriter.append` (the shape/dtype routing step 8 removes).
- `positronic/cfg/simulator.py` — `stack_cubes_loaders`, `multi_tote_loaders`; `positronic/simulator/mujoco/{sim,observers,transforms}.py` — `MujocoSim`/`MujocoFranka`/`MujocoGripper`/`MujocoCameras`, `save_state`/`FullSimState`, `SetBodyPosition`/`np_seed`. `MujocoSim` becomes a plain CS in step 1.
- `positronic/drivers/roboarm/command.py` — `TrajectoryPlayer` (shared by every arm/gripper driver; `set([])` clears the buffer → hold). Step 12 makes empty a no-op + adds an explicit `Hold`.
- `positronic/wire.py` — `wire()` (shared by the Harness, `DataCollectionController`, and `Replay` — all expose the same named producer ports), `ROBOT_STATIC_META`. Step 2b routes the **inference** path through the embodiment; teleop/replay follow with the teleop generalization (post-bimanual).
- `positronic/data_collection.py`, `positronic/replay_record.py` — the other two `wire()` producers; both build the same device CSs by hand (deduped into the factory with the teleop generalization, after step 11 — 2b leaves them on the current named wiring).
- `positronic/gui/eval.py` — `EvalUI` (human-in-the-loop: typed outcomes, notes, partial success, timeout).

configuronic (`~/Documents/dev/configuronic`, ours, `>=0.5.0`): `config.py` (`Config`, `override`, `instantiate`, `@module.path` / `.relative`), `cli.py` (`cfn.cli` nested command tree).

## 3.3 What we are explicitly NOT doing (filed for discipline)

- **Not integrating vla-eval as upstream.** No adopting their harness / Docker-per-benchmark architecture; external sims become our own embodiments under Mode B.
- **Not building an artifact contract beyond the Positronic Dataset** (no `run.json`/`methodology.json`/hash/README). Dataset + existing viewer carry MVP.
- **Not building a leaderboard UI, cloud queue, or submission-trust/anti-cheat in MVP.**
- **Not auto-sweeping schedulers** for the user — one per submission.
- **Not generalizing the Harness for multiple command channels yet** — the signal-dict boundary is channel-ready, but multi-channel routing/status-aggregation/codecs land with step 11 (first real bimanual).
- **Not implementing a Benchmark ABC** — the cfn factory pattern suffices until ~5 engines force a pattern.

## 3.4 Open questions

1. **Identity static-meta schema.** Exact key set for the identity block (the policy/codec/wrapper/session composition already surfaces via `policy.meta | session.meta` in `_build_episode_meta`); the LIBERO additions (init-state index, BDDL goal id).
2. **Real-hardware registration.** How a user declares their Franka (port/kinematics/cameras) to the catalog — a cfn config in their module via `--embodiment=@my.module.my_franka`, or a `~/.config/positronic/hardware.toml`?
3. **Auth ergonomics.** `--policy.headers='{"Authorization":"Bearer xxx"}'` works but is awkward — a `--auth-bearer` convenience or a `~/.netrc`-style file?
4. **Catalog discovery.** In-package now; Python entry-point plugins later — design-aware, not built.

---

# Part 4 — Design rationale (non-trivial whys)

_The reasoning behind the non-obvious calls. Each is **decision → tempting alternative → why**. Also feeds the positronic design paper._

## Foundational stance: composable, duck-typed control systems — why not `gym.Env`

pimm control systems compose by **duck-typing on signals**, not by implementing a rigid interface. A component is "whatever exposes the right named signal ports"; the World wires them. The embodiment contract is this stance applied to the model↔robot boundary: a set of named signal dicts — `observations` (→ policy) and `commands` (← policy) — that *any number* of control systems can satisfy.

**`gym.Env` is the anti-pattern.** Its `reset()/step(action)` + `observation_space/action_space` is a single rigid shape: one synchronous loop, one action per tick, one obs, the env owns the clock. It cannot express (a) N independently-swappable devices, (b) multi-rate async signals, or (c) latency-honest execution where the world advances while the model thinks. (Same "not a Gym wrapper" theme as the codec layer, one level down — at the runtime/embodiment boundary, not the data-transform boundary.) We *adapt* gym-style envs into the signal-dict contract (the integrator supplies command→action, obs→canonical, state→privileged, seed→init mappings); we don't adopt `gym.Env` as the interface.

## Whys

1. **Embodiment = a signal-dict contract returned by a factory, not a control-system class.** *Tempting:* an `Embodiment` base class, or fuse the devices into one CS. *Why not:* the contract must hold for **both** one fused CS (an external sim's atomic `env.step`) **and** N independent device CSs (real, where you swap an arm but keep the gripper/cameras). Pin it to "a control system" and you force one or the other; pin it to the **signal boundary** and both satisfy it by duck-typing. Fusion specifically kills device-swap composability. A factory (not a class) returns the bundle, so the same construction feeds the eval runner, data-collection, and replay.

2. **The Harness is name-free; the embodiment owns the names.** *Tempting:* let the Harness hand-roll `robot_state.q`/`grip` and special-case the gripper. *Why not:* the Harness is a pure router — it should know nothing about *which* signals exist. The embodiment provides the observation serializers (which own the canonical key names) and the command spec; adding a channel or a new embodiment is then a factory change, not a Harness change. The demux is uniform: an empty per-channel trajectory means "hold," which is the *driver's* job, not an arm-vs-gripper branch in the router. (The remaining type asymmetry — arm speaks `Command` objects, gripper speaks floats — is a driver cleanup, not a Harness one.)

3. **Embodiment ≠ task; success never enters the policy.** *Tempting:* observers/success are part of the env the policy sees. *Why not:* the policy is driven only by what it controls; success is privileged ground truth. The separation is what lets scoring move to analysis and lets one embodiment carry many tasks. The single policy-facing task element is the instruction (prompt).

4. **Record raw privileged state; score in analysis.** *Tempting:* compute success live (you already hold the state). *Why not:* one expensive rollout should feed many cheap, re-runnable criteria — record the entire `save_state` and any future criterion (and the TTS distribution) is derivable without re-running, decoupling the rollout from the still-evolving definition of success. TTS is a distribution, not a scalar, because scoring reads a time series.

5. **External sims run under our Harness (Mode B), not their native loop.** *Tempting:* run LIBERO's own synchronous loop so numbers match published baselines. *Why not:* that loop is latency-free and synchronous; re-hosting under our async, latency-honest Harness (the world advances while the model thinks; the robot executes the in-flight chunk through inference latency) is the methodological point — honesty over baseline-matching. The cost (numbers not bit-comparable to published baselines) is accepted deliberately.

6. **No-op = hold the last *commanded* setpoint (not measured).** *Tempting:* zero-delta, or hold the measured pose. *Why not:* only "re-assert the last *commanded* target" is identical across sticky-`ctrl` MuJoCo, a real low-level controller, and a wrapped external sim — the universal invariant that makes embodiments interchangeable. It is also what forces absolute-mode OSC when adapting a delta-controller env (a zero delta would drift to measured pose, not the commanded one).

7. **The World owns time; the clock lives in the scheduler.** *Tempting:* let the sim be the clock (as today), or add a dedicated "advance-time" control system. *Why not:* a sim-as-clock couples world-time to engine-time (forcing the reset-time-rewind hack), and a separate clock CS forces a fixed-`dt` discretization on *everything*. Putting the clock in `interleave` lets it jump to the **exact next scheduled event** ("as fast as we need"), makes sim-vs-real a single `virtual_time` flag, and — because world-time is decoupled from any engine's internal time — turns reset into a clean **state-only** discontinuity. Pacing is orthogonal: the same virtual clock either free-runs (eval) or tracks wall time (a live human in VR data-collection).

8. **The scene-owning embodiment owns the seed sequence (self-seeds).** *Tempting:* pass `base + i` on the RUN directive. *Why not:* re-randomization for trial *i* happens at trial *i-1*'s `FINISH` — before trial *i*'s context exists — so directive-borne seeds fight the lifecycle ordering. The embodiment that owns scene reset is the only place the seed sequence lives cleanly; it self-reports the seed it used into episode meta.

9. **The descriptor is a bare string.** *Tempting:* a structured dict (robot type, dof, joint names, control frame, URDF). *Why not:* nothing consumes structured fields yet — the descriptor is purely the per-call routing hook for a future multi-embodiment policy, so it stays a minimal identifier. The control frame in particular does not belong: a robot accepts commands in *different* frames, so the frame travels with the command/codec, not the embodiment's identity.

10. **Type-owned serialization at two boundaries; no `Kind`.** *Tempting:* per-signal declared serializers, or a `Kind` enum inferred from shape+dtype. *Why not:* the same `np.array` is ambiguous (a 7-vector is a pose-quat or 7 joints); only the **domain type** disambiguates. Give the type its (de)serialization at **both** the dataset `Signal` boundary and the offboard/network boundary and one principle removes `Kind`, the shape/dtype heuristic, per-signal declaration, and the lossy trajectory-flattening serializer at once — while making train↔infer parity structural (the same typed value crosses both boundaries).
