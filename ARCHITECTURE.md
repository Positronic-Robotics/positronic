# Architecture

Every integration in Positronic — a simulator, a model stack, a scene catalog, a scoring method, a
real rig — follows from what is below: the goals state what the system guarantees, the principles
are the means, and the invariants and decisions that follow are consequences. When a new case is
not settled here, decide it the same way — from the goals, through the principles.

[CLAUDE.md](CLAUDE.md) is how code here is written; the named rules in
[CODE_RULES.md](CODE_RULES.md) are what a review cites. A component with principles of its own keeps them in an
architecture document beside its code —
[`positronic/dataset/ARCHITECTURE.md`](positronic/dataset/ARCHITECTURE.md), which its `AGENTS.md`
links.

## Goals

**Any policy runs on any embodiment — and cannot tell sim from real.** A policy sees observations
and emits commands through one contract. Whether they come from the native MuJoCo world, a foreign
simulator behind the env wire, or a physical rig is invisible to it, unless the embodiment itself
chooses to leak the difference.

**Sim runs are reproducible.** Whenever the environment itself is non-stochastic, the same policy
over the same scene replays to the same rollout.

**Only the physical rollout is irreplaceable.** The one cost that cannot be re-paid offline is the
rollout and what was captured of it. Everything else — scores, action spaces, control frames,
thresholds, metrics, vendor formats — must stay re-derivable from the recording.

## Principles

**Bind late.** The party that owns a requirement declares it; the run does not fix it in advance.
A policy trained in another end-effector frame declares that frame; a policy needing a different
controller declares its control mode; a trial carries its own instruction and done predicate.
Binding these per session collapses the comparison — two policies wanting different execution can
no longer be interleaved in one A/B session, which is what evaluation exists to do. Corollary: the
library must supply the tools that make late binding possible — codecs, per-trial tasks,
projections over raw recordings.

**Every decision lives with the party that has the information.** Only a driver knows its motion
capabilities, so drivers own how they reach a commanded setpoint. Only a sensor knows its own
cadence, so sensors run at their own rate instead of a rate the loop imposes. Only a policy knows
what its model was trained on, so translation to model I/O ships with the policy.

**Components are functions over flowing data.** A component sees nothing but its inputs and touches
nothing but its outputs. Whatever varies enters as data — time is an observation field, hardware
identity is a `meta` port every driver emits — never as a global, a constructor argument, or a
config the run captured once. Separable stages stay separate, and processing is deterministic where
possible. One property, many payoffs: a component moves across process and machine boundaries
unchanged, either side of a boundary can be implemented in any language, and every boundary can be
tapped for recording or replay.

**Capture raw, project on demand.** Record the raw-most values the loop saw, completely; defer
every choice a projection can express. Care belongs on the expensive layer (protocol, capture
completeness, event ontology), not on the cheap one — a projection can be recomputed tonight, a
missing capture is lost forever.

**Guarantees are structural, not conventional.** What must not cross a boundary must be unable to
cross it. The policy does not see the seed, the task id or the rig's frame convention because
those keys are dropped at the wire — not because every codec is trusted to ignore them. A
guarantee held up by agreement is not one, and for an evaluation "we do not look at it" is not a
claim an outsider can check.

## Positronic owns the control loop

The world runner and harness execute every episode: they drive the clock, deliver observations to
the policy, schedule and play back action chunks, and own resets and episode boundaries. This holds
for every evaluation and data-collection run — in simulation and on real hardware, for any
embodiment, scene source, or scoring method. A foreign component never runs the loop and calls into
Positronic; Positronic runs the loop and calls into it.

## Every run produces a Positronic dataset

The dataset is the invariant output. Any run — eval or collection, sim or real, whatever the
scoring — records the complete episode as a Positronic dataset (signal files plus episode meta, in
the native dataset layout) under its output directory. An operator may omit the output directory to
throw a smoke run away; that is a per-run choice, not an integration shape — every integration
records through the same writer path, and none may exist that can only produce scores without the
dataset. Scores, videos, and metrics are derived from data the loop recorded; they never replace
the dataset.

## Foreign components plug in through shims

A third-party component joins by having its interface wrapped into Positronic's APIs — never by
Positronic's components being wrapped into its. (Positronic's own MuJoCo sim is native: it runs
in-process inside the world and needs no shim.)

| Foreign component | Runs as | Shim into our API |
|---|---|---|
| Foreign simulator (LIBERO, Isaac Lab / RoboLab) | env server in its own interpreter, behind the `env_server` wire | client-side `EnvAdapter` mapping the canonical embodiment contract ↔ the sim's raw payloads |
| Model stack (LeRobot, GR00T, OpenPI) | inference server behind the WebSocket wire | vendor `Codec` translating raw observations ↔ model I/O |
| Scenes / task batteries | instantiated inside the env server | reset tokens (suite, task, seed) carried through the `EnvAdapter` |
| Scoring / success criteria | computed where the ground truth lives (usually the env server) | reported alongside observations and recorded into the dataset; aggregation happens on the Positronic side |
| Hardware embodiment | pimm drivers inside the world | the same canonical embodiment contract the sims speak |

The corollary for frameworks that ship their own eval harness: when a third-party benchmark expects
a policy object plugged into *its* loop, the integration still separates that framework's sim/task
layer and serves it behind the env wire. Handing a Positronic policy to a foreign loop forfeits
both invariants — the run produces no dataset, and execution is scheduled by code we don't control.

Two import boundaries follow for `positronic/vendors/`, where the shims live, and a structural test
(`positronic/tests/test_vendor_boundary.py`) enforces both. Nothing outside `positronic/vendors/`
imports from it: core defines the contract and a vendor adapts to it, so an import the other way
inverts the dependency and drags the vendor's optional extra into code that must work without it.
And no vendor imports another: each shim answers to its own upstream, on its own pinned deps and
often its own interpreter — a helper two of them need belongs in core.

## Derived decisions

What the goals and principles force. Each is stated with what forces it —
revisit a decision only by revisiting its premises.

**Time is an observation.** A policy that cannot tell sim from real cannot be allowed to read the
wall clock, and a reproducible sim needs a single owner of "now". A real rig is asynchronous
besides: sensors and deciders each run at their own frequency, so there is no global tick to share.
Hence "now" reaches the policy as a field of the observation (`obs_time_ns`), the world hands every
control system its clock, and no component reads time at point of use. Trajectories are stamped in
the same time frame the observations carry, so a virtual clock, a slowed sim, or a replayed episode
changes nothing downstream.

**The layer owns the plan, the harness plays it, the driver executes.** A policy speaks in
trajectories — waypoints with absolute timestamps — because a model predicts a horizon, not an
instant. But a trajectory on the wire makes every driver buffer the future, and makes the recording
guess which prefix of that buffer actually ran. So the plan stops at the harness: a command channel
carries the single command due at the moment it is emitted, the driver executes the latest one and
holds otherwise, and emission time *is* execution time. Continuous-update schemes (RTC, temporal
ensembling) therefore need no special mechanism: they are layers that hand back a new trajectory
more often, and the harness keeps playing the old one until they do.

**The harness stays thin.** It is the one layer standing between any policy and any embodiment, so
anything it encodes about either side breaks the any-to-any goal. It assembles the observation
dict, calls the session, plays the returned trajectory one command per channel per round, and runs
episode lifecycle — nothing else. Scheduling, blending, history stacking and error recovery live in
the layer stack around the policy; a session returning `None` means "keep executing the current
trajectory".

**Inference cost is a fact of the trial, owned by the harness.** The policy declares its heavy work
as functions, and the framework runs each one off the loop thread. That work costs the trial either
the wall time it took or nothing: the task's `charge_inference_time` flag asks a sim for the former,
and a real rig pays it regardless. Only the harness reads the flag. Paying nothing means the loop
waits for the work, which keeps a virtual clock still; paying wall time means letting the world run,
though no further ahead of the work's start than wall time has. The harness reads the world clock and
gives the reading to the policy stack as the call's `time_ns`. A scheduling layer stamps its chunk with
that value, and never learns the mode.

**Recordings are canonical; codecs bind the dialect late.** The dataset records every run in the
canonical conventions (frames, key names, absolute time) — never in a model's dialect. Every
model-facing view — action space, control frame, vendor format — is a codec's projection.
A codec owns `encode`/`decode` and its `training_encoder` in one object, so the projection that
builds the training set and the transformation applied at inference are one specification and
cannot diverge.

**The rollout records, analysis scores.** Positronic computes no verdicts in the loop: the run
records the raw privileged state as-is (for a sim task, the entire simulator state), and success
criteria are functions over those recordings, computed in a separate analysis pass — one expensive
rollout, many cheap criteria experiments. A stop-signal may end a trial early, and a foreign env
may report its own success at termination; both are captured as original data from where the
ground truth lives, not as scores — analysis is free to use, recompute, or override them. A
criterion baked into the run is bound too early: changing it would mean re-running the robot.

**Telemetry is observability, not data.** The dataset is the recording of the robot's world under the
run's clock, which for a sim eval is virtual and may be slowed, frozen or replayed. Operational
telemetry — the wall-clock cost of each phase, machine load, inference latency — describes the
machinery around that world, and it only means something in wall time. Marrying the two forces a
virtual clock onto wall-clock measurements (a slowed sim would report inflated compute cost) and
leaks a timing vocabulary into the dataset core. Hence telemetry is a set of sidecar files, per
process, next to the dataset but never inside it: nested spans for the phase split and a free-running
machine-load sampler, wall-clock native, owned by `positronic/telemetry.py`. The pass report is an
offline reduce over those raw files, so nothing is stored twice and the dataset stays clock-agnostic.

**An adoption loses nothing.** A task is defined in real or in one simulator, and carries that
simulator with it — object poses, success criteria and horizons included. What a customer buys is
one API across all of them: they implement a single Positronic policy interface and their model
runs on every supported env, giving up nothing the env offers natively. Two requirements hold that
up, one on each side of the interface:

- Given a policy that already drives an env directly, it must be possible to construct a Positronic
  `Policy` equivalent to it. This binds policy construction as much as it binds the adoption.
- An adoption's capabilities match what its env provides natively, so a Positronic run reproduces
  the env's own run: a deterministic env to byte-identical outcomes (modulo wire format), a
  non-deterministic env to an identical sim/inference call sequence (same count, same order).

Every sim-env adoption ships a native-vs-Positronic parity test that drives one pinned episode
through both stacks and asserts this, re-run on every bump of the sim's pinned version.

The episode horizon is one case of that reproduction rather than a rule of its own: a task that
defines a horizon has it enforced by the env, which reports expiry through the same terminal `done`
a success uses; a task that defines none leaves nothing to reproduce. The harness `Task.timeout` is
only a runaway-cost safety net, so the config that knows the benchmark derives the timeout from the
horizon it declares rather than taking one on faith — a budget below the horizon would silently
truncate valid episodes and score them as failures.
