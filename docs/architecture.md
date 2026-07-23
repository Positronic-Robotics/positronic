# Architecture

Every integration in Positronic — a simulator, a model stack, a scene catalog, a scoring method, a
real rig — is shaped by this document. It is a derivation, not a rule list: the goals state what
the system guarantees, the principles are the means, and the invariants and decisions that follow
are consequences. When a new case is not settled here, decide it the same way — from the goals,
through the principles.

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
library must supply the tools that make late binding possible — codecs, per-trial context,
projections over raw recordings.

**Every decision lives with the party that has the information.** Only a driver knows its motion
capabilities, so drivers plan through waypoints. Only a sensor knows its own cadence, so sensors
run at their own rate instead of a rate the loop imposes. Only a policy knows what its model was
trained on, so translation to model I/O ships with the policy.

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

## Derived decisions

The load-bearing consequences of the goals and principles. Each is stated with what forces it —
revisit a decision only by revisiting its premises.

**Time is an observation.** A policy that cannot tell sim from real cannot be allowed to read the
wall clock, and a reproducible sim needs a single owner of "now". A real rig is asynchronous
besides: sensors and deciders each run at their own frequency, so there is no global tick to share.
Hence "now" reaches the policy as a field of the observation (`obs_time_ns`), the world hands every
control system its clock, and no component reads time at point of use. Trajectories are stamped in
the same time frame the observations carry, so a virtual clock, a slowed sim, or a replayed episode
changes nothing downstream.

**Trajectory is the command.** Ownership puts execution with the driver: the policy emits a
trajectory of waypoints with absolute timestamps, and the driver plays it at its own control rate,
planning through the waypoints as well as it knows how. Signals are last-value-wins, so a new
trajectory overwrites the current one — the previous command is merely context for the next.
Continuous-update schemes (RTC, temporal ensembling) therefore need no special mechanism: they are
wrappers that rewrite the command more often. An empty trajectory cancels the channel and the
device holds.

**The harness stays thin.** It is the one layer standing between any policy and any embodiment, so
anything it encodes about either side breaks the any-to-any goal. It assembles the observation
dict, calls the session, demuxes the returned waypoints per command channel, and runs episode
lifecycle — nothing else. Scheduling, blending, history stacking and error recovery live in the
wrapper stack around the policy; a session returning `None` means "keep executing the current
trajectory".

**Recordings are canonical; codecs bind the dialect late.** The dataset records every run in the
canonical conventions (frames, key names, absolute time) — never in a model's dialect. Every
model-facing view — action space, control frame, vendor format — is a codec's projection.
A codec owns `encode`/`decode` and its `training_encoder` in one object, so the projection that
builds the training set and the transformation applied at inference are one specification and
cannot diverge.

**The rollout records, analysis scores.** The run computes no verdicts: it records the raw
privileged state as-is (for a sim task, the entire simulator state), and success criteria are
functions over those recordings, computed in a separate analysis pass — one expensive rollout, many
cheap criteria experiments. The only live exception is an optional stop-signal that ends a trial
without judging it; whether that end was a success is, like everything else, an analysis question.
A criterion baked into the run is bound too early: changing it would mean re-running the robot.
