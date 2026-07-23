# Architecture invariants

Every integration in Positronic — a simulator, a model stack, a scene catalog, a scoring method, a
real rig — is shaped by the invariants below. They are not conventions of individual modules; they
define what an integration is allowed to look like.

## Principles

The invariants follow from five principles. When a new case is not settled by the invariants, decide
it from these.

**Everything must be recoverable from the recording.** The only irreversible cost is the physical
rollout and what was captured of it. Scores, action spaces, control frames, thresholds, metrics and
vendor formats are projections — re-derivable offline from a complete recording. So capture raw and
complete, and defer every choice a projection can express. Care belongs on the expensive layer
(protocol, capture completeness, event ontology), not on the cheap one.

**Bind late.** The party that owns a requirement declares it; the run does not fix it in advance. A
policy trained in another end-effector frame declares that frame; a policy needing a different
controller declares its control mode; a trial carries its own instruction and done predicate.
Binding these per session collapses the comparison — two policies wanting different execution can no
longer be interleaved in one A/B session, which is what evaluation exists to do.

**Whatever varies travels as data, not as ambient state.** Time enters the pipeline as an
observation field; control mode is a declared channel with a rig default; hardware identity is a
`meta` port every driver emits. Nothing that varies is read from a global, a constructor argument,
or a config the run captured once. This is what makes the two principles above mechanical rather
than aspirational: a value that flows through a channel is recordable, substitutable and replayable
by construction — a virtual clock, a different controller, a replayed episode all come for free.

**One specification, both directions.** A transformation between our data and a model's world is
defined once and drives both training and inference. Two implementations of one contract diverge,
and the divergence stays invisible until the robot performs badly. A `Codec` owns `encode`/`decode`
and its training encoder in one object for exactly this reason; the rule generalizes to any contract
with both a training-time and a runtime side.

**Guarantees are structural, not conventional.** What must not cross a boundary must be unable to
cross it. The policy does not see the seed, the task id or the rig's frame convention because those
keys are dropped at the wire — not because every codec is trusted to ignore them. A guarantee held
up by agreement is not one, and for an evaluation "we do not look at it" is not a claim an outsider
can check.

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
