# Architecture invariants

Every integration in Positronic — a simulator, a model stack, a scene catalog, a scoring method, a
real rig — is shaped by two invariants. They are not conventions of individual modules; they define
what an integration is allowed to look like.

## Positronic owns the control loop

The world runner and harness execute every episode: they drive the clock, deliver observations to
the policy, schedule and play back action chunks, and own resets and episode boundaries. This holds
for every evaluation and data-collection run — in simulation and on real hardware, for any
embodiment, scene source, or scoring method. A foreign component never runs the loop and calls into
Positronic; Positronic runs the loop and calls into it.

## Every run produces a Positronic dataset

The dataset is the invariant output. Any run — eval or collection, sim or real, whatever the
scoring — records the complete episode as a Positronic dataset (signals, episode meta, per-episode
`.rrd`) under its output directory. Scores, videos, and metrics are derived from data the loop
recorded; they never replace the dataset, and an integration path that yields scores without the
dataset is not acceptable.

## Foreign components plug in through shims

A third-party component joins by having its interface wrapped into Positronic's APIs — never by
Positronic's components being wrapped into its:

| Foreign component | Runs as | Shim into our API |
|---|---|---|
| Simulator (LIBERO, Isaac Lab / RoboLab, MuJoCo) | env server in its own interpreter, behind the `env_server` wire | client-side `EnvAdapter` mapping the canonical embodiment contract ↔ the sim's raw payloads |
| Model stack (LeRobot, GR00T, OpenPI) | inference server behind the WebSocket wire | vendor `Codec` translating raw observations ↔ model I/O |
| Scenes / task batteries | instantiated inside the env server | reset tokens (suite, task, seed) carried through the `EnvAdapter` |
| Scoring / success criteria | computed where the ground truth lives (usually the env server) | reported alongside observations and recorded into the dataset; aggregation happens on the Positronic side |
| Hardware embodiment | pimm drivers inside the world | the same canonical embodiment contract the sims speak |

The corollary for frameworks that ship their own eval harness: when a third-party benchmark expects
a policy object plugged into *its* loop, the integration still separates that framework's sim/task
layer and serves it behind the env wire. Handing a Positronic policy to a foreign loop forfeits
both invariants — the run produces no dataset, and execution is scheduled by code we don't control.
