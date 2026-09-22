# Positronic Policy API

This document walks through the Positronic's policy API design, and its execution and ownership contracts.

## Introduction

AI changed how software is built. Hand-written code gave way to
general-purpose models. An agentic system has three parts — a model that
decides, tools that act, and a harness that connects the two. Both
interfaces are standardized. The harness reaches any model through the same
completions API, and calls any tool the same way, whether the tool is a
shell command, an MCP server, or a third-party service. The standards make
the parts interchangeable. Any model works with any tools, and swapping one
never touches the other.

Robotics has no such architecture. AI already makes decisions on robots, but
each stack binds one model to one robot with bespoke code — the way software
was bound to hardware before operating systems decoupled them. A model
cannot move to a new robot, and a robot cannot pick up a better model.

In Positronic Robotics our goal is to let any AI model control any robot, in
simulation and in reality. The person with a problem picks the model and the
robot, and the two connect. The two "any"s rule out hand-wiring — nobody
writes a control loop per model per robot per world. The goal requires a
single interface between models and robots, and this document designs it.

This interface is harder to design than the ones digital AI settled on. A
completions API is synchronous: the client asks, waits, and the answer ends
the exchange. A robot is inherently asynchronous: the world keeps moving
while the model thinks, so an answer arrives to a world that has already
changed. Sensors run at their own rates, data lags or arrives unevenly, and
a command takes time to execute. Robots themselves are diverse, with
different bodies, different sensors, different command languages, different
control strategies. The model is heavy and usually runs on another machine,
while the robot must be controlled here and now. And a physical episode
cannot be re-run: understanding what happened relies on what was recorded,
across every machine involved.

Simulation is hard for the opposite reason. The rig's world is messy, and
the simulator's world is too tidy — synchronous, deterministic, time under
the program's control. Policies are tested in simulation before they reach
a robot, and a policy that can sense this tidiness comes to depend on it
and loses it on the robot. So the API must not reveal which world it runs
against, and the framework must be able to charge a model call's real
duration to simulated time.

The rest of the document is the design: the goals, where existing
interfaces fall short, the design decisions, and the API itself.

## Goals

- **Expressive.** A model call is slow while the robot keeps moving, and how
  a policy bridges that — executing the previous chunk, blending a late plan
  into the motion underway, choosing when to re-plan — is where policies
  differ most. New schemes appear constantly and each must fit without a
  framework change.
- **Any robot.** The API assumes nothing about the robot: what a policy
  must know about its body reaches it as data, so a new robot is new data,
  not a new API.
- **Any world.** To a policy, simulation and the real robot are the same
  world: the same code runs in both, behaves the same, and pays for its
  model calls in both.
- **Composable.** Building a new policy must be easy: a policy is assembled
  from parts written once, so a developer writes only what is new.
- **Remote-native.** Heavy computation wants its own machine, while control
  runs best close to the sensors and the robot. A policy therefore spans
  machines, and the API and the framework must make that split easy.
- **Debuggable.** An episode cannot be re-run, so understanding it relies on
  what was recorded. From the records alone one can reconstruct what the
  policy saw, what it decided, and what it asked of its model — even when
  those happened on different machines.

## Existing interfaces

Robot policies are already served over the wire. Physical Intelligence's
openpi server (also behind RoboArena), NVIDIA's GR00T inference service,
LeRobot's serving stack and academic platforms like XPolicyLab all share
the same shape - the client sends observations, the server returns a chunk
of actions, and the client runs the loop. This shape traces back to
the gym environment loop — `action = policy(obs); obs = env.step(action)` —
where the world truly waits while the policy thinks. A robot's world keeps
moving, and the interface cannot express what that demands:

- Act while thinking. Between request and reply the policy cannot see or
  do anything. A policy that watches the force sensor and freezes the arm
  while its model computes cannot be written.
- Choose the next moment. The client asks on a schedule of its own: every
  k steps, or when the action queue drains. A policy that wants to re-plan
  early because the object slipped has no way to ask for that.
- Know the time. Chunks are timestamped by presuming a fixed control
  period, latency is measured and then only logged, and in simulation
  thinking is free. The policy never learns how stale its observations are
  or when its answers take effect.

LeRobot's hand-written inference thread, openpi's blocking chunk client and
hand-tuned replan constants are all patches around these limits, written
again at every robot.

This design starts from the moving world instead. A policy acts, watches
and paces itself in it, so the list of expressible schemes has no end: the
next idea fits without a new interface.

## Processors and runs

A `Processor[InputT, OutputT]` is a reusable definition. Its constructor holds
configuration; `run(runtime, *dependencies)` returns one generator with its
own state. `Policy` is an alias for `Processor[Obs, Step]`.

The runtime starts and primes each generator. Its first yield must be `None`;
after that, `send(input)` returns an output. The aliases `ProcessorRun[InputT,
OutputT]` and `PolicyRun` describe these generators. State belongs in generator
locals, so starting the same definition again creates a fresh episode.

```python
from positronic.policy.base import Policy, PolicyRun, Runtime, Step


class Hold(Policy):
    def run(self, runtime: Runtime) -> PolicyRun:
        obs = yield
        while True:
            obs = yield Step({}, runtime.time_ns + 100_000_000)
```

The harness receives a complete policy definition. It creates one runtime and
starts one run per episode. Dependencies are resolved by the definition;
`Runtime.start` also accepts dependencies for constructing child runs.
Whoever starts a run closes it. Submitted work must finish before runs close
resources that work may still use.
The harness closes its executor, draining workers, then closes the policy run.
Generator closure happens between resumptions, never concurrently with a send.

## Control and time

A policy receives the latest available sensor values, task, and rig descriptor.
The harness refreshes each signal's serialized fields when its `updated` flag
is set, copying arrays so later device writes cannot change an earlier input.
Observation mappings are read-only. Time comes from `runtime.time_ns`; the
observation contains neither `obs_time_ns` nor `wall_time_ns`.

A `Step` contains commands to emit immediately and an absolute `resume_at_ns`
on the runtime's clock. The harness clamps the interval from the policy call's
start to 5 ms–1 s. When no observation is available, a real rig polls every
100 ms. Simulation yields to the other control systems and checks the policy's
deadline on simulator ticks.

Every newly available submitted answer can resume the entire policy stack
before its requested time. Multiple resumptions can have the same time and
`runtime.tick`; code must not assume a positive elapsed time between calls.
The harness re-reads signals even then, so newly arrived values are visible.

`ChunkedSchedule(fps, horizon_sec=None)` submits a function returning an ordered
sequence of command mappings. It anchors the first command when it reads the
answer, emits due commands, and requests another chunk after the execution
window. `horizon_sec` caps that window. An overdue call combines due commands,
keeping the latest value per channel. The harness contains no trajectory player.

`StopOnFault` withholds commands and child calls while an arm is unavailable.
Calls resume on the same child run once the arms are available. `TemporalStack`
records selected channels and samples their history; place it outside the
scheduler to collect history during chunk execution.

## Submitted work

`runtime.submit(function, *args, **kwargs)` returns `Answer[T]` immediately.
The function is an ordinary callable, including a remote inference call.
Submitted functions must not mutate the processor run's episode state.
Submission passes arguments by reference: neither the run nor the worker may
mutate those inputs until the call finishes. Copy a reusable buffer before
submitting it. Handles are checked without waiting for the worker or network.

- `done()` checks whether the result is available on the episode clock.
- `result()` returns the function's result, re-raises its exception, or raises
  `NotAnswered` if it is too early. A remote error may have a different type.
- `cancel()` cancels queued work when possible. Running calls may finish.

The executor uses worker threads and receives a clock function. It has no
control-system signal dependencies. Real execution exposes finished answers
immediately; the harness polls pending work at intervals of at most 5 ms.
Charged simulation exposes answers only after simulated time includes their
queueing and execution duration. Uncharged simulation handles completion before
advancing time. Chains of uncharged calls at the same instant are unrestricted
and can prevent the simulator from advancing.

Shutdown is checked between episode iterations. Waiting for uncharged work can
delay shutdown; the wait deliberately does not poll the stop signal. Cleanup
stops at the first error, without guaranteeing closure of remaining resources.

## Composition and codecs

`Sequential(outer, middle, inner)` nests its components in that order. A processor
receives the live child run or callable as a dependency and controls whether and
how often to invoke it. A codec encodes inputs and decodes outputs; around a
`Step`, it transforms commands and preserves `resume_at_ns`.
Other processors may replace the child's requested time. An empty command
mapping emits nothing. `Codec.wrap` copies the input mapping to a dict before
encoding; it accepts one observation value, not a positional-argument envelope.
Context-dependent decoding belongs around the inference callable so it uses
that call's input; a codec around a scheduler sees the current control input.

Codecs and processors can mix anywhere in a sequence. Codecs also support `|`
for sequential data conversion and `&` for parallel conversion with merged
outputs. They have no clock and do not attach action timestamps. `Metadata`
adds declarations, such as training cadence, without transforming data.

`processor.meta()` reports definition metadata. `Sequential.meta()` flattens
and combines component metadata, with later components winning on shared keys.
`to_spec()` returns a registered name and plain-data constructor arguments;
compositions contain nested specs. A component without a supported wire spec
can still run locally, but cannot be delivered to a rig.

## Remote deployments

`RemotePolicy` takes a wire name and a session address. Each run opens a server
session, receives its ID and declared client stack, and runs that stack around
an ordinary remote inference function. The run ends the session when closed.
The transport connection also bounds its lifetime: disconnects release the
session after outstanding calls finish. A wrong session ID produces an error
and closes the requesting connection.
Credentials are supplied separately through headers, outside the address.

Server configuration lives in `positronic.offboard.spec`:

- `ModelSource` discovers checkpoint IDs and loads a `Model`.
- `Model(obs, session_id=...)` returns model outputs. Stateless models can ignore
  the ID. A stateful backend must isolate sessions or reject concurrent owners.
- `Model.end_session(id)` releases episode state. `Model.close()` releases
  loaded resources when the server switches models or shuts down.
- `PolicyDeployment(source, local, codec=None, compress_images=False)` assembles
  the model source, client processor stack, optional server codec, and transport
  compression setting.

The server returns full action chunks; client scheduling selects their execution
window. It serializes calls to a loaded model. Session parameters may change the
client stack or server codec, but cannot replace the model source. A metadata
probe creates and ends a session without claiming a stateful model's episode.

## Recording and logging

The harness records sensor and executed-command signals as an episode dataset.
Timing is part of logging: framework spans cover processor resumptions, codecs,
and submitted jobs, with parent-child links. Suspended generators keep no span
open; background jobs remain children of the call that submitted them. These
are wall-clock timings, independent of simulation charging. Parent durations
include synchronous children and should not be added to them.

The server reports per-request component durations even without telemetry files.
Inference input/output recording and per-run metadata are not implemented.

## Deferred

- TODO: Record dropped and late waypoints in the scheduling processor.
- TODO: Allow selecting which answers wake a policy early with `wake_on`.
- TODO: Define per-run metadata and inference input/output recording.
- TODO: Decide whether observations include each sensor's source timestamp.
- TODO: Align wall-clock logs across machines; #528 tracks the clock problem.
