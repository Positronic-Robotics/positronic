# Positronic Policy API

This document walks through the Positronic's policy API design, including the
reasoning that shaped it and the API itself.

## Introduction

Physical AI puts a robot under control of a model, end to end: what the robot
senses flows in, what it should do flows out. Different sensors do have
different frequencies, data may lag or come unevenly, and commands are not
executed instantaneously. Moreover, there are plenty of different "species" of
robots, and they have different bodies, different sensors, different command
languages, different control strategies.

These constraints make the API harder to design than the ones digital AI
settled on. A completions API is synchronous: the client asks, waits, and the
answer ends the exchange. A robot is inherently asynchronous: the world keeps
moving while the model thinks, so an answer arrives to a world that has
already changed. The model is heavy and usually runs on another machine, while
the robot must be controlled here and now. And a physical episode cannot be
re-run: understanding what happened relies on what was recorded, across every
machine involved.

Simulators are how policies are tested before they reach a robot, and a
simulator is a much more structured world than a rig: synchronous,
deterministic, time under the program's control. A policy that can sense this
structure comes to depend on it and loses it on the robot. So the API must not
reveal which world it runs against, and the framework must be able to charge a
model call's real duration to simulated time — otherwise inference is free in
simulation and costly on the robot.

The rest of the document is the design: the goals, the requirements, and the
API itself.

## Goals

- **Expressive.** A model call is slow while the robot keeps moving, and how
  a policy bridges that — executing the previous chunk, blending a late plan
  into the motion underway, choosing when to re-plan — is where policies
  differ most. New schemes appear constantly and each must fit without a
  framework change.
- **Embodiment-agnostic.** The API assumes nothing about the robot; what a
  policy must know about its body reaches it as data, so a new robot is new
  data, not a new API.
- **Interchangeable.** To a policy, simulation and the real robot are the
  same world: the same code runs in both, behaves the same, and pays for its
  model calls in both.

## Requirements

### Policies and sessions

- A `Policy` is the algorithm that controls a robot from observed data.
- Control happens in a `Session`; a `Policy` makes them.
- Sessions are independent, and several may exist at the same moment.
- A `Policy` is told which robot it is to control, so the session comes ready
  for it.
- The framework may cancel a session at any moment; a session never ends
  itself.

### Control

- Control happens in a cycle between a session and the framework; the
  session's part of the cycle is a turn.
- One turn: `(observations, time) -> (commands, resume_at)` — the current
  observations and time in, the commands to execute now (possibly none) out.
- The `time` argument is the only clock a session has; the framework sets it.
- Observations and commands can be of any type, structured or unstructured (a
  robot command, an image).
- Observations are the freshest the framework has when the turn starts.
- Returned commands are emitted towards the robot driver in the same turn.
- `resume_at` is when the session wants its next turn: an absolute instant on
  the clock of `time`, strictly in the future.
- The framework grants the turn best-effort at `resume_at`: it may be earlier
  or later.
- The framework has no pace of its own: turns happen when sessions ask.

### Served functions

- A policy's heavy work (model inference) lives in functions it defines and
  the framework serves — in-process or on a server; the session cannot tell.
- Invoking one starts the work off the turn and returns a handle at once.
- The session may read the handle when it next has control.
- A function's inputs and outputs are types the framework can serialize:
  plain types and numpy, selected domain classes. An unsupported type fails
  at the call.
- A session computes only while in control or inside a served call.
- Served functions are pure: one may cache, but dropping the cache must not
  change what it computes.

### Composability

- A `Layer` is a config-time recipe: per control session it makes a session
  wrapping the inner session it is given.
- A chain of layers is a layer.
- Chain order fully determines behavior.
- A session communicates only through the observations and commands flowing
  through it; it knows nothing of its neighbours or its position.
- What an inner session sees is its outer's choice — except `time`: one value
  per turn, the same at every depth, not the chain's to alter.
- A `Codec` is a data transform written once and applicable to both: around a
  served function, its inputs and outputs; as a trivial layer, observations
  down and commands up, `resume_at` untouched.
- Any policy fits the API as it stands: implement a session directly, or
  compose it from layers and served functions. Neither ever requires a
  framework change.

### Logging

- A log is a set of named series per control session; a sample is a value
  stamped on 3 timelines: the turn number, control time, and wall time.
- If requested, the framework logs the timelines' correspondence: one sample
  per turn.
- If requested, the framework logs every function call: its identity, submit
  time, and answer time.
- If requested, the framework logs every session's invocations in a chain it
  assembled.
- A session may append a value to a named series of its own; the framework
  stamps it.
- A session names its series locally; the framework keeps the full names
  distinct across sessions and stable across runs.
- A function may attach records to the call it is answering, stored where the
  function runs and joined to the log by call identity.
- Logging never affects control: it adds no waiting and no failure path to
  the loop.

### Remote policies

- The framework natively supports remote policies: the whole policy lives on
  a server.
- The server describes the full policy: the functions it serves and the stack
  around them.
- One URL is enough to fully specify the policy, given that the server
  returns a proper declaration.
- Declarations are backward compatible: the framework evolves without
  changing what an already-served policy does.
- A declaration the rig cannot honor is refused at the handshake.

## TODO

The Python shape of the turn, the function handles, and the logging conduit —
an object protocol, with generators as a supported authoring style the
framework adapts.

## Deferred, not to decide now

- The shape of the robot description, and a server's ability to refuse one.
- The protocol delivering the declared stack to the rig — versioning and
  compatibility.
