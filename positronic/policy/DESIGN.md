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
- **Embodiment-agnostic.** The API assumes nothing about the robot: what a
  policy must know about its body reaches it as data, so a new robot is new
  data, not a new API.
- **Interchangeable.** To a policy, simulation and the real robot are the
  same world: the same code runs in both, behaves the same, and pays for its
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

## Design

### Policies and sessions

Controlling a robot is stateful work, and one algorithm may drive several
robots at once.

- A `Policy` is the algorithm that controls a robot from observed data.
- Control happens in a `Session`; a `Policy` makes them.
- Sessions are independent, and several may exist at the same moment.
- A `Policy` is told which robot it is to control, so the session comes ready
  for it.
- The framework may cancel a session at any moment; a session never ends
  itself.

### Control

The robot moves continuously, but code acts in moments. The framework bridges
the two with signals, a concept it takes from
[pimm](../../pimm/README.md), Positronic's runtime. A
signal is a single value that its owner updates at its own rate. A reader
takes the latest value whenever it looks. Nothing queues and nothing waits.

The policy session lives in this world as a reader of observations and a
writer of commands. Everything around it is asynchronous, but the session
itself is called synchronously — a plain function call, repeated:

- One call: `(observations, time) -> (commands, resume_at)` — the current
  observations and time in, the commands to execute now (possibly none) out.
- Observations are the freshest the framework has at the moment of the call.
- Returned commands are emitted towards the robot driver immediately.
- Observations and commands can be of any type, structured or unstructured (a
  robot command, an image).
- The `time` argument is the only clock a session has; the framework sets it.
- `resume_at` is how the session paces itself: the instant at which it wants
  to be called next, absolute on the clock of `time`, strictly in the future.
- The framework calls best-effort at `resume_at`: it may be earlier or later,
  and the session reads the actual moment from `time`.

### Served functions

The remote-native goal splits a policy in two: heavy computation on its own
machine, control close to the robot. Served functions are that split in the
API. The policy defines its heavy work as functions, and the framework runs
every call. The session stays plain synchronous code, and control keeps
running while a call is in flight. The same function may run in-process
today and on a GPU server tomorrow: placement is a deployment choice, and
the plumbing — serialization, the wire — is the framework's job. This also
allows the framework to charge the call to the right clock in simulation and
to log it for the record.

- The session cannot tell where a function runs.
- Between calls the framework promises nothing: not the same process, not
  surviving state, not order, not exactly-once delivery. A function answers
  from its arguments alone; a cache is a speed-up, never a dependency.
- Invoking a function starts the work and returns a handle at once, never
  waiting. The session reads the handle when it next has control.
- A function's inputs and outputs are types the framework can serialize:
  plain types and numpy, selected domain classes. An unsupported type fails
  at the call.
- A session computes only inside its own call or inside a served function.

### Composability

A neural policy is never just the model. Around it sit data transforms —
normalize, change frames, encode actions — the pre- and post-processing
every ML pipeline has. Physical AI adds a second kind of part, one that
works in time: decide when to call the model, execute the chunk it
returned, blend a late plan into the motion underway. The API gives each
kind its own shape. A `Codec` transforms data and knows nothing of time. A
`Layer` wraps a session and lives on the same clock it does.

- A `Layer` is a recipe fixed at configuration time. When a policy session
  is created, each layer makes its session, wrapping the one inside it.
- A chain of layers is a layer, and its order fully determines behavior.
- Each session in the chain communicates only through the observations and
  commands flowing through it. It knows nothing of its neighbours or its
  position.
- What an inner session sees is its outer's choice — except `time`. The
  framework sets `time` once per outermost call, and every session in the
  chain sees that same value.
- A `Codec` is a pair of transforms — encode and decode, as in a video
  codec. Around a served function, encode converts the arguments and decode
  converts the answer. As a trivial layer, encode converts the observations
  going down and decode converts the commands coming up, with `resume_at`
  untouched.

Layers and codecs are offered, not imposed: a policy may always implement
its session directly against the call itself.

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
