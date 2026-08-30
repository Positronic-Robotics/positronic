# Positronic Policy API

This document walks through the Positronic's policy API design, including the
reasoning that shaped it and the API itself.

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

## Design

### The life of an episode

The code on the rig is given one URL, and that is all it knows about the
policy. It connects and receives a description of the policy's pieces —
which run near the robot and which stay remote. It assembles the local
half: a chain of parts that transform observations on the way to the
model and commands on the way back.

An episode begins, and the assembled half becomes a session — the
running instance of the policy that controls this robot for this
episode. The framework calls the session repeatedly, sending the sensor
data and the current world time. Every call returns commands and the
time of the next call. The framework sends the commands to the robot
immediately.

The session holds everything the episode remembers between calls. This
state lives on the rig: a reaction that crosses a wire arrives late,
and when the network fails it does not arrive at all.

The model is heavy: it needs a GPU and a machine of its own, and the rig
rarely has them. Heavy pieces stay remote. The model is also too slow for
this loop, so the session never waits for it. The session starts the
model call and continues to control the robot. The session acts on the
answer when it arrives. The server only answers these calls. It is
stateless: every call is a pure function of its arguments.

The framework records everything that crosses a boundary, as it
happens: what the session saw, what it decided, and what it asked of
the model, on every machine involved. The episode ends when the
framework closes the session. The record remains.

The sections below describe each piece.

### Policies and sessions

One algorithm may drive several robots at once.

- A `Policy` is the algorithm that controls a robot from observed data.
- Control happens in a `Session`; a `Policy` makes them.
- Sessions are independent, and several may exist at the same moment.
- A `Policy` is told which robot it is to control, so the session comes ready
  for it.
- The framework may cancel a session at any moment; a session never ends
  itself.

### Control

The robot moves continuously, but code acts in moments. The framework
connects the two with signals, a concept from
[pimm](../../pimm/README.md), Positronic's runtime. A signal is a single
value that its owner updates at its own rate. A reader takes the latest
value whenever it looks. Nothing queues and nothing waits.

The session is a reader of observations and a writer of commands.
Everything around the session is asynchronous, but the framework calls it
synchronously — a plain function call, repeated:

- One call: `(observations, time) -> (commands, resume_at)` — the current
  observations and time in, the commands to execute now (possibly none) out.
- Observations are the freshest the framework has at the moment of the call.
- Returned commands are emitted towards the robot driver immediately.
- Observations and commands are named channels. A channel value can be
  of any type, structured or unstructured (a robot command, an image).
- The `time` argument is the only clock a session has. The framework sets
  it, and it strictly grows from call to call.
- `resume_at` is how the session paces itself: the instant at which it wants
  to be called next, absolute on the clock of `time`, strictly in the future.
- The framework calls best-effort at `resume_at`: it may be earlier or later,
  and the session reads the actual moment from `time`.

### Served functions

The policy defines its heavy work as stateless functions,
`infer(obs) -> actions`, and gives them to the framework. The session
starts a call and does not wait — control continues while the framework
runs the function, in process or on a GPU server. The framework runs
every call, so it can charge the call to the right clock in simulation
and record it.

- The session cannot tell where a function runs.
- Between calls the framework promises nothing: not the same process, not
  surviving state, not order, not exactly-once delivery. A function answers
  from its arguments alone; a cache is a speed-up, never a dependency.
- Invoking a function starts the work and returns a handle at once, never
  waiting. The session reads the handle when it next has control.
- A failure the framework can see — a lost connection, a dead worker, a
  value that does not serialize — ends the handle with an error. Only a
  function that does not return leaves a handle open.
- A session may cancel a call it will not read. The framework stops the
  work where it can: it stops retries and drops the queued call. A call
  that already runs may run to its end. Functions are pure, so a dropped
  call loses nothing.
- A function's inputs and outputs are types the framework can serialize:
  plain types and numpy, selected domain classes. An unsupported type fails
  at the call.
- A session computes only inside its own call or inside a served function.

### Composability

A neural policy is never just the model. Data transforms surround it —
normalize, change frames, encode actions — the pre- and post-processing
every ML pipeline has. Physical AI adds a second kind of part, one that
works in time: decide when to call the model, execute the actions it
returned, blend a late plan into the motion underway. The API gives each
kind its own shape. A `Codec` transforms data and does not see time. A
`Layer` wraps a session and runs on the session's clock.

- A `Layer` is a recipe fixed at configuration time. When a policy session
  is created, each layer makes its session, wrapping the one inside it.
- A chain of layers is a layer, and its order fully determines behavior.
- Each session in the chain communicates only through the observations and
  commands flowing through it. It knows nothing of its neighbours or its
  position.
- What an inner session sees is its outer's choice — except `time`. The
  framework sets `time` once per outermost call, and every session in the
  chain sees that same value.
- A layer calls its inner session at most once per outer call. A second
  call would repeat the same `time`. Sessions depend on strict growth: a
  session that divides by its time step must not get a zero step.
- A `Codec` is a pair of transforms — encode and decode, as in a video
  codec. Around a served function, encode converts the arguments and decode
  converts the answer. As a trivial layer, encode converts the observations
  going down and decode converts the commands coming up, with `resume_at`
  untouched.

Layers and codecs are offered, not imposed: a policy may always implement
its session directly against the call itself.

### Remote policies

Execution splits between the rig and the server. The definition does not:
the layers and codecs on the rig belong to the same design as the functions
behind the wire, and halves defined apart drift apart. The server owns the
whole definition and sends it to the rig as a description.

- One URL is enough to use a policy: the description carries everything the
  rig needs to assemble its half.
- Descriptions are backward compatible: the framework evolves without
  changing what an already-served policy does (as much as possible).
- A description the rig cannot honor is refused at the handshake.

### Recording

A system split between a rig and a server is hard to debug. Data flows
in two dimensions — through time and through the layers — and the
recording reconstructs both flows after the episode.

Each session produces one recording on the rig: a set of named series, where
a series holds values of one type — arrays, images, numbers. A recording is
built for a visualizer like [rerun.io](https://rerun.io). Every value is
placed on three timelines: the call number, control time, and wall time.

- If requested, the framework records the flow at every boundary it carries:
  what enters and leaves each session in a chain it assembled and each
  served function, and when.
- A session may append values to named series of its own. The framework
  places each value on the timelines.
- A session names its series locally. The framework keeps the full names
  distinct across sessions and stable across runs.
- Served functions record too: into the recording itself, or into storage of
  their own, joined later.
- Recording never affects control: it adds no waiting and no failure path.

## API

The design above, as Python protocols.

### The session call

```python
# Observations and commands are named channels.
Obs = Mapping[str, Any]
Commands = Mapping[str, Any]


class Session(Protocol):
    # `time` and the returned `resume_at` are seconds on the same clock.
    def __call__(self, obs: Obs, time: float) -> tuple[Commands, float]: ...

    # Called by the framework, at any moment. There is no other end.
    def close(self) -> None: ...
```

### Served functions

```python
# This is pimm's `Answer`.
class Answer(Protocol):
    def done(self) -> bool: ...

    # The result once done, re-raising what the function raised. Reading it
    # earlier is an error: a session never waits.
    def result(self) -> Any: ...

    # The session will not read this answer. The framework stops the work
    # where it can. Never waits.
    def cancel(self) -> None: ...


# Calling one starts the work and returns the `Answer` at once.
Fn = Callable[..., Answer]
```

### The runtime

```python
# The framework's standing offer to one session. Every session gets its own.
class Runtime(Protocol):
    # The served functions, each wrapped into an `Fn`: a worker pool in
    # process, a stub over the wire.
    @property
    def fns(self) -> Mapping[str, Fn]: ...

    # Append `value` to this session's series `name`. The framework places
    # it on the timelines.
    def record(self, name: str, value: Any) -> None: ...
```

### Policies

```python
class Policy(Protocol):
    # The policy's heavy work: plain callables, given by name, served
    # back as `rt.fns`.
    @property
    def functions(self) -> Mapping[str, Callable]: ...

    # `context` carries the robot the session will control, and whatever
    # else the framework knows about the episode.
    def new_session(self, context: dict[str, Any], rt: Runtime) -> Session: ...
```

### Layers and codecs

```python
# The framework's handle to the next session in. No `time` parameter: the
# framework stamps the inner call with the outermost call's time, so the
# whole chain sees one `time`.
Inner = Callable[[Obs], tuple[Commands, float]]


class Layer(Protocol):
    def make_session(self, inner: Inner, rt: Runtime) -> Session: ...


class Codec(Protocol):
    def encode(self, value: Any) -> Any: ...

    # `context` is the input of the matching `encode` — a decode may need
    # it (relative commands need the pose the observation carried).
    def decode(self, value: Any, context: Any) -> Any: ...
```

`layer_a | layer_b` is a layer. `chain.wrap(policy)` is a policy: its
sessions are the chained sessions, with a handle between each pair. The
framework makes every session in the chain and gives each one its own
`Runtime`. The framework closes every session it made, and a session
closes what it made itself. `close` never travels through the chain.

## Deferred, not to decide now

- The shape of the robot description, and a server's ability to refuse one.
- The protocol delivering the description to the rig — versioning and
  compatibility.
- Source times for observations — whether the framework passes the
  timestamp of each sensor value to the session. The pimm signals
  already carry these timestamps.

## TODO

Open review comments from #652.

- [x] Promise that every served call ends. A dropped request or a dead
  worker must not leave a handle that never completes. The handle
  completes with the result or with an error. A transport failure is an
  error.
- [x] Keep the strict growth of `time`. A layer calls its inner session
  at most once per outer call. A second call would repeat the same
  `time` and break a session that divides by its time step.
- [x] Decide if the framework stamps each observation with its source
  time. Moved to the Deferred section: the question is whether the
  framework provides this data at all.
- [x] Answer the comment on one `Runtime` for a chain. The framework
  makes every session in the chain and gives each one its own `Runtime`,
  derived inside the framework. The public API does not change.
- [x] Correct the bullet that says observations can be of any type. The
  value of each channel can be of any type. The observation itself is a
  mapping of named channels.
- [ ] Add the wall-time authority to the recording decisions. Records
  from two machines share one wall-time timeline only if the clocks
  agree. This belongs with the deferred recording mechanics and #528.
- [ ] Define how a remote error crosses the wire. A custom exception
  needs its class on the rig to be re-raised. A framework-owned error
  type is the usual answer. This waits for the wire work.
- [ ] State that `record` copies the value before it returns. A session
  can change a mutable array after the call, and a late recorder would
  then store the changed value.
- [ ] Correct the story sentence about the server. The server also sends
  the description at the setup. Write: after the setup, the server only
  answers served calls.
- [ ] Answer the comment that asks for `time` in `Inner`. The design
  omits the argument so a layer cannot change the time. The framework
  stamps it, and the comment above `Inner` says so.
