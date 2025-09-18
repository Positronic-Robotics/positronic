# Pimm: Positronic Immediate Mode Middleware

Pimm (Positronic IMMediate) is Positronic's Python-native middleware for stitching
robotic systems out of small, composable control loops. It takes inspiration from
immediate-mode GUI frameworks such as Dear ImGui and egui: every loop reads the
current state, computes the next command, emits it, and yields control right away.
There is no hidden background graph and no global event bus – just explicit data
flow through signals and a deterministic scheduler.

Pimm powers the pipelines in [`positronic/`](../positronic), including the data
collection stack, robot drivers, MuJoCo simulator bindings, and inference loops.
This document explains the core mental model and how to put the library to work in
your own control systems.

## Why Immediate Mode for Robotics?

- **Freshest data wins.** Signals are single-writer/single-direction channels. They
  always expose the latest message, which mirrors how sensor streams, controller
  inputs, and robot commands behave in practice.
- **Deterministic scheduling.** Control loops are plain Python generators that
  yield `pimm.Sleep(seconds)` (or `pimm.Pass()` for zero delay). The orchestrator
  interleaves them with a predictable order, enabling reproducible simulator runs
  and easier debugging.
- **Python-first ergonomics.** No ROS graph description, no coloured async APIs.
  Everything stays as regular Python objects so you can unit test components and
  mix them with existing ML tooling.

## Core Concepts

### Signals

A *signal* is a unidirectional channel made of a `SignalEmitter[T]` and a
`SignalReceiver[T]` exchanging `Message[T]` objects (payload + integer timestamp).

- Emitters are non-blocking; if a downstream queue is full, the oldest value is
  dropped so the newest data always propagates.
- Receivers cache the last observed message. Calling `reader.read()` returns the
  newest `Message` or `None` if nothing was ever emitted. `reader.value` raises
  `NoValueException` until the first message arrives.
- Signals carry mutually exclusive pieces of information. If you can only act on
  the most recent command, it should share the same signal (e.g. joint-space and
  Cartesian commands for a robot arm).

Pimm ships several ready-to-use transports:

- `World.local_pipe()` – lock-free deque for components living in the same process.
- `World.mp_pipe()` – `multiprocessing.Queue` backed pipe for cross-process links.
- `World.shared_memory()` – zero-copy transport for large blobs when paired with an
  `SMCompliant` payload (see below).
- `BroadcastEmitter` – fan-out helper that keeps multiple emitters in sync, heavily
  used in `positronic.data_collection` to tee robot commands to both the robot and
  logging agents.

Utility wrappers make common patterns trivial:

```python
commands = pimm.ValueUpdated(robot.commands)          # detect "new" values
latest_or_idle = pimm.DefaultReceiver(commands, (None, False))
filtered = pimm.map(robot_state, my_transform)        # transform on read
```

### Control Loops

A control loop has the signature `loop(should_stop: SignalReceiver, clock: Clock)
-> Iterator[pimm.Sleep]` and drives one aspect of the system (driver, sensor,
inference, UI, etc.). Example:

```python
import pimm

class JoystickBridge:
    def __init__(self):
        self.pose: pimm.SignalEmitter[pimm.geom.Transform3D] = pimm.NoOpEmitter()

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        while not should_stop.value:
            pose = read_gamepad_transform()
            self.pose.emit(pose, ts=clock.now_ns())
            yield pimm.Sleep(0.01)
```

Pimm deliberately avoids `async` coroutines: generators keep call sites uncoloured
and make deterministic interleaving straightforward.

### World and Scheduling

`World` is the runtime that wires signals and executes loops:

```python
with pimm.World() as world:
    joystick = JoystickBridge()
    planner = Planner()

    # Wire channels
    joystick.pose, planner.ee_pose = world.mp_pipe()
    planner.commands, robot.commands = world.mp_pipe()

    # Run planner inline, driver in a subprocess
    world.start_in_subprocess(robot.run)
    world.run(joystick.run, planner.run)
```

Key behaviours:

- The context manager ensures background processes, shared memory blocks, and
  queues are cleaned up even on exceptions.
- `world.start_in_subprocess(loop)` forks a daemon process that drives a loop
  until it finishes or `world.should_stop` is set.
- `world.interleave(*loops)` exposes the deterministic scheduler if you want to
  embed it in your own main loop; `world.run` simply executes it and sleeps.
- All loops receive a `should_stop` signal so they can exit
  cooperatively when the world shuts down.

`Clock` instances supply timestamps. `World` defaults to `SystemClock` (monotonic
wall time), but any custom `Clock` works – for example, `positronic.simulator.mujoco.sim.MujocoSim`
implements `Clock` so simulator time drives the rest of the system.

### Shared Memory Payloads

For high-bandwidth data (robot state vectors, camera frames) Pimm relies on the
`SMCompliant` protocol. Implementations describe how to size, write, and read a
buffer. `pimm.shared_memory.NumpySMAdapter` covers numpy arrays:

```python
class MujocoFrankaState(State, pimm.shared_memory.NumpySMAdapter):
    def __init__(self):
        super().__init__(shape=(22,), dtype=np.float32)

    # encode() fills self.array with the latest simulator state

with pimm.World() as world:
    robot.state, ds_writer.inputs['robot_state'] = world.shared_memory()
```

Emitters ensure every payload matches the first message's shape and dtype. Readers
materialise lightweight views onto the shared block; re-emit after mutating data
so receivers observe the changes.

## Building Larger Systems

`positronic/data_collection.py` shows a representative assembly:

1. Controllers, simulators, dataset writers, and GUI widgets each expose a `run`
   generator.
2. A `World` instance wires signals with a mix of `mp_pipe`, `shared_memory`, and
   `BroadcastEmitter` so every participant receives the freshest inputs.
3. Top-level orchestration starts long-running drivers in subprocesses (arm,
   gripper, cameras) while interleaving coordination logic (`DataCollectionController`)
   in the main process.
4. Utilities like `ValueUpdated`, `DefaultReceiver`, and `RateLimiter` keep loops
   responsive without adding manual edge tracking.

The same pattern appears in `positronic/robot_controller.py`, which connects a
hardware driver to a CLI, and in the MuJoCo simulator bindings where the simulator
itself implements `Clock` so physics time controls the rest of the world.

## Testing Helpers

Pimm is designed to be unit-test-friendly:

- `pimm.tests.testing.MockClock` provides a deterministic, manually stepped clock.
- Signals are regular Python objects, so you can stub them with simple fakes or
  the `NoOpEmitter`/`NoOpReceiver` placeholders found throughout `positronic/`.
- `ValueUpdated` makes it easy to assert *when* new data appears, which is used
  heavily in the shared-memory tests.

Refer to `pimm/tests/` and `positronic/tests/` for practical patterns, including
full pipelines validated through the scheduler.

## Project Status and Roadmap

Pimm is currently in **alpha** and under active development. Near-term priorities
include decoupling signal graph construction from scheduling so control loops can
be declared without deciding where they execute. Expect APIs to stabilize over
time as we keep refining the ergonomics for ML-focused robotics stacks.

If you use the library outside the main Positronic repo, please share feedback.
