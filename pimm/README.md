# Pimm: Immediate-Mode Middleware for Robotics

Pimm (Positronic IMMediate) is a tiny runtime that lets you describe a robotics
application as a handful of control loops, wire their inputs and outputs, and let
an orchestrator keep everybody up to date. It borrows the *immediate-mode* mindset
from GUI frameworks such as [ImGui](https://github.com/ocornut/imgui)/[egui](https://github.com/emilk/egui):
every loop reads the freshest data, computes the next command, emits it, and yields
right away—no hidden graph, no long-lived callbacks.

What makes this valuable for ML-oriented robotics is that you stay in plain
Python. There is no special DSL, no ROS launch files, and you can unit-test each
piece exactly like any other Python class.

## First Contact: Two Loops and a Wire

Below is the smallest useful Pimm program. A sensor publishes a sine wave based on
system time, and a logger prints whatever comes in. Notice how we only describe
*what* each system does; the `World` takes care of choosing transports and keeping
both loops alive (with the sensor happily running in a background process).

```python
import math
import time

import pimm


class SineSensor(pimm.ControlSystem):
    def __init__(self, hz: float = 0.5):
        self._hz = hz
        self.signal = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        while not should_stop.value:
            now = clock.now()            # monotonic seconds supplied by the World
            sample = math.sin(2 * math.pi * self._hz * now)
            self.signal.emit(sample, ts=clock.now_ns())
            yield pimm.Sleep(0.02)


class ConsoleLogger(pimm.ControlSystem):
    def __init__(self):
        self.source = pimm.ControlSystemReceiver(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        while not should_stop.value:
            try:
                value = self.source.value
                print(f"{clock.now():.2f}s -> {value:+.3f}")
            except pimm.NoValueException:
                pass
            yield pimm.Sleep(0.1)


with pimm.World() as world:
    sensor = SineSensor()
    logger = ConsoleLogger()

    world.connect(sensor.signal, logger.source)

    for sleep in world.start(logger, background=sensor):
        time.sleep(sleep.seconds)
```

Run it and you will see a stream of timestamps and sine values. A few important
things happened under the hood:

- `world.connect(...)` recorded that the sensor feeds the logger. Later, `world.start`
  noticed they run in different processes (the sensor goes to the background) and
  automatically chose a multiprocessing queue for that connection.
- The loop returned by `world.start(...)` schedules the logger in the main process
  and forwards `Sleep` values so you can decide how to wait (here we simply
  `time.sleep`).
- When the with-block ends—or the user kills the program—the world notifies every
  loop through the `should_stop` signal so they can exit gracefully.

That is the whole workflow: build small `ControlSystem` classes, `connect` their
signals, and `start` whichever group you want to supervise directly. Everything
else is plumbing you no longer have to write.

## Signals: One Pipe per Idea

Signals are the heart of Pimm. Conceptually, a signal is a one-way pipe that
carries mutually exclusive information. When a message travels down the pipe it
replaces whatever was there before, so a reader always sees the newest payload—even
if some messages were dropped along the way.

This leads to a simple rule of thumb:

> **If two pieces of data cannot both be acted on, they belong to the same signal.**

Examples:

- A robot controller that can accept *either* joint-space *or* Cartesian commands
  should publish them on a single signal, because executing one invalidates the
  other.
- A camera driver can bundle the latest image frame, camera intrinsics, or error
  state into one message. The logger only needs the freshest view of that stream;
  history is irrelevant.
- WebXR inputs (poses + button presses) form a signal because you only care about
  the latest controller state.

Because the newest value overwrites older ones, Pimm keeps signals lightweight and
non-blocking. You can treat them as "push the latest observation" rather than
"append to a queue". When you *do* need historical data (e.g., recording datasets)
that logic lives in a dedicated control system such as `positronic.dataset.ds_writer_agent`.

For common patterns Pimm provides small adapters:

- `pimm.ValueUpdated(reader)` pairs each value with a boolean indicating if the
  timestamp changed since the last read.
- `pimm.DefaultReceiver(reader, default)` supplies safe defaults until real data
  arrives.
- `pimm.map(signal, func)` transforms data on the way in or out without creating
  extra glue code.

## Control Systems and the World

A control system is any class that subclasses `pimm.ControlSystem` and implements
`run(should_stop, clock) -> Iterator[pimm.Sleep]`. Inside that method you are free
to use regular Python. Emitters (`ControlSystemEmitter`) and receivers
(`ControlSystemReceiver`) are just fields on the object; they keep track of their
owner so the world can wire them correctly.

The `World` runtime plays three roles:

1. **Connection planner.** Each call to `world.connect(emitter, receiver, ...)`
   records a link. When you later call `world.start`, the world consults those
   links, decides which ones stay local, which ones must cross processes, and
   binds the underlying transports accordingly. Optional `emitter_wrapper` /
   `receiver_wrapper` hooks let you decorate signals (rate limiting, logging,
   transforms) at bind time.
2. **Scheduler.** `world.start(main, background=None)` starts the chosen control
   systems, spawns background ones in daemon processes, and returns an iterator of
   `Sleep` commands for the main-process group. If you prefer a convenience wrapper
   you can call `world.run(*main_loops)` instead.
3. **Lifecycle manager.** Entering the `with pimm.World()` block creates shared
   resources; exiting it cleans up queues, shared-memory buffers, and background
   processes even if exceptions occur.

### Mirroring Connectors

Once you have declared your wiring it is often handy to keep a handle to the
"other side" of that connection in the supervising process. `World.mirror` makes
this trivial: pass it either a `ControlSystemEmitter` or `ControlSystemReceiver`
and it fabricates the matching endpoint with the same owning control system. The
world then immediately wires the pair using the same transport selection logic as
`connect`, so the mirrored endpoint stays in sync with its peer even if the peer
lives in a background process.

This lets the main process nudge or observe background systems after
`world.start(...)` begins yielding sleeps. For example, in
[`positronic/run_inference.py`](../positronic/run_inference.py) we mirror the
dataset writer's command receiver:

```python
commands = world.mirror(ds_agent.command)
commands.emit(DsWriterCommand(type=DsWriterCommandType.START_EPISODE))
for sleep in world.start(inference, bg_cs):
    time.sleep(sleep.seconds)
```

Because the mirrored emitter shares ownership with the dataset agent, the world
chooses the appropriate pipe automatically. You can use the same trick to mirror
state receivers for quick debugging dashboards or to expose one-off control knobs
without writing bespoke plumbing.

Control systems deliberately use generators rather than Python `async` because it
keeps your code uncoloured. You can call a loop from tests using a simple `for`
loop, or plug it into different schedulers without rewriting it.

## Moving Large Blobs

Some signals (camera frames, robot state vectors) require a large bandwidth and/or
small latency, and hence benefit from communication channels that avoid serialisation.
`world.mp_pipe(transport=pimm.world.TransportMode.SHARED_MEMORY)` returns an emitter/receiver pair backed by shared memory. Payloads implement the `SMCompliant` protocol—
`pimm.shared_memory.NumpySMAdapter` handles numpy arrays—so emitters know how big
the buffer should be and receivers can view the data without copying. If your data
implements `SMCompliant` protocol, the shared memory communication will be used
automatcially without your explicit request.

## Learn from Real Pipelines

- [`positronic/data_collection.py`](../positronic/data_collection.py) wires WebXR
  input, robot drivers, cameras, a dataset logger, and a DearPyGui dashboard. It
  showcases extensive use of `world.connect`, shared memory, and background loops.
- [`positronic/simulator/mujoco/sim.py`](../positronic/simulator/mujoco/sim.py)
  exposes MuJoCo as a `ControlSystem` that also serves as the global clock, so the
  simulator drives every other loop deterministically.
- [`positronic/tests/`](../positronic/tests) contains compact fixtures that mock
  clocks, pipes, and loops for unit testing.

Pimm is alpha software and continues to evolve. The next milestone is decoupling
connection planning from scheduling so you can describe the signal graph once and
choose execution placement later. Feedback and pull requests are very welcome!

## Where Pimm Fits

**ROS 2 (Python).** ROS 2 behaves like a robot operating system: it layers
distributed publish/subscribe, service discovery, QoS controls, and IDL tooling on
top of DDS. The trade-off is more moving parts and configuration. Pimm keeps the
graph in your Python code: you describe control systems, call `world.connect`, and
let the runtime decide whether a link is a queue or a subprocess pipe. That keeps
the experience developer-friendly while still being extensible enough to drive
production hardware (see [`positronic/drivers/`](../positronic/drivers/)).

**dora-rs.** [Dora](https://dora-rs.ai/) organizes dataflow graphs through declarative
manifests. An external agent launches executors—potentially written in other
languages—to route events. Pimm takes the opposite approach: you wire everything
imperatively inside Python, the world wires transports on the fly, and there is no
separate control plane to manage. It suits teams who prefer to keep orchestration
alongside the code they ship.

**Gym environments.** OpenAI Gym / Gymnasium focus on simulator loops (`reset`,
`step`) for reinforcement learning. They stop at the environment boundary: no
guidance for wiring actual cameras, controllers, or dataset writers. Pimm acts as
the runtime that lets those learned policies coexist with real sensors, robot
drivers, and GUIs in one control loop.
