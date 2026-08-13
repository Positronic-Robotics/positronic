"""What a console observes of a camera, by where the world schedules it.

Composes the arrangement a run uses — two cameras and a `Harness`, wired through `wire_embodiment` —
with a console reading the camera signals, and runs it twice: once with the console foreground beside
the harness (`rollouts-console local`), once background in its own process (`rollouts-console real`).

    uv run --locked python utilities/frame_delivery_probe.py
    uv run --locked python utilities/frame_delivery_probe.py --stall-ms 300

Each frame carries its own sequence number in its pixels, so a frame the console never observed is a
gap in the sequence rather than an inference from a sampling window. Every frame is stamped with
`CLOCK_MONOTONIC`, which is system-wide, so the reported delivery latency subtracts the emit stamp
from the read and is what the transport cost, with the producer's start-up excluded.

`--stall-ms` blocks the harness for that long each round, which is what a synchronous inference call
does to the loop it runs in (`Harness._step` calls the policy session, then paces).
"""

import argparse
import statistics
import time
from dataclasses import dataclass, field

import numpy as np

import pimm
from positronic import keys, wire
from positronic.eval import Embodiment, Observation
from positronic.policy.harness import Harness

CAMERA_PERIOD_S = 1 / 30
CONSOLE_PERIOD_S = 1 / 60
# Long enough to outlast a spawned process importing positronic, which is what a foreground console
# waits on; a probe that stops sooner reports a delivery failure that is not there.
FIRST_FRAME_TIMEOUT_S = 30.0
FRAME_SHAPE = (48, 64, 3)


def _stamp(image: np.ndarray, seq: int) -> None:
    """Write `seq` into the frame's first pixels, so a reader can name the frame it is holding."""
    image[0, :4, 0] = np.frombuffer(np.uint32(seq).tobytes(), dtype=np.uint8)


def _seq_of(image: np.ndarray) -> int:
    return int(np.frombuffer(bytes(image[0, :4, 0]), dtype=np.uint32)[0])


class Camera(pimm.ControlSystem):
    """A camera: one numbered frame per round, through the shared-memory adapter a real driver uses."""

    def __init__(self):
        self.frames = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        adapter = None
        seq = 0
        while not should_stop.value:
            image = np.zeros(FRAME_SHAPE, dtype=np.uint8)
            _stamp(image, seq)
            adapter = pimm.shared_memory.NumpySMAdapter.lazy_init(image, adapter)
            self.frames.emit(adapter, clock.now_ns())
            seq += 1
            yield pimm.Sleep(CAMERA_PERIOD_S)


@dataclass
class CameraObservations:
    """What one console saw of one camera over the measurement window."""

    waited_s: float | None = None
    first_latency_s: float | None = None
    latencies_s: list[float] = field(default_factory=list)
    seqs: set[int] = field(default_factory=set)

    def observe(self, seq: int, latency_s: float, waited_s: float) -> None:
        if self.waited_s is None:
            self.waited_s = waited_s
            self.first_latency_s = latency_s
        self.latencies_s.append(latency_s)
        self.seqs.add(seq)

    @property
    def emitted(self) -> int:
        """Frames the camera produced from the first one observed onwards — the window's denominator."""
        return max(self.seqs) - min(self.seqs) + 1 if self.seqs else 0

    def report(self, name: str) -> str:
        if self.waited_s is None:
            return f'    {name}: NOTHING DELIVERED'
        median_ms = statistics.median(self.latencies_s) * 1e3
        missed = self.emitted - len(self.seqs)
        return (
            f'    {name}: first frame is #{min(self.seqs)}, seen after {self.waited_s:.2f}s '
            f'(in flight {self.first_latency_s * 1e3:.1f}ms, median {median_ms:.1f}ms), '
            f'saw {len(self.seqs)}/{self.emitted} frames, missed {missed}'
        )


class Console(pimm.ControlSystem):
    """A console: reads every camera each round and reports what it observed of each."""

    def __init__(self, label: str, window_s: float):
        self.label = label
        self.window_s = window_s
        self.cameras = pimm.ReceiverDict(self, default=None)
        self.directive = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        seen = {name: CameraObservations() for name in self.cameras}
        started = time.monotonic()
        while not should_stop.value:
            waited = time.monotonic() - started
            for name, receiver in self.cameras.items():
                message = receiver.read()
                if message.data is None:
                    continue
                # Read the clock after the read, so a frame emitted mid-round is not timed as arriving early.
                seen[name].observe(_seq_of(message.data.array), time.monotonic() - message.ts / 1e9, waited)
            delivering = [record for record in seen.values() if record.waited_s is not None]
            if len(delivering) == len(seen) and waited - max(r.waited_s for r in delivering) >= self.window_s:
                break
            if waited >= FIRST_FRAME_TIMEOUT_S:
                break
            yield pimm.Sleep(CONSOLE_PERIOD_S)

        print(f'  {self.label}:')
        for name, record in seen.items():
            print(record.report(name))


class StallingPeer(pimm.ControlSystem):
    """A foreground peer that blocks rather than yields, standing in for a synchronous inference call."""

    def __init__(self, stall_s: float):
        self.stall_s = stall_s

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        while not should_stop.value:
            time.sleep(self.stall_s)
            yield pimm.Sleep(CAMERA_PERIOD_S)


class IdlePolicy:
    """Enough policy to build a harness. No episode is ever started, so it is never asked to act."""

    def new_session(self, *_args, **_kwargs):
        return type('Session', (), {'close': lambda _self: None})()

    def close(self):
        pass


def _compose(cameras: int) -> tuple[Embodiment, Harness]:
    cams = [Camera() for _ in range(cameras)]
    embodiment = Embodiment(
        descriptor='probe',
        observations={f'{keys.IMAGE_PREFIX}{i}': Observation(cam.frames, None) for i, cam in enumerate(cams)},
        commands={},
        static_meta={},
        meta_source=None,
        control_systems=tuple(cams),
        simulated=False,
    )
    return embodiment, Harness(IdlePolicy(), embodiment, on_episode_complete=lambda *_a, **_k: None)


def run_one(*, console_foreground: bool, cameras: int, window_s: float, stall_s: float) -> None:
    where = 'FOREGROUND (`rollouts-console local`)' if console_foreground else 'BACKGROUND (`rollouts-console real`)'
    print(f'--- console {where} ---')
    embodiment, harness = _compose(cameras)
    console = Console('console', window_s)
    with pimm.World() as world:
        wire.wire_embodiment(world, harness, embodiment, None)
        world.connect(console.directive, harness.directive)
        for name, observation in embodiment.observations.items():
            world.connect(observation.source, console.cameras[name])
        foreground: list[pimm.ControlSystem] = [harness]
        if stall_s > 0:
            foreground.append(StallingPeer(stall_s))
        background = list(embodiment.control_systems)
        (foreground if console_foreground else background).append(console)
        world.run(foreground, background)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cameras', type=int, default=2, help='how many cameras to compose (the lab rig has 2)')
    parser.add_argument('--window', type=float, default=3.0, help='seconds to observe after the first frame')
    parser.add_argument('--stall-ms', type=float, default=0.0, help='blocking work per round in the harness loop')
    args = parser.parse_args()

    for console_foreground in (True, False):
        run_one(
            console_foreground=console_foreground,
            cameras=args.cameras,
            window_s=args.window,
            stall_s=args.stall_ms / 1e3,
        )


if __name__ == '__main__':
    main()
