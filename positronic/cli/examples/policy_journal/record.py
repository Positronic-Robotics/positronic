"""Record one journaled episode of a simulated motor.

    uv run --locked positronic/cli/examples/policy_journal/record.py JOURNAL

A motor integrates a velocity command every 2 ms and publishes its position. ``Move`` runs its inference
on a worker thread and plays the velocity chunks it returns. The harness journals the episode to the new
directory JOURNAL and writes no dataset. ``replay.py`` replays the journal offline.
"""

import argparse
from pathlib import Path
from typing import Any

import pimm
from positronic import wire
from positronic.cli.examples.policy_journal.move import MOTOR, POSITION, Move, infer
from positronic.eval import Command, Embodiment, Observation, Task
from positronic.policy.base import Obs
from positronic.policy.harness import Harness, Rollout
from positronic.policy.journal import Journal


class Motion(pimm.ControlSystem):
    """Integrate the commanded velocity once per two-millisecond physics step."""

    def __init__(self) -> None:
        self.command = pimm.ControlSystemReceiver[int](self)
        self.position = pimm.ControlSystemEmitter[int](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> pimm.Run[None]:
        position, velocity = 0, 0
        while not should_stop.value:
            yield pimm.Sleep(0.002)
            if (command := pimm.read_updated(self.command)) is not None:
                velocity = command.data
            position += velocity
            self.position.emit(position)


def record(journal: Journal) -> dict[str, Any]:
    """Run one 0.21 s episode on the simulated motor; return its terminal payload."""
    positions = []

    def counted(obs: Obs) -> list[dict[str, int]]:
        positions.append(obs[POSITION])
        return infer(obs)

    with pimm.World(virtual_time=True) as world:
        motion = Motion()
        embodiment = Embodiment(
            descriptor='journal-example',
            observations={POSITION: Observation(motion.position, None)},
            commands={MOTOR: Command(motion.command, None)},
            prepare_handlers={},
            static_meta={},
            meta_source=None,
            simulated=True,
        )
        harness = Harness(embodiment)
        wire.wire_embodiment(world, harness, embodiment, record=False)
        caller = world.pair(harness.perform_task)
        loop = world.start([harness, motion])
        task = Task('move', 0.21, charge_inference_time=False)
        finished = caller(Rollout(task, Move(counted), output_path=None, journal=journal))
        try:
            while not finished.done():
                next(loop)
            payload = finished.result()
        finally:
            world.request_stop()
            list(loop)
    print(f'the inference ran {len(positions)} times, at positions {positions}')
    return payload


def main(path: Path) -> None:
    print('episode:', record(Journal(path)))
    print('journal:', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('journal', type=Path, help='A directory for the journal; it must not exist')
    main(parser.parse_args().journal)
