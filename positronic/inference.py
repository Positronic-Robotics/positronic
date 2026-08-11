"""Legacy ``positronic-inference`` CLI: the attended keyboard ``real`` path plus the ``sim`` and ``stats``
aliases over ``cli.eval.run``."""

from collections import Counter
from contextlib import nullcontext

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.embodiment
import positronic.cfg.policy as policy_cfg
from positronic import wire
from positronic.cfg.eval.sim.positronic import stack_cubes
from positronic.cli.eval.run import completion_sink, prepare_output_dir, run, warm_up
from positronic.dataset.local_dataset import LocalDatasetWriter, load_all_datasets
from positronic.eval import Embodiment
from positronic.gui.keyboard import KeyboardControl
from positronic.policy.harness import Directive, Harness
from positronic.utils.logging import init_logging


class KeyboardHandler:
    def __init__(self, task: str | None = None):
        self.task = task

    def harness_directive(self, key: str) -> Directive | None:
        match key:
            case 's':
                return Directive.RUN(task=self.task)
            case 'p':
                return Directive.FINISH()
            case 'r':
                return Directive.ABORT()
        return None


def real(policy, embodiment: Embodiment, task: str | None, output_dir=None):
    """Run one hardware embodiment attended and headless, the keyboard deciding when an episode starts,
    finishes or aborts.

    The world is composed here rather than by the runner: an attended surface is the binary's own business,
    and the keyboard is the only one this library ships. There is no viewer — a console that shows the
    cameras is a binary of its own, composing a world around ``Harness``, ``wire.wire_embodiment`` and
    ``gui.dpg_ui``. A run ends when ``KeyboardControl`` returns — on ``q``, or on a stdin that is not a
    terminal — since a main-process control system returning stops the world.
    """
    if embodiment.simulated:
        raise ValueError('the keyboard path drives hardware in real time; run a simulated embodiment as `sim`')

    warm_up(policy)
    output_dir = prepare_output_dir(policy, output_dir)
    harness = Harness(policy, embodiment, on_episode_complete=completion_sink(policy))
    keyboard = KeyboardControl(quit_key='q')
    handler = KeyboardHandler(task=task)
    print('Keyboard controls: [s]tart, sto[p], abo[r]t, [q]uit')

    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    try:
        with writer_cm as dataset_writer, pimm.World() as world:
            ds_agent = wire.wire_embodiment(world, harness, embodiment, dataset_writer)
            world.connect(
                keyboard.keyboard_inputs,
                harness.directive,
                # `World.connect` types its wrapper same-in-same-out; a keystroke->Directive map is not.
                emitter_wrapper=pimm.map(handler.harness_directive),  # pyright: ignore[reportArgumentType]
            )
            if ds_agent is not None:
                world.connect(harness.ds_command, ds_agent.command)
            producers = [cs for cs in embodiment.control_systems if cs is not None]
            world.run([harness, keyboard], [*producers, ds_agent])
    finally:
        policy.close()


real_cfg = cfn.Config(real, embodiment=positronic.cfg.embodiment.droid, policy=policy_cfg.placeholder)


# Console entry point for [project.scripts].
@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({
        'run': real_cfg,
        'real': real_cfg,  # `real` is the documented name for the hardware path
        'sim': run.override(eval=stack_cubes),
        'stats': stats,
    })


@cfn.config(fields=['eval.object', 'eval.external_camera', 'eval.tote_placement'])
def stats(output_dir: str, fields: list[str]):
    dataset = load_all_datasets(pos3.sync(output_dir))
    counts = Counter()
    for i in range(len(dataset)):
        static = dataset[i].static
        counts[tuple(static.get(f, 'N/A') for f in fields)] += 1

    n = len(fields)
    subtotals = [0] * n
    prev_key = None

    def _print_subtotal(level):
        row = list(prev_key[:level]) + ['Total'] + [''] * (n - level - 1) + [str(subtotals[level])]
        print('\t'.join(row))
        subtotals[level] = 0

    print('\t'.join(fields + ['count']))
    for key, count in sorted(counts.items()):
        if prev_key is not None:
            change_level = next((i for i in range(n) if key[i] != prev_key[i]), n)
            for level in range(n - 1, change_level, -1):
                _print_subtotal(level)

        print('\t'.join([*key, str(count)]))
        for level in range(n):
            subtotals[level] += count
        prev_key = key

    if prev_key is not None:
        for level in range(n - 1, -1, -1):
            _print_subtotal(level)


if __name__ == '__main__':
    _internal_main()
