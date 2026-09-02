"""Legacy ``positronic-inference`` CLI: the attended keyboard ``real`` path plus the ``sim`` and ``stats``
aliases over ``cli.eval.run``."""

from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.embodiment
import positronic.cfg.eval.real.droid
import positronic.cfg.eval.real.trossen
import positronic.cfg.policy as policy_cfg
from pimm.logging import init_logging
from positronic import keys, wire
from positronic.cfg.eval.sim.positronic import stack_cubes
from positronic.cli.eval.run import prepare_output_dir, run
from positronic.dataset.local_dataset import LocalDatasetWriter, load_all_datasets
from positronic.drivers.keyboard import KeyboardControl
from positronic.eval import Embodiment, Task
from positronic.policy.harness import Harness


class KeyboardOperator(pimm.ControlSystem):
    """Turns keystrokes into episodes: ``s`` asks for one through ``perform_task``, ``p`` ends the live one.

    It holds the pending answers because that is where an episode's terminal — and any refused ask —
    arrives; both are printed as they land. ``next_task`` is called once per press.
    """

    def __init__(self, next_task: Callable[[], Task]):
        self._next_task = next_task
        self.keystrokes = pimm.ControlSystemReceiver[str](self)
        self.perform_task = pimm.calls.ControlSystemCaller[Task, dict[str, Any]](self)
        self.done = pimm.ControlSystemEmitter[dict[str, Any]](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        pending: list[pimm.calls.Answer[dict[str, Any]]] = []
        while not should_stop.value:
            if (key := pimm.value_updated(self.keystrokes)) is not None:
                match key:
                    case 's':
                        pending.append(self.perform_task(self._next_task()))
                    case 'p':
                        self.done.emit({keys.EVAL_ENDED_BY: keys.ENDED_BY_OPERATOR})
            running = []
            for answer in pending:
                if answer.done():
                    self._report(answer)
                else:
                    running.append(answer)
            pending = running
            yield pimm.Sleep(0.01)

    @staticmethod
    def _report(answer: pimm.calls.Answer[dict[str, Any]]) -> None:
        """Print what the episode ended on, or why it never will."""
        try:  # rules-allow: swallowed-error — the operator is who this failure is for
            print(f'Episode ended: {answer.result()}')
        except Exception as e:
            print(f'Episode failed: {e}')


def real(policy, embodiment: Embodiment, next_task: Callable[[], Task], output_dir=None):
    """Run one hardware embodiment attended and headless, the keyboard deciding when an episode starts and
    finishes.

    The world is composed here rather than by the runner: an attended surface is the binary's own business,
    and the keyboard is the only one this library ships. There is no viewer — a console that shows the
    cameras is a binary of its own, composing a world around ``Harness``, ``wire.wire_embodiment`` and
    ``gui.dpg_ui``. A run ends when ``KeyboardControl`` returns — on ``q``, or on a stdin that is not a
    terminal — since a control system returning stops the world.
    """
    if embodiment.simulated:
        raise ValueError('the keyboard path drives hardware in real time; run a simulated embodiment as `sim`')

    # The policy is this function's to close from here on, and everything below can raise:
    # `prepare_output_dir` syncs a directory and snapshots sources into it, and `LocalDatasetWriter`
    # scans the one it is given.
    try:
        _run_attended(policy, embodiment, next_task, output_dir)
    finally:
        policy.close()


def _run_attended(policy, embodiment: Embodiment, next_task: Callable[[], Task], output_dir) -> None:
    """Record from a warmed policy until the keyboard returns. The caller owns the policy."""
    output_dir = prepare_output_dir(output_dir)
    keyboard = KeyboardControl(quit_key='q')
    operator = KeyboardOperator(next_task)
    harness = Harness(policy, embodiment)
    print('Keyboard controls: [s]tart, sto[p], [q]uit')

    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World() as world:
        ds_agent = wire.wire_embodiment(world, harness, embodiment, dataset_writer, done=operator.done)
        world.connect(keyboard.keyboard_inputs, operator.keystrokes)
        world.connect(operator.perform_task, harness.perform_task)
        if ds_agent is not None:
            world.connect(harness.ds_command, ds_agent.command)
        producers = [cs for cs in embodiment.control_systems if cs is not None]
        world.run([harness, keyboard, operator], [*producers, ds_agent])


real_cfg = cfn.Config(
    real,
    embodiment=positronic.cfg.embodiment.droid,
    next_task=positronic.cfg.eval.real.droid.attended_trials,
    policy=policy_cfg.placeholder,
)


# The Trossen station, whose arm and cameras are the ones its demonstrations were recorded with. Every
# trial opens at the same start pose the operator's own trials open at.
trossen_cfg = real_cfg.override(
    embodiment=positronic.cfg.embodiment.trossen, next_task=positronic.cfg.eval.real.trossen.attended_trials
)


# Console entry point for [project.scripts].
@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({
        'run': real_cfg,
        'real': real_cfg,  # `real` is the documented name for the hardware path
        'trossen': trossen_cfg,
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
