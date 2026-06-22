"""Legacy ``positronic-inference`` CLI: the attended ``real``/``phail`` aliases over the ``cli.eval.run`` runner."""

from collections import Counter
from pathlib import Path

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.embodiment
import positronic.cfg.policy as policy_cfg
from positronic.cfg.eval.sim.positronic import stack_cubes
from positronic.cli.eval.run import Driver, main, run
from positronic.dataset.local_dataset import load_all_datasets
from positronic.gui.dpg import DearpyguiUi
from positronic.gui.eval import EvalUI
from positronic.gui.keyboard import KeyboardControl
from positronic.policy.harness import Directive
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


@cfn.config(ui_scale=1)
def eval_ui(ui_scale):
    def make(output_dir: Path | None) -> Driver:
        gui = EvalUI(output_dir, ui_scale=ui_scale)
        return Driver(gui, gui.directive, pimm.utils.identity, [])

    return make


@cfn.config(show_gui=False)
def keyboard(show_gui, task):
    def make(output_dir: Path | None) -> Driver:
        keyboard = KeyboardControl(quit_key='q')
        keyboard_handler = KeyboardHandler(task=task)
        print('Keyboard controls: [s]tart, sto[p], abo[r]t, [q]uit')
        return Driver(
            None if not show_gui else DearpyguiUi(),
            keyboard.keyboard_inputs,
            pimm.map(keyboard_handler.harness_directive),
            [keyboard],
        )

    return make


run_cfg = cfn.Config(main, embodiment=positronic.cfg.embodiment.droid, policy=policy_cfg.placeholder, driver=keyboard)


# Console entry point for [project.scripts].
@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({
        'run': run_cfg,
        'real': run_cfg,  # `real` is the documented name for the hardware path
        'sim': run.override(eval=stack_cubes),
        'phail': run_cfg.override(
            policy=policy_cfg.phail_multiple,
            driver=eval_ui,
            **{'driver.ui_scale': 3, 'embodiment.robot_arm.collision_coeff': 2.0},
        ),
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
