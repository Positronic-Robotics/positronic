"""Legacy ``positronic-inference`` CLI: the attended keyboard ``real`` path plus the ``sim`` and ``stats``
aliases over ``cli.eval.run``."""

import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.embodiment
import positronic.cfg.eval.real.droid
import positronic.cfg.eval.real.trossen
import positronic.cfg.policy as policy_cfg
from pimm.logging import init_logging
from positronic import keys
from positronic.cfg.eval.sim.positronic import stack_cubes
from positronic.cli.eval.run import prepare_output_dir, run, run_world
from positronic.dataset.local_dataset import load_all_datasets
from positronic.drivers.keyboard import KeyboardControl
from positronic.eval import Embodiment, Task
from positronic.policy import Policy
from positronic.policy.harness import Rollout

logger = logging.getLogger(__name__)


class KeyboardOperator(KeyboardControl):
    """The keyboard that runs episodes: ``s`` asks for one through ``perform_task``, ``p`` ends the live one,
    ``q`` ends the run.

    One episode is in flight at a time: a press while one runs is declined here, with a warning. It holds
    the pending answer because that is where the episode's terminal — or a refused ask — arrives, and it
    logs that as it lands. ``next_task`` makes the trial and the policy opens its session, once per accepted
    press. Every episode records into ``output_path``, and none records when that is ``None``.
    """

    def __init__(self, next_task: Callable[[], Task], policy: Policy, output_path: Path | None):
        super().__init__(quit_key='q')
        self._next_task = next_task
        self._policy = policy
        self._output_path = output_path
        self._pending: pimm.calls.Answer[dict[str, Any]] | None = None
        self.perform_task = pimm.calls.ControlSystemCaller[Rollout, dict[str, Any]](self)
        self.done = pimm.ControlSystemEmitter[dict[str, Any]](self)

    def _each_round(self, key: str | None) -> None:
        super()._each_round(key)
        if self._pending is not None and self._pending.done():
            try:  # rules-allow: swallowed-error — the operator is who this failure is for
                logger.info(f'Episode ended: {self._pending.result()}')
            except Exception as e:
                logger.error(f'Episode failed: {e}')
            self._pending = None
        match key:
            case 's' if self._pending is not None:
                logger.warning('An episode is already running: press [p] to stop it')
            case 's':
                # A model that will not open a session ends the episode, not the run: the operator hears it
                # and presses again.
                try:  # rules-allow: swallowed-error — the operator is who this failure is for
                    self._pending = self.perform_task(Rollout(self._next_task(), self._policy, self._output_path))
                except Exception as e:
                    logger.error(f'Episode failed to open: {e}')
            case 'p':
                self.done.emit({keys.EVAL_ENDED_BY: keys.ENDED_BY_OPERATOR})


def real(policy, embodiment: Embodiment, next_task: Callable[[], Task], output_dir=None):
    """Run one hardware embodiment attended and headless, the keyboard deciding when an episode starts and
    finishes.

    The keyboard is the only attended surface this library ships, and there is no viewer — a console that
    shows the cameras composes a world of its own. A run ends when the operator returns — on ``q``, or on a
    stdin that is not a terminal — since a control system returning stops the world.
    """
    if embodiment.simulated:
        raise ValueError('the keyboard path drives hardware in real time; run a simulated embodiment as `sim`')

    # The policy is this function's to close from here on, and everything below can raise:
    # `prepare_output_dir` syncs a directory and snapshots sources into it, and `run_world` builds the
    # world the rig runs in.
    try:
        output_path = prepare_output_dir(output_dir)
        operator = KeyboardOperator(next_task, policy, output_path)
        logger.info('Keyboard controls: [s]tart, sto[p], [q]uit')
        run_world(embodiment, operator, record=output_path is not None, done=operator.done)
    finally:
        policy.close()


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
