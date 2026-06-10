import logging
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.embodiment
import positronic.cfg.policy as policy_cfg
from positronic import utils, wire
from positronic.dataset.ds_writer_agent import TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter, load_all_datasets
from positronic.eval import Embodiment, Task
from positronic.gui.dpg import DearpyguiUi
from positronic.gui.eval import EvalUI
from positronic.gui.keyboard import KeyboardControl
from positronic.policy.base import SampledPolicy
from positronic.policy.harness import Directive, Harness, default_wrappers
from positronic.utils.logging import init_logging

logger = logging.getLogger(__name__)


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
                return Directive.HOME()
        return None


@dataclass
class Driver:
    """What a driver config supplies: the directive source ``main`` wires into the Harness."""

    gui: DearpyguiUi | None
    directives: pimm.SignalEmitter
    directive_wrapper: Callable
    control_systems: list[pimm.ControlSystem]
    episode_ended: pimm.ControlSystemReceiver | None = None


class TrialSequencer(pimm.ControlSystem):
    """Pure trial sequencer: emits RUN per trial, then waits for the harness to end the episode.

    The harness owns trial termination (timeout, later the task's stop-signal); the
    sequencer just paces the trials and carries the per-trial RUN context.
    """

    def __init__(self, trial_count: int, run_context: dict[str, Any]):
        self.trial_count = trial_count
        self.run_context = run_context
        self.directives = pimm.ControlSystemEmitter(self)
        self.episode_ended = pimm.ControlSystemReceiver(self, default=None)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        for i in range(self.trial_count):
            context = {**self.run_context, 'eval.trial_index': i, 'eval.trial_count': self.trial_count}
            self.directives.emit(Directive.RUN(**context))
            while not should_stop.value and not self.episode_ended.read().updated:
                yield pimm.Sleep(0.01)
        yield pimm.Sleep(0.5)  # let the DsWriterAgent commit the last episode before world exit


@cfn.config(ui_scale=1)
def eval_ui(ui_scale):
    gui = EvalUI(ui_scale=ui_scale)
    return Driver(gui, gui.directive, pimm.utils.identity, [])


@cfn.config(show_gui=False)
def keyboard(show_gui, task):
    keyboard = KeyboardControl(quit_key='q')
    keyboard_handler = KeyboardHandler(task=task)
    print('Keyboard controls: [s]tart, sto[p], [r] home, [q]uit')
    return Driver(
        None if not show_gui else DearpyguiUi(),
        keyboard.keyboard_inputs,
        pimm.map(keyboard_handler.harness_directive),
        [keyboard],
    )


@cfn.config(trial_count=1, timeout=15, inference_latency=False, show_gui=False)
def sequencer(trial_count, timeout, inference_latency, show_gui):
    gui = None if not show_gui else DearpyguiUi()
    driver = TrialSequencer(trial_count, {'timeout': timeout, 'inference_latency': inference_latency})
    return Driver(gui, driver.directives, pimm.utils.identity, [driver], driver.episode_ended)


def _seed_counter(policy, output_dir: Path):
    """If policy is a SampledPolicy, seed its episode counter from existing episodes in output_dir."""
    if not isinstance(policy, SampledPolicy):
        return
    try:
        dataset = load_all_datasets(output_dir)
    except ValueError:
        return
    if len(dataset) == 0:
        return
    seeded = policy.counter.seed_from(dataset)
    logger.info(f'Seeded counter from {seeded} existing episodes')


def _completion_sink(policy):
    """Harness ``on_episode_complete`` callback that tallies completed episodes.

    Returns the ``SampledPolicy``'s counter ``record`` (which reads the sampled
    key from the session and bumps its tally), or ``None`` for non-sampled
    policies. The harness fires it on each clean episode completion.
    """
    return policy.counter.record if isinstance(policy, SampledPolicy) else None


def _connect_ds_command(world, harness, ds_agent, policy):
    """Connect harness.ds_command to ds_agent."""
    if ds_agent is None:
        return
    world.connect(harness.ds_command, ds_agent.command)


def main(
    embodiment: Embodiment,
    policy,
    driver: Driver,
    output_dir: str | Path | None = None,
    task: Task | None = None,
    wrap=default_wrappers,
):
    """Run inference for an embodiment, real or simulated.

    ``task`` (when given) supplies the policy-facing instruction and the privileged
    ground-truth signals to record; without it the instruction rides the driver.
    """
    harness = Harness(
        policy,
        embodiment,
        instruction=task.instruction if task is not None else None,
        wrap=wrap,
        on_episode_complete=_completion_sink(policy),
    )
    gui = driver.gui

    if output_dir is not None:
        output_dir = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])
        _seed_counter(policy, output_dir)

    time_mode = TimeMode.MESSAGE if embodiment.simulated else TimeMode.CLOCK
    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(virtual_time=embodiment.simulated) as world:
        privileged = task.privileged if task is not None else {}
        ds_agent = wire.wire_embodiment(world, harness, embodiment, dataset_writer, time_mode, privileged=privileged)
        if gui is not None:
            # HACK: GUI cameras are matched to observations by the `image.` name prefix, which
            # hard-binds GUI wiring to the observation naming convention. TODO: resolve this
            # coupling (the right binding is still open).
            for name, obs in embodiment.observations.items():
                if name.startswith('image.'):
                    world.connect(obs.source, gui.cameras[name])
        world.connect(driver.directives, harness.directive, emitter_wrapper=driver.directive_wrapper)
        if driver.episode_ended is not None:
            world.connect(harness.episode_ended, driver.episode_ended)
        _connect_ds_command(world, harness, ds_agent, policy)

        # Sim runs devices + recorder in-process under the virtual clock; real runs them as
        # background subprocesses. The harness, driver, and GUI placement is identical.
        devices = [cs for cs in [*embodiment.control_systems, ds_agent] if cs is not None]
        if embodiment.simulated:
            world.run([harness, *driver.control_systems, *devices], gui)
        else:
            world.run([harness, *driver.control_systems], [*devices, gui])


run_cfg = cfn.Config(main, embodiment=positronic.cfg.embodiment.droid, policy=policy_cfg.placeholder, driver=keyboard)


# Console entry point for [project.scripts].
@pos3.with_mirror()
def _internal_main():
    # Imported here to break the circular import: positronic.cfg.eval imports this module.
    from positronic.cfg.eval import run as eval_run  # noqa
    from positronic.cfg.eval.sim.positronic import stack_cubes  # noqa

    init_logging()
    cfn.cli({
        'run': run_cfg,
        'real': run_cfg,  # `real` is the documented name for the hardware path
        'sim': eval_run.override(eval=stack_cubes),
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
