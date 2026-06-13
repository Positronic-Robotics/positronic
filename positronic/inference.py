import logging
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

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
    """An attended operator surface: the directive source ``main`` wires into the Harness.

    Driver configs produce a factory called with the resolved local output directory, since
    the directory exists only after ``pos3.sync`` inside ``main``.
    """

    gui: DearpyguiUi | None
    directives: pimm.SignalEmitter
    directive_wrapper: Callable
    control_systems: list[pimm.ControlSystem]


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
        print('Keyboard controls: [s]tart, sto[p], [r] home, [q]uit')
        return Driver(
            None if not show_gui else DearpyguiUi(),
            keyboard.keyboard_inputs,
            pimm.map(keyboard_handler.harness_directive),
            [keyboard],
        )

    return make


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


def main(
    embodiment: Embodiment,
    policy,
    driver: Callable[[Path | None], Driver] | None = None,
    output_dir: str | Path | None = None,
    task: Task | None = None,
    trials: list[dict] | None = None,
    show_gui: bool = False,
    wrap=default_wrappers,
):
    """Run inference for an embodiment, real or simulated.

    ``task`` (when given) supplies the policy-facing instruction, the per-trial ``timeout``
    bounding self-driven trials, the privileged ground-truth signals to record, and the seeded
    scene reset run at each trial start; without it the instruction rides the driver. Exactly
    one of ``driver`` (attended: a factory producing the operator surface that emits the
    directives) or ``trials`` (unattended: the harness runs the plan itself) must be given;
    ``show_gui`` applies to the unattended path (attended surfaces bring their own).
    """
    assert (driver is None) != (trials is None), 'Provide exactly one of driver or trials'
    harness = Harness(
        policy, embodiment, task=task, trials=trials, wrap=wrap, on_episode_complete=_completion_sink(policy)
    )

    if output_dir is not None:
        output_dir = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])
        _seed_counter(policy, output_dir)

    driver = driver(output_dir) if driver is not None else None
    gui = driver.gui if driver is not None else (DearpyguiUi() if show_gui else None)

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
        if driver is not None:
            world.connect(driver.directives, harness.directive, emitter_wrapper=driver.directive_wrapper)
        if ds_agent is not None:
            world.connect(harness.ds_command, ds_agent.command)

        # Sim runs devices + recorder in-process under the virtual clock; real runs them as
        # background subprocesses. The harness, driver, and GUI placement is identical.
        foreground = driver.control_systems if driver is not None else []
        devices = [cs for cs in [*embodiment.control_systems, ds_agent] if cs is not None]
        if embodiment.simulated:
            world.run([harness, *foreground, *devices], gui)
        else:
            world.run([harness, *foreground], [*devices, gui])


run_cfg = cfn.Config(main, embodiment=positronic.cfg.embodiment.droid, policy=policy_cfg.placeholder, driver=keyboard)


# Console entry point for [project.scripts].
@pos3.with_mirror()
def _internal_main():
    # Imported here to break the circular import: positronic.cfg.eval imports this module.
    from positronic.cfg.eval import attended, run as eval_run  # noqa
    from positronic.cfg.eval.sim.positronic import stack_cubes  # noqa

    init_logging()
    cfn.cli({
        'run': run_cfg,
        'real': run_cfg,  # `real` is the documented name for the hardware path
        'sim': eval_run.override(eval=stack_cubes),
        'sim_ui': attended.override(eval=stack_cubes),
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
