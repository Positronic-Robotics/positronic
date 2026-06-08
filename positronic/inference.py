import logging
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.embodiment
import positronic.cfg.policy as policy_cfg
import positronic.cfg.simulator
from positronic import utils, wire
from positronic.dataset.ds_writer_agent import TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter, load_all_datasets
from positronic.embodiment import Embodiment
from positronic.gui.dpg import DearpyguiUi
from positronic.gui.eval import EvalUI
from positronic.gui.keyboard import KeyboardControl
from positronic.policy.base import SampledPolicy
from positronic.policy.harness import Directive, Harness
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


class TimedDriver(pimm.ControlSystem):
    """Control system that orchestrates inference episodes by sending directives."""

    def __init__(self, num_iterations: int, simulation_time: float, task: str | None = None):
        self.num_iterations = num_iterations
        self.simulation_time = simulation_time
        self.directives = pimm.ControlSystemEmitter(self)
        self.task = task

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        for i in range(self.num_iterations):
            meta = {'simulation.iteration': str(i), 'simulation.total_iterations': str(self.num_iterations)}
            if self.task:
                meta['task'] = self.task
            self.directives.emit(Directive.RUN(**meta))
            yield pimm.Sleep(self.simulation_time)
            self.directives.emit(Directive.FINISH())
            yield pimm.Sleep(0.5)


@cfn.config(ui_scale=1)
def eval_ui(ui_scale):
    gui = EvalUI(ui_scale=ui_scale)
    return gui, (gui.directive, pimm.utils.identity), []


@cfn.config(show_gui=False)
def keyboard(show_gui, task):
    keyboard = KeyboardControl(quit_key='q')
    keyboard_handler = KeyboardHandler(task=task)
    print('Keyboard controls: [s]tart, sto[p], [r] home, [q]uit')
    return (
        None if not show_gui else DearpyguiUi(),
        (keyboard.keyboard_inputs, pimm.map(keyboard_handler.harness_directive)),
        [keyboard],
    )


@cfn.config(num_iterations=1, simulation_time=15, show_gui=False)
def timed(num_iterations, simulation_time, show_gui, task):
    gui = None if not show_gui else DearpyguiUi()
    driver = TimedDriver(num_iterations, simulation_time, task=task)
    return gui, (driver.directives, pimm.utils.identity), [driver]


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
    driver: tuple,
    output_dir: str | Path | None = None,
    simulate_inference: bool | float = False,
):
    """Run inference for an embodiment, real or simulated."""
    harness = Harness(
        policy, embodiment, simulate_inference=simulate_inference, on_episode_complete=_completion_sink(policy)
    )
    gui, harness_emitter, foreground_cs = driver

    if output_dir is not None:
        output_dir = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])
        _seed_counter(policy, output_dir)

    time_mode = TimeMode.MESSAGE if embodiment.simulated else TimeMode.CLOCK
    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(virtual_time=embodiment.simulated) as world:
        ds_agent = wire.wire_embodiment(world, harness, embodiment, dataset_writer, time_mode)
        if gui is not None:
            # HACK: GUI cameras are matched to observations by the `image.` name prefix. This
            # hard-binds GUI wiring to the observation naming convention; TODO: resolve via an
            # explicit camera declaration on the embodiment rather than a string prefix.
            for name, obs in embodiment.observations.items():
                if name.startswith('image.'):
                    world.connect(obs.source, gui.cameras[name])
        world.connect(harness_emitter[0], harness.directive, emitter_wrapper=harness_emitter[1])
        _connect_ds_command(world, harness, ds_agent, policy)

        # Sim runs devices + recorder in-process under the virtual clock; real runs them as
        # background subprocesses. The harness, driver, and GUI placement is identical.
        devices = [cs for cs in [*embodiment.control_systems, ds_agent] if cs is not None]
        if embodiment.simulated:
            world.run([harness, *foreground_cs, *devices], gui)
        else:
            world.run([harness, *foreground_cs], [*devices, gui])


run_cfg = cfn.Config(main, embodiment=positronic.cfg.embodiment.droid, policy=policy_cfg.placeholder, driver=keyboard)


sim_cfg = run_cfg.override(
    embodiment=positronic.cfg.embodiment.sim,
    driver=timed.override(simulation_time=15, task='Pick up the green cube and place it on the red cube.'),
)


# Separate function for [projects.scripts]
@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({
        'run': run_cfg,
        'real': run_cfg,  # back-compat alias for the hardware path (documented as `real`)
        'sim': sim_cfg,
        'phail': run_cfg.override(
            policy=policy_cfg.phail_multiple,
            driver=eval_ui,
            **{'driver.ui_scale': 3, 'embodiment.robot_arm.collision_coeff': 2.0},
        ),
        'sim_pnp': sim_cfg.override(**{
            'embodiment.loaders': positronic.cfg.simulator.multi_tote_loaders,
            'embodiment.observers': {},
            'driver.task': 'Pick up objects from the red tote and place them in the green tote.',
        }),
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
