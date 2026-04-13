import logging
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3

import pimm
import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.policy as policy_cfg
import positronic.cfg.simulator
from positronic import utils, wire
from positronic.dataset.ds_writer_agent import DsWriterCommandType, TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter, load_all_datasets
from positronic.gui.dpg import DearpyguiUi
from positronic.gui.eval import EvalUI
from positronic.gui.keyboard import KeyboardControl
from positronic.policy.base import SampledPolicy
from positronic.policy.harness import Directive, Harness
from positronic.policy.sampler import BalancedSampler
from positronic.simulator.mujoco.observers import BodyDistance, StackingSuccess
from positronic.simulator.mujoco.sim import MujocoCameras, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
from positronic.utils import package_assets_path
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
                return Directive.STOP()
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


def _seed_sampler(policy, output_dir: Path):
    """If policy has a BalancedSampler, seed it from existing episodes in output_dir."""
    if not isinstance(policy, SampledPolicy) or not isinstance(policy.sampler, BalancedSampler):
        return
    try:
        dataset = load_all_datasets(output_dir)
    except ValueError:
        return
    if len(dataset) == 0:
        return
    meta_key = f'inference.policy.{policy._key_field}'
    group_fields = policy.sampler.group_fields or ()
    for i in range(len(dataset)):
        static = dataset[i].static
        key = static.get(meta_key)
        if key is not None:
            policy.sampler.count(key, {f: static.get(f) for f in group_fields})
    logger.info(f'Seeded sampler from {len(dataset)} existing episodes')


def _connect_ds_command(world, harness, ds_agent, policy):
    """Connect harness.ds_command to ds_agent, with optional sampler tap."""
    if ds_agent is None:
        return

    def _tap(cmd):
        if cmd.type is DsWriterCommandType.STOP_EPISODE:
            policy.count_current()
        return cmd

    wrapper = pimm.map(_tap) if isinstance(policy, SampledPolicy) else pimm.utils.identity
    world.connect(harness.ds_command, ds_agent.command, emitter_wrapper=wrapper)


def main(
    robot_arm: pimm.ControlSystem,
    gripper: pimm.ControlSystem,
    cameras: dict[str, pimm.ControlSystem],
    policy,
    driver: tuple,
    output_dir: str | Path | None = None,
):
    """Runs inference on real hardware."""
    harness = Harness(policy, static_meta=wire.ROBOT_STATIC_META)

    camera_instances = cameras
    camera_emitters = {name: cam.frame for name, cam in camera_instances.items()}

    gui, harness_emitter, foreground_cs = driver
    if output_dir is not None:
        output_dir = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])
        _seed_sampler(policy, output_dir)

    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World() as world:
        ds_agent = wire.wire(world, harness, dataset_writer, camera_emitters, robot_arm, gripper, gui, TimeMode.CLOCK)
        world.connect(harness_emitter[0], harness.directive, emitter_wrapper=harness_emitter[1])
        _connect_ds_command(world, harness, ds_agent, policy)

        bg_cs = [*camera_instances.values(), robot_arm, gripper, ds_agent, gui]
        main_cs = [harness, *foreground_cs]
        for cmd in world.start(main_cs, bg_cs):
            time.sleep(cmd.seconds)


def main_sim(
    mujoco_model_path: str,
    policy,
    loaders: Sequence[MujocoSceneTransform],
    camera_fps: int,
    driver: tuple,
    camera_dict: Mapping[str, str],
    output_dir: str | Path | None = None,
    simulate_timeout: bool = False,
    observers: Mapping[str, Any] | None = None,
):
    observers = observers or {}
    sim = MujocoSim(mujoco_model_path, loaders, observers=observers)
    robot_arm = MujocoFranka(sim, suffix='_ph')
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    mujoco_cameras = MujocoCameras(sim.model, sim.data, resolution=(320, 240), fps=camera_fps)
    cameras = {name: mujoco_cameras.cameras[orig_name] for name, orig_name in camera_dict.items()}

    static_meta = {'simulation.mujoco_model_path': mujoco_model_path, **wire.ROBOT_STATIC_META}
    harness = Harness(policy, static_meta=static_meta, simulate_timeout=simulate_timeout)
    control_systems = [mujoco_cameras, sim, robot_arm, gripper, harness]

    gui, harness_emitter, foreground_cs = driver

    if output_dir is not None:
        output_dir = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])
        _seed_sampler(policy, output_dir)

    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        ds_agent = wire.wire(world, harness, dataset_writer, cameras, robot_arm, gripper, gui, TimeMode.MESSAGE)
        if ds_agent is not None:
            for observer_name in observers.keys():
                ds_agent.add_signal(observer_name)
                world.connect(sim.observations[observer_name], ds_agent.inputs[observer_name])
        _connect_ds_command(world, harness, ds_agent, policy)
        world.connect(harness_emitter[0], harness.directive, emitter_wrapper=harness_emitter[1])

        for _ in world.start([*foreground_cs, *control_systems, ds_agent], gui):
            pass


main_sim_cfg = cfn.Config(
    main_sim,
    mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
    policy=policy_cfg.placeholder,
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
    camera_fps=15,
    driver=timed.override(simulation_time=15, task='Pick up the green cube and place it on the red cube.'),
    # We use 3 cameras not because we need it, but because Mujoco does not render
    # the second image when using only 2 cameras
    camera_dict={'image.wrist': 'handcam_left_ph', 'image.exterior': 'back_view_ph', 'image.agent_view': 'agentview'},
    observers={
        'box_distance': BodyDistance('box_0_body', 'box_1_body'),
        'stacking_success': StackingSuccess('box_0_body', 'box_1_body', 'hand_ph', full_report=True),
    },
)


droid_setup = cfn.Config(
    main,
    robot_arm=positronic.cfg.hardware.roboarm.franka_droid,
    gripper=positronic.cfg.hardware.gripper.robotiq,
    cameras={
        'image.wrist': positronic.cfg.hardware.camera.zed_m.override(
            view='left', resolution='hd720', fps=30, image_enhancement=True
        ),
        'image.exterior': positronic.cfg.hardware.camera.zed_2i.override(
            view='left', resolution='hd720', fps=30, image_enhancement=True
        ),
    },
    driver=keyboard,
    policy=policy_cfg.placeholder,
)


# Separate function for [projects.scripts]
@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({
        'sim': main_sim_cfg,
        'real': droid_setup,
        'phail': droid_setup.override(
            policy=policy_cfg.phail_multiple, driver=eval_ui, **{'driver.ui_scale': 3, 'robot_arm.collision_coeff': 2.0}
        ),
        'sim_pnp': main_sim_cfg.override(
            loaders=positronic.cfg.simulator.multi_tote_loaders,
            observers={},
            **{'driver.task': 'Pick up objects from the red tote and place them in the green tote.'},
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
