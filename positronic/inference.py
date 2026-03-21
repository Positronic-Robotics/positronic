import time
from collections.abc import Callable, Mapping, Sequence
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
from positronic.dataset.ds_writer_agent import DsWriterCommand, TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.gui.dpg import DearpyguiUi
from positronic.gui.eval import EvalUI
from positronic.gui.keyboard import KeyboardControl
from positronic.policy.harness import Directive, Harness
from positronic.simulator.mujoco.observers import BodyDistance, StackingSuccess
from positronic.simulator.mujoco.sim import MujocoCameras, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
from positronic.utils import package_assets_path
from positronic.utils.logging import init_logging


class DsWriterCommandMetaBridge(pimm.ControlSystem):
    """Runs in main process: injects harness meta into RUN commands.

    This avoids computing metadata inside background GUI processes, which would
    otherwise use a spawned copy of `Harness`/policy and can log stale
    or wrong policy meta when policies metadata is resampled in the main process.
    """

    def __init__(self, meta_provider: Callable[[], dict[str, Any]]):
        self.meta_provider = meta_provider
        self.in_cmd = pimm.ControlSystemReceiver[DsWriterCommand](self, default=None, maxsize=10)
        self.out_cmd = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        while not should_stop.value:
            msg = self.in_cmd.read()
            if msg is not None and msg.updated and msg.data is not None:
                cmd = msg.data
                out = None
                if cmd.type == DsWriterCommand.START(None).type:
                    meta = (cmd.static_data or {}) | self.meta_provider()
                    out = DsWriterCommand(cmd.type, meta)

                self.out_cmd.emit(out if out is not None else cmd)
            yield pimm.Pass()


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

    def ds_writer_command(self, key: str) -> DsWriterCommand | None:
        match key:
            case 's':
                return DsWriterCommand.START({'task': self.task} if self.task else {})
            case 'p':
                return DsWriterCommand.STOP()
            case 'r':
                return DsWriterCommand.ABORT()
        return None


class TimedDriver(pimm.ControlSystem):
    """Control system that orchestrates inference episodes by sending start/stop commands."""

    def __init__(self, num_iterations: int, simulation_time: float, task: str | None = None):
        self.num_iterations = num_iterations
        self.simulation_time = simulation_time
        self.ds_commands = pimm.ControlSystemEmitter(self)
        self.inf_commands = pimm.ControlSystemEmitter(self)
        self.task = task

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        for i in range(self.num_iterations):
            meta = {'simulation.iteration': str(i), 'simulation.total_iterations': str(self.num_iterations)}
            if self.task:
                meta['task'] = self.task
            self.ds_commands.emit(DsWriterCommand.START(meta))
            self.inf_commands.emit(Directive.RUN(task=self.task))
            yield pimm.Sleep(self.simulation_time)
            self.ds_commands.emit(DsWriterCommand.STOP())
            self.inf_commands.emit(Directive.HOME())
            yield pimm.Sleep(0.5)  # Let the things propagate


@cfn.config(ui_scale=1)
def eval_ui(ui_scale):
    gui = EvalUI(ui_scale=ui_scale)
    return gui, (gui.directive, pimm.utils.identity), (gui.ds_writer_command, pimm.utils.identity), []


@cfn.config(show_gui=False)
def keyboard(show_gui, task):
    keyboard = KeyboardControl(quit_key='q')
    keyboard_handler = KeyboardHandler(task=task)
    print('Keyboard controls: [s]tart, sto[p], [r] home, [q]uit')
    return (
        None if not show_gui else DearpyguiUi(),
        (keyboard.keyboard_inputs, pimm.map(keyboard_handler.harness_directive)),
        (keyboard.keyboard_inputs, pimm.map(keyboard_handler.ds_writer_command)),
        [keyboard],
    )


@cfn.config(num_iterations=1, simulation_time=15, show_gui=False)
def timed(num_iterations, simulation_time, show_gui, task):
    gui = None if not show_gui else DearpyguiUi()
    driver = TimedDriver(num_iterations, simulation_time, task=task)
    return gui, (driver.inf_commands, pimm.utils.identity), (driver.ds_commands, pimm.utils.identity), [driver]


def main(
    robot_arm: pimm.ControlSystem,
    gripper: pimm.ControlSystem,
    cameras: dict[str, pimm.ControlSystem],
    policy,
    driver: tuple,
    output_dir: str | Path | None = None,
):
    """Runs inference on real hardware."""
    harness = Harness(policy)

    # Convert camera instances to emitters for wire()
    camera_instances = cameras
    camera_emitters = {name: cam.frame for name, cam in camera_instances.items()}

    gui, harness_emitter, ds_writer_emitter, foreground_cs = driver
    if output_dir is not None:
        output_dir = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])

    robot_meta = robot_arm.robot_meta
    robot_meta['joint_signal'] = 'robot_state.q'
    robot_meta['pose_signals'] = ['robot_state.ee_pose', 'robot_commands.pose']

    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World() as world:
        ds_agent = wire.wire(world, harness, dataset_writer, camera_emitters, robot_arm, gripper, gui, TimeMode.CLOCK)
        ds_bridge = None
        world.connect(harness_emitter[0], harness.directive, emitter_wrapper=harness_emitter[1])
        if ds_agent is not None:
            ds_bridge = DsWriterCommandMetaBridge(lambda: robot_meta | harness.meta())
            world.connect(ds_writer_emitter[0], ds_bridge.in_cmd, emitter_wrapper=ds_writer_emitter[1])
            world.connect(ds_bridge.out_cmd, ds_agent.command)

        bg_cs = [*camera_instances.values(), robot_arm, gripper, ds_agent, gui]

        main_cs = [harness, ds_bridge, *foreground_cs]
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
    # Map signal names to emitters for wire()
    cameras = {name: mujoco_cameras.cameras[orig_name] for name, orig_name in camera_dict.items()}
    harness = Harness(policy, simulate_timeout=simulate_timeout)
    control_systems = [mujoco_cameras, sim, robot_arm, gripper, harness]

    sim_meta = {
        'simulation.mujoco_model_path': mujoco_model_path,
        'joint_signal': 'robot_state.q',
        'pose_signals': ['robot_state.ee_pose', 'robot_commands.pose'],
    }

    gui, harness_emitter, ds_writer_emitter, foreground_cs = driver

    if output_dir is not None:
        output_dir = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])

    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        ds_agent = wire.wire(world, harness, dataset_writer, cameras, robot_arm, gripper, gui, TimeMode.MESSAGE)
        ds_bridge = DsWriterCommandMetaBridge(lambda: sim_meta.copy() | harness.meta())
        if ds_agent is not None:
            for observer_name in observers.keys():
                ds_agent.add_signal(observer_name)
                world.connect(sim.observations[observer_name], ds_agent.inputs[observer_name])
        world.connect(harness_emitter[0], harness.directive, emitter_wrapper=harness_emitter[1])
        if ds_agent is not None:
            # Note: do NOT use ds_writer_emitter wrapper here. In eval_ui it may run in a
            # background process and inject stale harness metadata.
            world.connect(ds_writer_emitter[0], ds_bridge.in_cmd, emitter_wrapper=ds_writer_emitter[1])
            world.connect(ds_bridge.out_cmd, ds_agent.command)

        for _ in world.start([*foreground_cs, *control_systems, ds_agent, ds_bridge], gui):
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
    })


if __name__ == '__main__':
    _internal_main()
