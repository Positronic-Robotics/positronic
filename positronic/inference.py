import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import configuronic as cfn
import tqdm

import pimm
import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.policy.action
import positronic.cfg.policy.observation
import positronic.cfg.policy.policy
import positronic.cfg.simulator
import positronic.utils.s3 as pos3
from positronic import utils, wire
from positronic.dataset.ds_writer_agent import DsWriterCommand, TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.gui.dpg import DearpyguiUi
from positronic.gui.eval import EvalUI
from positronic.gui.keyboard import KeyboardControl
from positronic.policy.action import ActionDecoder
from positronic.policy.inference import Inference, InferenceCommand
from positronic.policy.observation import ObservationEncoder
from positronic.simulator.mujoco.observers import BodyDistance, StackingSuccess
from positronic.simulator.mujoco.sim import MujocoCameras, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
from positronic.utils import package_assets_path
from positronic.utils.logging import init_logging


class KeyboardHanlder:
    def __init__(self, meta_getter: Callable[[], dict[str, Any]]):
        self.meta_getter = meta_getter

    def inference_command(self, key: str) -> InferenceCommand | None:
        if key == 's':
            logging.info('Starting inference...')
            return InferenceCommand.START()
        elif key == 'p':
            logging.info('Stopping inference...')
            return InferenceCommand.STOP()
        elif key == 'r':
            logging.info('Resetting...')
            return InferenceCommand.RESET()
        return None

    def ds_writer_command(self, key: str) -> DsWriterCommand | None:
        if key == 's':
            return DsWriterCommand.START(self.meta_getter())
        elif key == 'p':
            return DsWriterCommand.STOP()
        elif key == 'r':
            return DsWriterCommand.ABORT()
        return None


class Driver(pimm.ControlSystem):
    """Control system that orchestrates inference episodes by sending start/stop commands."""

    def __init__(self, num_iterations: int, simulation_time: float, meta_getter: Callable[[], dict[str, Any]]):
        self.num_iterations = num_iterations
        self.simulation_time = simulation_time
        self.ds_commands = pimm.ControlSystemEmitter(self)
        self.inf_commands = pimm.ControlSystemEmitter(self)
        self.meta_getter = meta_getter

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        for i in range(self.num_iterations):
            meta = self.meta_getter()
            meta['simulation.iteration'] = i

            self.ds_commands.emit(DsWriterCommand.START(meta))
            self.inf_commands.emit(InferenceCommand.START())
            yield pimm.Sleep(self.simulation_time)
            self.ds_commands.emit(DsWriterCommand.STOP())
            self.inf_commands.emit(InferenceCommand.RESET())
            yield pimm.Sleep(0.2)  # Let the things propagate


def driver(control_mode, show_gui, meta_getter, **kwargs):
    match control_mode:
        case 'eval_ui':
            gui = EvalUI()

            def inject_metadata(cmd: DsWriterCommand) -> DsWriterCommand:
                if cmd.type == DsWriterCommand.START(None).type:
                    # Merge inference metadata into the start command
                    # Note: cmd.static_data from UI takes precedence if keys collide?
                    # Actually usually we want system config (inference.meta) + user input (task)
                    # inference.meta() is a dict.
                    new_static = meta_getter().copy()
                    new_static.update(cmd.static_data)
                    return DsWriterCommand(cmd.type, new_static)
                return cmd

            return gui, (gui.inference_command, lambda x: x), (gui.ds_writer_command, inject_metadata), []
        case 'keyboard':
            gui = None if not show_gui else DearpyguiUi()
            keyboard = KeyboardControl()
            keyboard_handler = KeyboardHanlder(meta_getter=meta_getter)
            print('Keyboard controls: [s]tart, sto[p], [r]eset')
            return (
                gui,
                (keyboard.keyboard_inputs, pimm.map(keyboard_handler.inference_command)),
                (keyboard.keyboard_inputs, pimm.map(keyboard_handler.ds_writer_command)),
                [keyboard],
            )
        case 'timed':
            gui = None if not show_gui else DearpyguiUi()
            num_iterations = kwargs.get('num_iterations', 1)
            simulation_time = kwargs.get('simulation_time', 10.0)
            driver = Driver(num_iterations, simulation_time, meta_getter)
            return (gui, (driver.inf_commands, lambda x: x), (driver.ds_commands, lambda x: x), [driver])
        case _:
            raise ValueError(f'Unknown control mode: {control_mode}')


def main(
    robot_arm: pimm.ControlSystem,
    gripper: pimm.ControlSystem,
    cameras: dict[str, pimm.ControlSystem],
    observation_encoder: ObservationEncoder,
    action_decoder: ActionDecoder,
    policy,
    policy_fps: int = 15,
    task: str | None = None,
    output_dir: str | Path | None = None,
    show_gui: bool = False,
    control_mode: str = 'keyboard',
):
    """Runs inference on real hardware."""
    inference = Inference(observation_encoder, action_decoder, policy, policy_fps, task)

    # Convert camera instances to emitters for wire()
    camera_instances = cameras
    camera_emitters = {name: cam.frame for name, cam in camera_instances.items()}

    if control_mode == 'timed':
        raise ValueError("Control mode 'timed' is not supported in main()")

    gui, inference_emitter, ds_writer_emitter, foreground_cs = driver(control_mode, show_gui, inference.meta)

    if output_dir is not None:
        output_dir = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])

    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World() as world:
        ds_agent = wire.wire(
            world, inference, dataset_writer, camera_emitters, robot_arm, gripper, gui, TimeMode.MESSAGE
        )
        world.connect(inference_emitter[0], inference.command, emitter_wrapper=inference_emitter[1])
        if ds_agent is not None:
            world.connect(ds_writer_emitter[0], ds_agent.command, emitter_wrapper=ds_writer_emitter[1])

        bg_cs = [*camera_instances.values(), robot_arm, gripper, ds_agent, gui]

        for cmd in world.start([inference] + foreground_cs, bg_cs):
            time.sleep(cmd.seconds)


def main_sim(
    mujoco_model_path: str,
    observation_encoder: ObservationEncoder,
    action_decoder: ActionDecoder,
    policy,
    loaders: Sequence[MujocoSceneTransform],
    camera_fps: int,
    policy_fps: int,
    simulation_time: float,
    camera_dict: Mapping[str, str],
    task: str | None,
    output_dir: str | Path | None = None,
    show_gui: bool = False,
    num_iterations: int = 1,
    simulate_timeout: bool = False,
    control_mode: str = 'timed',
    observers: Mapping[str, Any] | None = None,
):
    if observers is None:
        observers = {
            'box_distance': BodyDistance('box_0_body', 'box_1_body'),
            'stacking_success': StackingSuccess('box_0_body', 'box_1_body', 'hand_ph'),
        }
    sim = MujocoSim(mujoco_model_path, loaders, observers=observers)
    robot_arm = MujocoFranka(sim, suffix='_ph')
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    mujoco_cameras = MujocoCameras(sim.model, sim.data, resolution=(320, 240), fps=camera_fps)
    # Map signal names to emitters for wire()
    cameras = {name: mujoco_cameras.cameras[orig_name] for name, orig_name in camera_dict.items()}
    inference = Inference(observation_encoder, action_decoder, policy, policy_fps, task, simulate_timeout)
    control_systems = [mujoco_cameras, sim, robot_arm, gripper, inference]

    sim_meta = {'simulation.mujoco_model_path': mujoco_model_path, 'simulation.simulation_time': simulation_time}

    gui, inference_emitter, ds_writer_emitter, foreground_cs = driver(
        control_mode,
        show_gui,
        lambda: sim_meta.copy() | inference.meta(),
        num_iterations=num_iterations,
        simulation_time=simulation_time,
    )

    if output_dir is not None:
        output_dir = pos3.sync(output_dir, sync_on_error=True)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])

    writer_cm = LocalDatasetWriter(output_dir) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        ds_agent = wire.wire(world, inference, dataset_writer, cameras, robot_arm, gripper, gui, TimeMode.MESSAGE)
        if ds_agent is not None:
            for observer_name in observers.keys():
                ds_agent.add_signal(observer_name)
                world.connect(sim.observations[observer_name], ds_agent.inputs[observer_name])
        world.connect(inference_emitter[0], inference.command, emitter_wrapper=inference_emitter[1])
        if ds_agent is not None:
            world.connect(ds_writer_emitter[0], ds_agent.command, emitter_wrapper=ds_writer_emitter[1])

        sim_iter = world.start([*foreground_cs, *control_systems, ds_agent], gui)
        p_bar = tqdm.tqdm(total=simulation_time * num_iterations, unit='s')
        for _ in sim_iter:
            p_bar.n = round(sim.now(), 1)
            p_bar.refresh()


main_sim_cfg = cfn.Config(
    main_sim,
    mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
    action_decoder=positronic.cfg.policy.action.absolute_position,
    camera_fps=15,
    policy_fps=15,
    simulation_time=10,
    camera_dict={'image.handcam_left': 'handcam_left_ph', 'image.back_view': 'back_view_ph'},
    task='pick up the green cube and put in on top of the red cube',
)

main_sim_openpi_positronic = main_sim_cfg.override(
    policy=positronic.cfg.policy.policy.openpi,
    observation_encoder=positronic.cfg.policy.observation.openpi_positronic,
    action_decoder=positronic.cfg.policy.action.absolute_position,
    # We use 3 cameras not because we need it, but because Mujoco does not render
    # the second image when using only 2 cameras
    camera_dict={'image.wrist': 'handcam_left_ph', 'image.exterior': 'back_view_ph', 'image.agent_view': 'agentview'},
)

main_sim_openpi_droid = main_sim_cfg.override(
    # We use 3 cameras not because we need it, but because Mujoco does not render the second image when using
    # only 2 cameras.
    camera_dict={'image.wrist': 'handcam_back_ph', 'image.exterior': 'back_view_ph', 'image.agent_view': 'agentview'},
    policy=positronic.cfg.policy.policy.droid,
    observation_encoder=positronic.cfg.policy.observation.openpi_droid,
    action_decoder=positronic.cfg.policy.action.joint_delta,
    policy_fps=15,
)

main_sim_act = main_sim_cfg.override(
    policy=positronic.cfg.policy.policy.act,
    observation_encoder=positronic.cfg.policy.observation.eepose_mujoco,
    action_decoder=positronic.cfg.policy.action.absolute_position,
    # We use 3 cameras not because we need it, but because Mujoco does not render
    # the second image when using only 2 cameras
    camera_dict={
        'image.handcam_left': 'handcam_left_ph',
        'image.back_view': 'back_view_ph',
        'image.agent_view': 'agentview',
    },
)

openpi_droid = cfn.Config(
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
    policy=positronic.cfg.policy.policy.droid,
    observation_encoder=positronic.cfg.policy.observation.openpi_droid,
    action_decoder=positronic.cfg.policy.action.joint_delta,
    policy_fps=15,
)

openpi_positronic_real = openpi_droid.override(
    observation_encoder=positronic.cfg.policy.observation.openpi_positronic,
    action_decoder=positronic.cfg.policy.action.absolute_position,
)


# Separate function for [projects.scripts]
@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({
        'sim_act': main_sim_act,
        'sim_openpi_positronic': main_sim_openpi_positronic,
        'sim_openpi_droid': main_sim_openpi_droid,
        'sim_groot': main_sim_openpi_positronic.override(
            policy=positronic.cfg.policy.policy.groot,
            observation_encoder=positronic.cfg.policy.observation.groot_infer,
            action_decoder=positronic.cfg.policy.action.groot_infer,
        ),
        'droid_real': openpi_droid,
        'openpi_real': openpi_positronic_real,
        'groot_droid': openpi_droid.override(
            policy=positronic.cfg.policy.policy.groot,
            observation_encoder=positronic.cfg.policy.observation.groot_infer,
            action_decoder=positronic.cfg.policy.action.groot_infer,
        ),
        'sim_pnp': main_sim_cfg.override(
            loaders=positronic.cfg.simulator.multi_tote_loaders,
            task='pick up objects from the red tote and place them in the green tote',
            observers={},
        ),
    })


if __name__ == '__main__':
    _internal_main()
