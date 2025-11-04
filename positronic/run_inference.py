import select
import sys
import termios
import time
import tty
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any

import configuronic as cfn
import rerun as rr
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
from positronic import wire
from positronic.dataset.ds_writer_agent import DsWriterCommand, Serializers, TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.drivers import roboarm
from positronic.gui.dpg import DearpyguiUi
from positronic.policy.action import ActionDecoder
from positronic.policy.observation import ObservationEncoder
from positronic.simulator.mujoco.observers import BodyDistance, StackingSuccess
from positronic.simulator.mujoco.sim import MujocoCameras, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform

rr.init('positronic')
rr.save('positronic_inference.rrd')
# rr.spawn()


class InferenceCommandType(Enum):
    """Commands for the inference."""

    START = 'start'
    STOP = 'stop'
    RESET = 'reset'


@dataclass
class InferenceCommand:
    """Command for the inference."""

    type: InferenceCommandType
    payload: Any | None = None

    @classmethod
    def START(cls) -> 'InferenceCommand':
        """Convenience method for creating a START command."""
        return cls(InferenceCommandType.START, None)

    @classmethod
    def STOP(cls) -> 'InferenceCommand':
        """Convenience method for creating a STOP command."""
        return cls(InferenceCommandType.STOP, None)

    @classmethod
    def RESET(cls) -> 'InferenceCommand':
        """Convenience method for creating a RESET command."""
        return cls(InferenceCommandType.RESET, None)


class Inference(pimm.ControlSystem):
    def __init__(
        self,
        observation_encoder: ObservationEncoder,
        action_decoder: ActionDecoder,
        policy,
        inference_fps: int = 30,
        task: str | None = None,
        simulate_timeout: bool = False,
    ):
        self.observation_encoder = observation_encoder
        self.action_decoder = action_decoder
        self.policy = policy
        self.inference_fps = inference_fps
        self.task = task
        self.simulate_timeout = simulate_timeout

        self.frames = pimm.ReceiverDict(self)
        self.robot_state = pimm.ControlSystemReceiver(self)
        self.gripper_state = pimm.ControlSystemReceiver(self)
        self.robot_commands = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemEmitter(self)

        self.command = pimm.ControlSystemReceiver[InferenceCommand](self, default=None, maxsize=3)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        running = False

        # TODO: We should emit new commands per frame, not per inference fps
        rate_limiter = pimm.RateLimiter(clock, hz=self.inference_fps)

        while not should_stop.value:
            command_msg = self.command.read()
            if command_msg.updated:
                match command_msg.data.type:
                    case InferenceCommandType.START:
                        running = True
                        rate_limiter.reset()
                    case InferenceCommandType.STOP:
                        running = False
                    case InferenceCommandType.RESET:
                        self.robot_commands.emit(roboarm.command.Reset())
                        self.target_grip.emit(0.0)
                        self.policy.reset()
                        running = False
                        yield pimm.Pass()

            try:
                if not running:
                    continue

                robot_state = self.robot_state.value
                inputs = {
                    'robot_state.q': robot_state.q,
                    'robot_state.dq': robot_state.dq,
                    'robot_state.ee_pose': Serializers.transform_3d(robot_state.ee_pose),
                    'grip': self.gripper_state.value,
                }
                frame_messages = {k: v.value for k, v in self.frames.items()}
                # Extract array from NumpySMAdapter
                images = {k: v.array for k, v in frame_messages.items()}
                if len(images) != len(self.frames):
                    continue
                inputs.update(images)

                obs = self.observation_encoder.encode(inputs)
                if self.task is not None:
                    obs['task'] = self.task

                start = time.monotonic()
                action = self.policy.select_action(obs)
                roboarm_command, target_grip = self.action_decoder.decode(action, inputs)

                duration = time.monotonic() - start
                if self.simulate_timeout:
                    yield pimm.Sleep(duration)

                self.robot_commands.emit(roboarm_command)
                self.target_grip.emit(target_grip)
            except pimm.NoValueException:
                continue
            finally:
                yield pimm.Sleep(rate_limiter.wait_time())


class KeyboardControl(pimm.ControlSystem):
    def __init__(self):
        self.keyboard_inputs = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        # Check if stdin is a TTY before attempting terminal operations
        if not sys.stdin.isatty():
            print('WARNING: KeyboardControl cannot read input - stdin is not a terminal', file=sys.stderr)
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        try:
            while not should_stop.value:
                r, _, _ = select.select([sys.stdin], [], [], 0.0)
                if r:
                    self.keyboard_inputs.emit(sys.stdin.read(1))
                yield pimm.Sleep(0.01)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def key_to_inf_command(key: str) -> InferenceCommand | None:
    """Map keyboard inputs to inference commands, filtering out unmapped keys."""
    if key == 's':
        print('Starting inference...')
        return InferenceCommand.START()
    elif key == 'p':
        print('Stopping inference...')
        return InferenceCommand.STOP()
    elif key == 'r':
        print('Resetting...')
        return InferenceCommand.RESET()
    return None


def key_to_ds_command(key: str) -> DsWriterCommand | None:
    """Map keyboard inputs to dataset writer commands, filtering out unmapped keys."""
    if key == 's':
        return DsWriterCommand.START()
    elif key == 'p':
        return DsWriterCommand.STOP()
    elif key == 'r':
        return DsWriterCommand.STOP()
    return None


def main(
    robot_arm: pimm.ControlSystem,
    gripper: pimm.ControlSystem,
    cameras: dict[str, pimm.ControlSystem],
    observation_encoder: ObservationEncoder,
    action_decoder: ActionDecoder,
    policy,
    policy_fps: int = 15,
    task: str | None = None,
    output_dir: str | None = None,
    show_gui: bool = False,
):
    """Runs inference on real hardware."""
    inference = Inference(observation_encoder, action_decoder, policy, policy_fps, task)

    # Convert camera instances to emitters for wire()
    camera_instances = cameras
    camera_emitters = {name: cam.frame for name, cam in camera_instances.items()}

    gui = DearpyguiUi() if show_gui else None
    writer_cm = LocalDatasetWriter(pos3.upload(output_dir)) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World() as world:
        ds_agent = wire.wire(
            world, inference, dataset_writer, camera_emitters, robot_arm, gripper, gui, TimeMode.MESSAGE
        )

        keyboard = KeyboardControl()
        print('Keyboard controls: [s]tart, sto[p], [r]eset')

        world.connect(keyboard.keyboard_inputs, inference.command, emitter_wrapper=pimm.map(key_to_inf_command))
        if ds_agent is not None:
            world.connect(keyboard.keyboard_inputs, ds_agent.command, emitter_wrapper=pimm.map(key_to_ds_command))

        bg_cs = [*camera_instances.values(), robot_arm, gripper, ds_agent, gui]

        for cmd in world.start([inference, keyboard], bg_cs):
            time.sleep(cmd.seconds)


class Driver(pimm.ControlSystem):
    """Control system that orchestrates inference episodes by sending start/stop commands."""

    def __init__(self, num_iterations: int, simulation_time: float, meta: dict):
        self.num_iterations = num_iterations
        self.simulation_time = simulation_time
        self.ds_commands = pimm.ControlSystemEmitter(self)
        self.inf_commands = pimm.ControlSystemEmitter(self)
        self.meta = meta

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        for i in range(self.num_iterations):
            meta = self.meta.copy()
            meta['inference.iteration'] = i

            self.ds_commands.emit(DsWriterCommand.START(meta))
            self.inf_commands.emit(InferenceCommand.START())
            yield pimm.Sleep(self.simulation_time)
            self.ds_commands.emit(DsWriterCommand.STOP())
            self.inf_commands.emit(InferenceCommand.RESET())
            yield pimm.Sleep(0.2)  # Let the things propagate


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
    output_dir: str | None = None,
    show_gui: bool = False,
    num_iterations: int = 1,
    simulate_timeout: bool = False,
):
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

    meta = {
        'inference.mujoco_model_path': mujoco_model_path,
        'inference.policy_fps': policy_fps,
        'inference.simulation_time': simulation_time,
    }
    if task is not None:
        meta['inference.task'] = task

    gui = DearpyguiUi() if show_gui else None

    writer_cm = LocalDatasetWriter(pos3.upload(output_dir)) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        ds_agent = wire.wire(world, inference, dataset_writer, cameras, robot_arm, gripper, gui, TimeMode.MESSAGE)
        if ds_agent is not None:
            for observer_name in observers.keys():
                ds_agent.add_signal(observer_name)
                world.connect(sim.observations[observer_name], ds_agent.inputs[observer_name])
        driver = Driver(num_iterations, simulation_time, meta=meta)
        world.connect(driver.inf_commands, inference.command)
        if ds_agent is not None:
            world.connect(driver.ds_commands, ds_agent.command)
        sim_iter = world.start([driver, *control_systems, ds_agent], gui)
        p_bar = tqdm.tqdm(total=simulation_time * num_iterations, unit='s')
        for _ in sim_iter:
            p_bar.n = round(sim.now(), 1)
            p_bar.refresh()


main_sim_cfg = cfn.Config(
    main_sim,
    mujoco_model_path='positronic/assets/mujoco/franka_table.xml',
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
    cfn.cli({
        'sim_act': main_sim_act,
        'sim_openpi_positronic': main_sim_openpi_positronic,
        'sim_openpi_droid': main_sim_openpi_droid,
        'droid_real': openpi_droid,
        'openpi_real': openpi_positronic_real,
    })


if __name__ == '__main__':
    _internal_main()
