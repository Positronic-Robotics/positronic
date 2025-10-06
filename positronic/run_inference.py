from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
import torch
import tqdm

import pimm
import positronic.cfg.policy.action
import positronic.cfg.policy.observation
import positronic.cfg.policy.policy
import positronic.cfg.simulator
from positronic.dataset.ds_writer_agent import (
    DsWriterAgent,
    DsWriterCommand,
    Serializers,
    TimeMode,
)
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.drivers import roboarm
from positronic.gui.dpg import DearpyguiUi
from positronic.policy.action import ActionDecoder
from positronic.policy.observation import ObservationEncoder
from positronic.simulator.mujoco.observers import BodyDistance, StackingSuccess
from positronic.simulator.mujoco.sim import MujocoCamera, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform


def detect_device() -> str:
    """Select the best available torch device unless one is provided."""
    if torch.cuda.is_available():
        return 'cuda'

    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is not None:
        is_available = getattr(mps_backend, 'is_available', None)
        is_built = getattr(mps_backend, 'is_built', None)
        if callable(is_available) and is_available():
            if not callable(is_built) or is_built():
                return 'mps'

    return 'cpu'


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
        device: str,
        policy,
        inference_fps: int = 30,
        task: str | None = None,
    ):
        self.observation_encoder = observation_encoder
        self.action_decoder = action_decoder
        self.policy = policy.to(device)
        self.device = device
        self.inference_fps = inference_fps
        self.task = task

        self.frames = pimm.ReceiverDict(self)
        self.robot_state = pimm.ControlSystemReceiver(self)
        self.gripper_state = pimm.ControlSystemReceiver(self)
        self.robot_commands = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemEmitter(self)

        self.command = pimm.ControlSystemReceiver[InferenceCommand](self, maxsize=3)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        commands = pimm.DefaultReceiver(pimm.ValueUpdated(self.command), (None, False))
        running = False

        # TODO: We should emit new commands per frame, not per inference fps
        rate_limiter = pimm.RateLimiter(clock, hz=self.inference_fps)

        while not should_stop.value:
            command, is_updated = commands.value
            if is_updated:
                match command.type:
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
                if robot_state.status == roboarm.RobotStatus.MOVING:
                    # TODO: seems to be necessary to wait previous command to finish
                    continue

                inputs = {
                    'robot_state.q': robot_state.q,
                    'robot_state.dq': robot_state.dq,
                    'robot_state.ee_pose': Serializers.transform_3d(robot_state.ee_pose),
                    'grip': self.gripper_state.value,
                }
                frame_messages = {k: v.value for k, v in self.frames.items()}
                images = {f'image.{k}': v['image'] for k, v in frame_messages.items() if 'image' in v}
                if len(images) != len(self.frames):
                    continue
                inputs.update(images)

                obs = {}
                for key, val in self.observation_encoder.encode(inputs).items():
                    if isinstance(val, np.ndarray):
                        obs[key] = torch.from_numpy(val).to(self.device)
                    else:
                        obs[key] = torch.as_tensor(val).to(self.device)

                if self.task is not None:
                    obs['task'] = self.task

                action = self.policy.select_action(obs).squeeze(0).cpu().numpy()
                action_dict = self.action_decoder.decode(action, inputs)
                target_pos = action_dict['target_robot_position']

                roboarm_command = roboarm.command.CartesianMove(pose=target_pos)

                self.robot_commands.emit(roboarm_command)
                self.target_grip.emit(action_dict['target_grip'].item())
            except pimm.NoValueException:
                continue
            finally:
                yield pimm.Sleep(rate_limiter.wait_time())


# TODO: Inference for the real robot


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
    device: str | None = None,
    output_dir: str | None = None,
    show_gui: bool = False,
    num_iterations: int = 1,
):
    device = device or detect_device()
    observers = {
        'box_distance': BodyDistance('box_0_body', 'box_1_body'),
        'stacking_success': StackingSuccess('box_0_body', 'box_1_body', 'hand_ph'),
    }
    sim = MujocoSim(mujoco_model_path, loaders, observers=observers)
    robot_arm = MujocoFranka(sim, suffix='_ph')
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    cameras = {
        name: MujocoCamera(sim.model, sim.data, orig_name, (320, 240), fps=camera_fps)
        for name, orig_name in camera_dict.items()
    }
    inference = Inference(observation_encoder, action_decoder, device, policy, policy_fps, task)
    control_systems = list(cameras.values()) + [sim, robot_arm, gripper, inference]

    meta = {
        'inference.device': device,
        'inference.mujoco_model_path': mujoco_model_path,
        'inference.policy_fps': policy_fps,
        'inference.simulation_time': simulation_time,
    }
    if task is not None:
        meta['inference.task'] = task

    gui = DearpyguiUi() if show_gui else None

    writer_cm = LocalDatasetWriter(Path(output_dir)) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        ds_agent = None
        if dataset_writer is not None:
            signals_spec = dict.fromkeys(cameras.keys())
            signals_spec['robot_state'] = Serializers.robot_state
            signals_spec['robot_commands'] = Serializers.robot_command
            signals_spec['target_grip'] = None
            signals_spec['grip'] = None
            for observer_name in observers.keys():
                signals_spec[observer_name] = None
            ds_agent = DsWriterAgent(dataset_writer, signals_spec, time_mode=TimeMode.MESSAGE)
            control_systems.append(ds_agent)

            # TODO: It seems that the right way is for inference to report its inputs,
            # rather than collecting them directly from "hardware".
            world.connect(robot_arm.state, ds_agent.inputs['robot_state'])
            world.connect(inference.robot_commands, ds_agent.inputs['robot_commands'])
            world.connect(gripper.grip, ds_agent.inputs['grip'])
            world.connect(inference.target_grip, ds_agent.inputs['target_grip'])
            for observer_name in observers.keys():
                world.connect(sim.observations[observer_name], ds_agent.inputs[observer_name])

        cameras = cameras or {}
        for camera_name, camera in cameras.items():
            world.connect(camera.frame, inference.frames[camera_name])
            if ds_agent is not None:
                world.connect(camera.frame, ds_agent.inputs[camera_name])
            if gui is not None:
                world.connect(camera.frame, gui.cameras[camera_name])

        world.connect(robot_arm.state, inference.robot_state)
        world.connect(inference.robot_commands, robot_arm.commands)
        world.connect(gripper.grip, inference.gripper_state)
        world.connect(inference.target_grip, gripper.target_grip)

        driver = Driver(num_iterations, simulation_time, meta=meta)
        if ds_agent is not None:
            world.connect(driver.ds_commands, ds_agent.command)
        world.connect(driver.inf_commands, inference.command)

        sim_iter = world.start([driver, *control_systems], gui)

        p_bar = tqdm.tqdm(total=simulation_time * num_iterations, unit='s')
        for _ in sim_iter:
            p_bar.n = round(sim.now(), 1)
            p_bar.refresh()


main_sim_cfg = cfn.Config(
    main_sim,
    mujoco_model_path='positronic/assets/mujoco/franka_table.xml',
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
    observation_encoder=positronic.cfg.policy.observation.pi0,
    action_decoder=positronic.cfg.policy.action.absolute_position,
    policy=positronic.cfg.policy.policy.pi0,
    camera_fps=60,
    policy_fps=15,
    device=None,
    simulation_time=10,
    camera_dict={
        'handcam_left': 'handcam_left_ph',
        'back_view': 'back_view_ph',
    },
    task='pick up the green cube and put in on top of the red cube',
)

main_sim_pi0 = main_sim_cfg.override(
    policy=positronic.cfg.policy.policy.pi0,
    observation_encoder=positronic.cfg.policy.observation.pi0,
    action_decoder=positronic.cfg.policy.action.absolute_position,
    camera_dict={
        'left': 'handcam_left_ph',
        'side': 'back_view_ph',
    },
)

main_sim_act = main_sim_cfg.override(
    policy=positronic.cfg.policy.policy.act,
    observation_encoder=positronic.cfg.policy.observation.franka_mujoco_stackcubes,
    action_decoder=positronic.cfg.policy.action.absolute_position,
    camera_dict={
        'handcam_left': 'handcam_left_ph',
        'back_view': 'back_view_ph',
        'agent_view': 'agentview',
    },
)

if __name__ == '__main__':
    cfn.cli({
        'sim_pi0': main_sim_pi0,
        'sim_act': main_sim_act,
    })
