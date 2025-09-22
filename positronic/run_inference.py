from contextlib import nullcontext
from pathlib import Path
from typing import Iterator, Mapping, Sequence

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
    DsWriterCommandType,
    Serializers,
    TimeMode,
)
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.drivers import roboarm
from positronic.gui.dpg import DearpyguiUi
from positronic.policy.action import ActionDecoder
from positronic.policy.observation import ObservationEncoder
from positronic.simulator.mujoco.sim import MujocoCamera, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform


class Inference(pimm.ControlSystem):

    def __init__(self,
                 state_encoder: ObservationEncoder,
                 action_decoder: ActionDecoder,
                 device: str,
                 policy,
                 inference_fps: int = 30,
                 task: str | None = None):
        self.state_encoder = state_encoder
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

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        frames = {camera_name: pimm.DefaultReceiver(frame, {}) for camera_name, frame in self.frames.items()}

        rate_limiter = pimm.RateLimiter(clock, hz=self.inference_fps)

        while not should_stop.value:
            frame_messages = {k: v.read() for k, v in frames.items()}
            if not all('image' in v.data for v in frame_messages.values()):
                yield pimm.Sleep(0.001)
                continue

            images = {f"image.{k}": v.data['image'] for k, v in frame_messages.items()}

            robot_state = self.robot_state.read()
            if robot_state is None:
                yield pimm.Sleep(0.001)
                continue

            gripper_state = self.gripper_state.read()
            if gripper_state is None:
                yield pimm.Sleep(0.001)
                continue

            robot_state = robot_state.data

            if robot_state.status == roboarm.RobotStatus.MOVING:
                # TODO: seems to be necessary to wait previous command to finish
                yield pimm.Sleep(0.001)
                continue

            inputs = {
                'robot_position_translation': robot_state.ee_pose.translation,
                'robot_position_quaternion': robot_state.ee_pose.rotation.as_quat,
                'robot_joints': robot_state.q,
                'grip': gripper_state.data,
            }
            obs = {}
            for key, val in self.state_encoder.encode(images, inputs).items():
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

            yield pimm.Sleep(rate_limiter.wait_time())


# TODO: Inference for the real robot


def main_sim(
    mujoco_model_path: str,
    state_encoder: ObservationEncoder,
    action_decoder: ActionDecoder,
    policy,
    loaders: Sequence[MujocoSceneTransform],
    camera_fps: int,
    policy_fps: int,
    device: str,
    simulation_time: float,
    camera_dict: Mapping[str, str],
    task: str | None,
    output_dir: str | None = None,
    show_gui: bool = False,
):
    sim = MujocoSim(mujoco_model_path, loaders)
    robot_arm = MujocoFranka(sim, suffix='_ph')
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    cameras = {
        name: MujocoCamera(sim.model, sim.data, orig_name, (1280, 720), fps=camera_fps)
        for name, orig_name in camera_dict.items()
    }
    inference = Inference(state_encoder, action_decoder, device, policy, policy_fps, task)
    control_systems = list(cameras.values()) + [sim, robot_arm, gripper, inference]

    gui = None
    if show_gui:
        gui = DearpyguiUi()

    writer_cm = LocalDatasetWriter(Path(output_dir)) if output_dir is not None else nullcontext(None)
    with writer_cm as dataset_writer, pimm.World(clock=sim) as world:
        ds_agent = None
        if dataset_writer is not None:
            signals_spec = {k: None for k in cameras.keys()}
            signals_spec['robot_state'] = Serializers.robot_state
            signals_spec['robot_commands'] = Serializers.robot_command
            signals_spec['target_grip'] = None
            signals_spec['grip'] = None
            ds_agent = DsWriterAgent(dataset_writer, signals_spec, time_mode=TimeMode.MESSAGE)
            control_systems.append(ds_agent)

            world.connect(robot_arm.state, ds_agent.inputs['robot_state'])
            world.connect(inference.robot_commands, ds_agent.inputs['robot_commands'])
            world.connect(gripper.grip, ds_agent.inputs['grip'])
            world.connect(inference.target_grip, ds_agent.inputs['target_grip'])

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

        commands = world.pair(ds_agent.command) if ds_agent else None
        sim_iter = world.start(control_systems, gui)

        if commands is not None:
            commands.emit(DsWriterCommand(type=DsWriterCommandType.START_EPISODE))
        p_bar = tqdm.tqdm(total=simulation_time, unit='s')
        for _ in sim_iter:
            p_bar.n = round(sim.now(), 1)
            p_bar.refresh()
            if sim.now() > simulation_time:
                if commands is not None:
                    commands.emit(DsWriterCommand(type=DsWriterCommandType.STOP_EPISODE))
                world.request_stop()


main_sim_cfg = cfn.Config(
    main_sim,
    mujoco_model_path="positronic/assets/mujoco/franka_table.xml",
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
    state_encoder=positronic.cfg.policy.observation.pi0,
    action_decoder=positronic.cfg.policy.action.absolute_position,
    policy=positronic.cfg.policy.policy.pi0,
    camera_fps=60,
    policy_fps=15,
    device='cuda',
    simulation_time=10,
    camera_dict={
        'handcam_left': 'handcam_left_ph',
        'back_view': 'back_view_ph',
    },
    task="pick up the green cube and put in on top of the red cube",
)

main_sim_pi0 = main_sim_cfg.override(
    policy=positronic.cfg.policy.policy.pi0,
    state_encoder=positronic.cfg.policy.observation.pi0,
    action_decoder=positronic.cfg.policy.action.absolute_position,
    camera_dict={
        'left': 'handcam_left_ph',
        'side': 'back_view_ph',
    },
)

main_sim_act = main_sim_cfg.override(
    policy=positronic.cfg.policy.policy.act,
    state_encoder=positronic.cfg.policy.observation.franka_mujoco_stackcubes,
    action_decoder=positronic.cfg.policy.action.absolute_position,
    camera_dict={
        'handcam_left': 'handcam_left_ph',
        'back_view': 'back_view_ph',
    },
)

if __name__ == "__main__":
    cfn.cli({
        "sim_pi0": main_sim_pi0,
        "sim_act": main_sim_act,
    })
