from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import tqdm
import rerun as rr

import configuronic as cfgc
import ironic2 as ir
from pimm.drivers import roboarm
from pimm.drivers.camera.linux_video import LinuxVideo
from pimm.drivers.gripper.dh import DHGripper
from pimm.simulator.mujoco.sim import MujocoCamera, MujocoFranka, MujocoGripper, MujocoSim
from positronic.inference.action import ActionDecoder
from positronic.inference.inference import rerun_log_action, rerun_log_observation
from positronic.inference.state import StateEncoder
from positronic.simulator.mujoco.scene.transforms import MujocoSceneTransform

import pimm.cfg.hardware.roboarm
import pimm.cfg.hardware.gripper
import pimm.cfg.hardware.camera
import pimm.cfg.simulator
import positronic.cfg.inference.action
import positronic.cfg.inference.state
import positronic.cfg.inference.policy


class Inference:
    frames : dict[str, ir.SignalReader[Mapping[str, np.ndarray]]] = {}
    robot_state : ir.SignalReader[roboarm.State] = ir.NoOpReader()
    gripper_state : ir.SignalReader[float] = ir.NoOpReader()

    robot_commands : ir.SignalEmitter[roboarm.command.CommandType] = ir.NoOpEmitter()
    target_grip : ir.SignalEmitter[float] = ir.NoOpEmitter()

    def __init__(
        self,
        state_encoder: StateEncoder,
        action_decoder: ActionDecoder,
        device: str,
        policy,
        rerun_path: str | None = None,
    ):
        self.state_encoder = state_encoder
        self.action_decoder = action_decoder
        self.policy = policy.to(device)
        self.device = device
        self.rerun_path = rerun_path

    def run(self, should_stop: ir.SignalReader, clock: ir.Clock) -> Iterator[ir.Sleep]:
        frames = {
            camera_name: ir.DefaultReader(ir.ValueUpdated(frame), ({}, False))
            for camera_name, frame in self.frames.items()
        }

        reference_pose = None

        if self.rerun_path:
            rr.init("inference")
            rr.save(self.rerun_path)

        while not should_stop.value:
            frame_messages, is_updated = ir.utils.is_any_updated(frames)
            if not is_updated:
                yield ir.Sleep(0.001)
                continue

            images = {k: v.data['image'] for k, v in frame_messages.items()}

            with self.robot_state.zc_lock():
                if reference_pose is None:
                    reference_pose = self.robot_state.value.ee_pose.copy()

                inputs = {
                    'robot_position_translation': self.robot_state.value.ee_pose.translation,
                    'robot_position_rotation': self.robot_state.value.ee_pose.rotation.as_quat,
                    'robot_joints': self.robot_state.value.q,
                    'grip': self.gripper_state.value,
                    'reference_robot_position_translation': reference_pose.translation,
                    'reference_robot_position_quaternion': reference_pose.rotation.as_quat
                }
            obs = self.state_encoder.encode(images, inputs)
            for key in obs:
                obs[key] = obs[key].to(self.device)

            action = self.policy.select_action(obs).squeeze(0).cpu().numpy()
            action_dict = self.action_decoder.decode(action, inputs)
            target_pos = action_dict['target_robot_position']

            roboarm_command = roboarm.command.CartesianMove(pose=target_pos)

            # TODO: this should be inside the policy
            if self.policy.chunk_start():
                reference_pose = target_pos

            self.robot_commands.emit(roboarm_command)
            self.target_grip.emit(action_dict['target_grip'].item())

            if self.rerun_path:
                rerun_log_observation(clock.now(), obs)
                rerun_log_action(clock.now(), action)

            yield ir.Sleep(0.001)


def main(robot_arm: Any | None,
         gripper: DHGripper | None,
         cameras: Mapping[str, LinuxVideo] | None,
         state_encoder: StateEncoder,
         action_decoder: ActionDecoder,
         policy,
         rerun_path: str | None = None,
         device: str = 'cuda',
         ):

    with ir.World() as world:
        inference = Inference(state_encoder, action_decoder, device, policy, rerun_path)
        cameras = cameras or {}
        for camera_name, camera in cameras.items():
            camera.frame, inference.frames[camera_name] = world.mp_pipe()

        world.start_in_subprocess(*[camera.run for camera in cameras.values()])

        if robot_arm is not None:
            robot_arm.state, inference.robot_state = world.zero_copy_sm()
            inference.robot_commands, robot_arm.commands = world.mp_pipe()
            world.start_in_subprocess(robot_arm.run)

        if gripper is not None:
            inference.target_grip, gripper.target_grip = world.mp_pipe()
            world.start_in_subprocess(gripper.run)

        world.run(inference.run)


def main_sim(
        mujoco_model_path: str,
        state_encoder: StateEncoder,
        action_decoder: ActionDecoder,
        policy,
        rerun_path: str,
        loaders: Sequence[MujocoSceneTransform],
        fps: int,
        device: str,
        simulation_time: float,
):
    sim = MujocoSim(mujoco_model_path, loaders)
    robot_arm = MujocoFranka(sim, suffix='_ph')
    cameras = {
        'image.back': MujocoCamera(sim.model, sim.data, 'handcam_back_ph', (1280, 720), fps=fps),
        'image.front': MujocoCamera(sim.model, sim.data, 'handcam_front_ph', (1280, 720), fps=fps),
    }
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    inference = Inference(state_encoder, action_decoder, device, policy, rerun_path)

    with ir.World(clock=sim) as world:
        cameras = cameras or {}
        for camera_name, camera in cameras.items():
            camera.frame, inference.frames[camera_name] = world.local_pipe()

        robot_arm.state, inference.robot_state = world.local_pipe()
        inference.robot_commands, robot_arm.commands = world.local_pipe()
        gripper.grip, inference.gripper_state = world.local_pipe()

        inference.target_grip, gripper.target_grip = world.local_pipe()

        sim_iter = world.interleave(
            sim.run,
            *[camera.run for camera in cameras.values()],
            robot_arm.run,
            gripper.run,
            inference.run,
        )

        p_bar = tqdm.tqdm(total=simulation_time, unit='s')
        for _ in sim_iter:
            p_bar.n = round(sim.now(), 1)
            p_bar.refresh()
            if sim.now() > simulation_time:
                break


main_cfg = cfgc.Config(
    main,
    robot_arm=pimm.cfg.hardware.roboarm.franka,
    gripper=pimm.cfg.hardware.gripper.dh_gripper,
    state_encoder=positronic.cfg.inference.state.end_effector_224,
    action_decoder=positronic.cfg.inference.action.umi_relative,
    policy=positronic.cfg.inference.policy.act,
    cameras={
        'left': pimm.cfg.hardware.camera.arducam_left,
        'right': pimm.cfg.hardware.camera.arducam_right,
    },
    rerun_path="inference.rrd",
    device='cuda',
)


main_sim_cfg = cfgc.Config(
    main_sim,
    mujoco_model_path="positronic/assets/mujoco/franka_table.xml",
    loaders=pimm.cfg.simulator.stack_cubes_loaders,
    state_encoder=positronic.cfg.inference.state.end_effector_back_front,
    action_decoder=positronic.cfg.inference.action.relative_robot_position,
    policy=positronic.cfg.inference.policy.act,
    rerun_path="inference.rrd",
    fps=60,
    device='cuda',
    simulation_time=10,
)

if __name__ == "__main__":
    # TODO: add ability to specify multiple targets in CLI
    cfgc.cli(main_sim_cfg)
