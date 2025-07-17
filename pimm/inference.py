import time
from typing import Any, Iterator, Mapping

from mujoco import Sequence
import numpy as np
import torch
import tqdm
import rerun as rr

import geom
import configuronic as cfgc
import ironic2 as ir
from pimm.drivers import roboarm
from pimm.drivers.sound import SoundSystem
from pimm.drivers.camera.linux_video import LinuxVideo
from pimm.drivers.gripper.dh import DHGripper
from pimm.drivers.webxr import WebXR
from pimm.simulator.mujoco.sim import MujocoCamera, MujocoFranka, MujocoGripper, MujocoSim
import positronic
from positronic.inference.action import ActionDecoder
from positronic.inference.inference import rerun_log_action, rerun_log_observation
from positronic.inference.state import StateEncoder
from positronic.simulator.mujoco.scene.transforms import MujocoSceneTransform

import pimm.cfg.hardware.gripper
import pimm.cfg.webxr
import pimm.cfg.hardware.camera
import pimm.cfg.sound
import pimm.cfg.simulator


class Inference:
    frame_readers : dict[str, ir.SignalReader[Mapping[str, np.ndarray]]] = {}
    robot_state : ir.SignalReader[roboarm.State] = ir.NoOpReader()
    gripper_state : ir.SignalReader[float] = ir.NoOpReader()

    robot_commands : ir.SignalEmitter[roboarm.command.CommandType] = ir.NoOpEmitter()
    target_grip_emitter : ir.SignalEmitter[float] = ir.NoOpEmitter()

    def __init__(
        self,
        state_encoder: StateEncoder,
        action_decoder: ActionDecoder,
        device: torch.device,
        policy,
        rerun_path: str | None = None,
    ):
        self.state_encoder = state_encoder
        self.action_decoder = action_decoder
        self.policy = policy
        self.policy.to(device)
        self.device = device
        self.rerun_path = rerun_path

    def run(self, should_stop: ir.SignalReader, clock: ir.Clock) -> Iterator[ir.Sleep]:
        frame_readers = {
            camera_name: ir.DefaultReader(ir.ValueUpdated(frame_reader), ({}, False))
            for camera_name, frame_reader in self.frame_readers.items()
        }

        fps_counter = ir.utils.RateCounter("Inference")
        reference_pose = None

        if self.rerun_path:
            rr.init("inference")
            rr.save(self.rerun_path)
        frame_time = 0
        while not should_stop.value:
            image_messages, is_updated = ir.utils.is_any_updated(frame_readers)
            if not is_updated:
                yield ir.Pass()
                continue
            print(1 / (clock.now() - frame_time))
            frame_time = clock.now()
            images = {k: v.data['image'] for k, v in image_messages.items()}

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
            target_pos: geom.Transform3D = action_dict['target_robot_position']

            roboarm_command = roboarm.command.CartesianMove(
                pose=target_pos,
            )

            if self.policy.chunk_start():
                print(f"Chunk start: {clock.now()}")
                reference_pose = target_pos

            self.robot_commands.emit(roboarm_command)

            self.target_grip_emitter.emit(action_dict['target_grip'].item())

            if self.rerun_path:
                rerun_log_observation(clock.now(), obs)
                rerun_log_action(clock.now(), action)

            fps_counter.tick()

            yield ir.Sleep(0.001)


def main(robot_arm: Any | None,
         gripper: DHGripper | None,
         webxr: WebXR,
         sound: SoundSystem | None,
         cameras: Mapping[str, LinuxVideo] | None,
         output_dir: str | None = None,
         fps: int = 30,
         stream_video_to_webxr: str | None = None,
         ):

    with ir.World() as world:
        data_collection = DataCollection(operator_position, output_dir, fps)
        cameras = cameras or {}
        for camera_name, camera in cameras.items():
            camera.frame, data_collection.frame_readers[camera_name] = world.mp_pipe()

        webxr.controller_positions, data_collection.controller_positions_reader = world.mp_pipe()
        webxr.buttons, data_collection.buttons_reader = world.mp_pipe()

        if stream_video_to_webxr is not None:
            emitter, reader = world.mp_pipe()
            cameras[stream_video_to_webxr].frame = ir.BroadcastEmitter([emitter, cameras[stream_video_to_webxr].frame])

            webxr.frame = ir.map(reader, lambda x: x['image'])

        world.start_in_subprocess(webxr.run, *[camera.run for camera in cameras.values()])

        if robot_arm is not None:
            robot_arm.state, data_collection.robot_state = world.zero_copy_sm()
            data_collection.robot_commands, robot_arm.commands = world.mp_pipe(1)
            world.start_in_subprocess(robot_arm.run)

        if gripper is not None:
            data_collection.target_grip_emitter, gripper.target_grip = world.mp_pipe(1)
            world.start_in_subprocess(gripper.run)

        if sound is not None:
            data_collection.sound_emitter, sound.wav_path = world.mp_pipe()
            world.start_in_subprocess(sound.run)

        dc_steps = iter(world.interleave(data_collection.run))

        while not world.should_stop:
            try:
                time.sleep(next(dc_steps).seconds)
            except StopIteration:
                break


def main_sim(
        mujoco_model_path: str,
        state_encoder: StateEncoder,
        action_decoder: ActionDecoder,
        policy,
        rerun_path: str,
        loaders: Sequence[MujocoSceneTransform] = (),
        fps: int = 30,
        device: torch.device = torch.device('cuda:0'),
        simulation_time: float = 10,
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
            camera.frame, inference.frame_readers[camera_name] = world.local_pipe()

        robot_arm.state, inference.robot_state = world.local_pipe()
        inference.robot_commands, robot_arm.commands = world.local_pipe()
        gripper.grip, inference.gripper_state = world.local_pipe()

        inference.target_grip_emitter, gripper.target_grip = world.local_pipe()


        sim_iter = world.interleave(
            sim.run,
            *[camera.run for camera in cameras.values()],
            robot_arm.run,
            gripper.run,
            inference.run,
        )

        sim_iter = iter(sim_iter)

        for _ in tqdm.tqdm(sim_iter):
            if sim.now() > simulation_time:
                break


main_cfg = cfgc.Config(
    main,
    robot_arm=None,
    gripper=pimm.cfg.hardware.gripper.dh_gripper,
    webxr=pimm.cfg.webxr.webxr,
    sound=pimm.cfg.sound.sound,
    cameras=cfgc.Config(
        dict,
        left=pimm.cfg.hardware.camera.arducam_left,
        right=pimm.cfg.hardware.camera.arducam_right,
    ),
)

import positronic.cfg.inference.action
import positronic.cfg.inference.state
import positronic.cfg.inference.policy


main_sim_cfg = cfgc.Config(
    main_sim,
    mujoco_model_path="positronic/assets/mujoco/franka_table.xml",
    loaders=pimm.cfg.simulator.stack_cubes_loaders,
    state_encoder=positronic.cfg.inference.state.end_effector_back_front,
    action_decoder=positronic.cfg.inference.action.relative_robot_position,
    policy=positronic.cfg.inference.policy.act,
    rerun_path="positronic/assets/rerun/franka_table_stack_cubes.rerun",
    device=torch.device('cuda'),
    simulation_time=10,
)

if __name__ == "__main__":
    # TODO: add ability to specify multiple targets in CLI
    cfgc.cli(main_sim_cfg)
