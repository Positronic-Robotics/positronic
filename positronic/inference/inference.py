import asyncio
from collections import deque

import numpy as np
import torch
import rerun as rr

import geom
import ironic as ir
from positronic.inference.action import ActionDecoder
from .state import StateEncoder


def rerun_log_observation(ts, obs):
    rr.set_time_nanos('time', nanos=ts)

    def log_image(name, tensor, compress: bool = True):
        tensor = tensor.squeeze(0)
        tensor = (tensor * 255).type(torch.uint8)
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
        rr_img = rr.Image(tensor)
        if compress:
            rr_img = rr_img.compress()
        rr.log(name, rr_img)

    for k, v in obs.items():
        if k.startswith("observation.images."):
            log_image(k, v)

    for i, state in enumerate(obs['observation.state'].squeeze(0)):
        rr.log(f"observation/state/{i}", rr.Scalar(state.item()))


def rerun_log_action(ts, action):
    rr.set_time_nanos('time', nanos=ts)
    for i, action in enumerate(action):
        rr.log(f"action/{i}", rr.Scalar(action))


@ir.ironic_system(
    input_ports=['frame', 'start', 'stop'],
    input_props=['robot_data'],
    output_ports=['target_robot_position', 'target_grip']
)
class Inference(ir.ControlSystem):
    def __init__(
            self,
            state_encoder: StateEncoder,
            policy: ir.ControlSystem,
            action_decoder: ActionDecoder,
            rerun: bool = False,
            device: str = 'cuda'
    ):
        super().__init__()

        self.policy = policy
        self.state_encoder = state_encoder
        self.action_decoder = action_decoder
        self.running = False
        self.rerun = rerun
        self.device = device
        self.fps = ir.utils.FPSCounter("Inference")

    async def setup(self):
        """Initialize the policy"""
        self.policy.to(self.device)
        self.reference_pose = None


    @ir.on_message("start")
    async def handle_start(self, message: ir.Message):
        """Handle policy start message"""
        self.policy.reset()
        self.fps.reset()
        self.running = True

    @ir.on_message("stop")
    async def handle_stop(self, message: ir.Message):
        """Handle policy stop message"""
        self.fps.report()
        self.running = False

    @ir.on_message("frame")
    async def handle_frame(self, message: ir.Message):
        """Process image and generate actions"""
        if not self.running:
            return

        robot_data = (await self.ins.robot_data()).data
        obs = self.state_encoder.encode(message.data, robot_data)

        if self.reference_pose is None:
            self.reference_pose = geom.Transform3D(
                robot_data['robot_position_translation'],
                geom.Rotation.from_quat(robot_data['robot_position_quaternion'])
            )

        robot_data['reference_robot_position_translation'] = self.reference_pose.translation
        robot_data['reference_robot_position_quaternion'] = self.reference_pose.rotation.as_quat

        for key in obs:
            obs[key] = obs[key].to(self.device)

        action = self.policy.select_action(obs).squeeze(0).cpu().numpy()
        action_dict = self.action_decoder.decode(action, robot_data)

        if self.policy.chunk_start():
            self.reference_pose = geom.Transform3D(
                robot_data['robot_position_translation'],
                geom.Rotation.from_quat(robot_data['robot_position_quaternion'])
            )

        if self.rerun:
            rerun_log_observation(message.timestamp, obs)
            rerun_log_action(message.timestamp, action)

        write_ops = []
        for key, value in action_dict.items():
            out_port = getattr(self.outs, key)
            write_ops.append(out_port.write(ir.Message(value, message.timestamp)))

        await asyncio.gather(*write_ops)
        self.fps.tick()