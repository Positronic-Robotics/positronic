import asyncio

import hydra
from omegaconf import DictConfig
import torch
import rerun as rr

import ironic as ir
from .state import StateEncoder
from .policy import get_policy



def rerun_log_observation(ts, obs):
    rr.set_time_seconds('time', ts)

    def log_image(name, tensor, compress = True):
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
    rr.set_time_seconds('time', ts)
    for i, v in enumerate(action['target_robot_position'].translation):
        rr.log(f"action/translation_{i}", rr.Scalar(v))
    for i, v in enumerate(action['target_robot_position'].quaternion):
        rr.log(f"action/quaternion_{i}", rr.Scalar(v))
    rr.log(f"action/grip", rr.Scalar(action['target_grip']))



@ir.ironic_system(
    input_ports=['frame', 'start', 'stop'],
    input_props=['robot_data'],
    output_ports=['target_robot_position', 'target_grip']
)
class Inference(ir.ControlSystem):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.policy_factory = lambda: get_policy(cfg.checkpoint_path, cfg.get('policy_args', {}))
        self.cfg = cfg
        self.state_encoder = StateEncoder(hydra.utils.instantiate(cfg.state))
        self.action_decoder = hydra.utils.instantiate(cfg.action)
        self.policy = None
        self.running = False
        self.fps = ir.utils.FPSCounter("Inference")

    async def setup(self):
        """Initialize the policy"""
        self.policy = self.policy_factory()
        self.policy.to(self.cfg.device)

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
        for key in obs:
            obs[key] = obs[key].to(self.cfg.device)

        action = self.policy.select_action(obs).squeeze(0).cpu().numpy()
        action_dict = self.action_decoder.decode(action, robot_data)

        if self.cfg.rerun:
            rerun_log_observation(message.timestamp, obs)
            rerun_log_action(message.timestamp, action)

        write_ops = []
        for key, value in action_dict.items():
            out_port = getattr(self.outs, key)
            write_ops.append(out_port.write(ir.Message(value, message.timestamp)))

        await asyncio.gather(*write_ops)
        self.fps.tick()
