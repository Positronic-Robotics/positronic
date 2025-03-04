import asyncio
import logging
import sys

import hydra
from omegaconf import DictConfig
import rerun as rr

import drivers
import ironic as ir
from drivers.roboarm.franka import Franka
from drivers.gripper.dh import DHGripper
from inference import Inference

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("policy_runner.log", mode="w")])


# TODO: Extract to a proper control system that outputs pressed keys
@ir.ironic_system(output_ports=["start_policy", "stop_policy"])
class PolicyRunnerSystem(ir.ControlSystem):

    def __init__(self):
        super().__init__()
        self.policy_running = False
        # Set stdin in raw mode
        import termios
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty_settings = termios.tcgetattr(sys.stdin)
        tty_settings[3] = tty_settings[3] & ~(termios.ECHO | termios.ICANON)  # lflags
        termios.tcsetattr(sys.stdin, termios.TCSANOW, tty_settings)

        # Set stdin non-blocking
        import fcntl
        import os
        flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
        logging.info("Keyboard input initialized")

    async def step(self):
        """Handle user input for policy control"""
        try:
            import select
            # Check if there's data available to read (timeout=0 for non-blocking)
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:  # if sys.stdin has data
                char = sys.stdin.read(1)
                if char == 's':
                    self.policy_running = not self.policy_running
                    if self.policy_running:
                        await self.outs.start_policy.write(ir.Message(True))
                        logging.info('Started policy execution')
                    else:
                        await self.outs.stop_policy.write(ir.Message(True))
                        logging.info('Stopped policy execution')
        except Exception as e:
            logging.error(f"Error reading input: {e}")
        return ir.State.ALIVE

    async def cleanup(self):
        """Restore terminal settings"""
        import termios
        termios.tcsetattr(sys.stdin, termios.TCSANOW, self.old_settings)
        await super().cleanup()
        print('Policy runner stopped')


@hydra.main(version_base=None, config_path="configs", config_name="policy_runner")
def main(cfg: DictConfig):
    asyncio.run(async_main(cfg))


async def async_main(cfg: DictConfig):
    if cfg.rerun:
        rr.init("inference", spawn=False)
        if ':' in cfg.rerun:
            rr.connect(cfg.rerun)
        elif cfg.rerun is not None:
            rr.save(cfg.rerun)

    franka = Franka(cfg.franka.ip, cfg.franka.relative_dynamics_factor, cfg.franka.gripper_force)
    policy_runner = PolicyRunnerSystem()
    cam = drivers.from_config.sl_camera(cfg.camera)

    inference = Inference(cfg.inference)
    gripper = DHGripper("/dev/ttyUSB0")

    franka.bind(target_position=inference.outs.target_robot_position)
    gripper.bind(grip=inference.outs.target_grip)

    properties_to_dump = ir.utils.properties_dict(
        robot_joints=franka.outs.joint_positions,
        robot_position_translation=ir.utils.map_property(lambda t: t.translation, franka.outs.position),
        robot_position_quaternion=ir.utils.map_property(lambda t: t.rotation.as_quat, franka.outs.position),
        ext_force_ee=franka.outs.ext_force_ee,
        ext_force_base=franka.outs.ext_force_base,
        grip=gripper.outs.grip if gripper else None)

    inference.bind(
        frame=cam.outs.frame,
        robot_data=properties_to_dump,
        start=policy_runner.outs.start_policy,
        stop=policy_runner.outs.stop_policy,
    )

    system = ir.compose(policy_runner, franka, cam, inference, gripper)

    def rerun_cleanup():
        if cfg.rerun:
            rr.disconnect()

    await ir.utils.run_gracefully(system, extra_cleanup_fn=rerun_cleanup)


if __name__ == "__main__":
    main()
