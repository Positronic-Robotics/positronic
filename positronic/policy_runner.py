import asyncio
import logging
import sys

import rerun as rr
import fire

import positronic.cfg.env
import positronic.cfg.inference.inference
import ironic as ir

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


@ir.config(
    env=positronic.cfg.env.franka,
    inference=positronic.cfg.inference.inference.umi_inference,
    rerun=None
)
async def async_main(
        env: ir.ControlSystem,
        inference: ir.ControlSystem,
        rerun: str | None = None
):
    if rerun:
        rr.init("inference2", spawn=True)
        # if ':' in rerun:
        #     rr.connect(rerun)
        # elif rerun is not None:
        #     rr.save(rerun)

    policy_runner = PolicyRunnerSystem()

    env.bind(
        target_position=inference.outs.target_robot_position,
        target_grip=inference.outs.target_grip
    )

    properties_to_dump = ir.utils.properties_dict(
        robot_joints=env.outs.joint_positions,
        robot_position_translation=ir.utils.map_property(lambda t: t.translation, env.outs.position),
        robot_position_quaternion=ir.utils.map_property(lambda t: t.rotation.as_quat, env.outs.position),
        ext_force_ee=env.outs.ext_force_ee,
        ext_force_base=env.outs.ext_force_base,
        grip=env.outs.grip)

    inference.bind(
        frame=env.outs.frame,
        robot_data=properties_to_dump,
        start=policy_runner.outs.start_policy,
        stop=policy_runner.outs.stop_policy,
    )

    system = ir.compose(policy_runner, env, inference)

    def rerun_cleanup():
        if rerun:
            rr.disconnect()

    await ir.utils.run_gracefully(system, after_cleanup_fn=rerun_cleanup)


def main(*args, **kwargs):
    asyncio.run(run_with_config(*args, **kwargs))


async def run_with_config(*args, **kwargs):
    await async_main.override_and_instantiate(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(main)
