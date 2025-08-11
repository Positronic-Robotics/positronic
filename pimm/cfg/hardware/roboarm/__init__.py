import configuronic as cfn

import pimm.cfg.hardware.motors


@cfn.config(ip="172.168.0.2",
            relative_dynamics_factor=0.2,
            cartesian_mode="positronic",
            home_joints=[0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0])
def franka(ip: str, relative_dynamics_factor: float, cartesian_mode: str, home_joints: list[float]):
    from pimm.drivers.roboarm.franka import Robot, CartesianMode  # noqa: F401
    return Robot(ip=ip,
                 relative_dynamics_factor=relative_dynamics_factor,
                 cartesian_mode=CartesianMode(cartesian_mode),
                 home_joints=home_joints)


@cfn.config(ip='192.168.1.10', relative_dynamics_factor=0.5)
def kinova(ip, relative_dynamics_factor):
    from pimm.drivers.roboarm.kinova.driver import Robot

    return Robot(ip=ip, relative_dynamics_factor=relative_dynamics_factor)


@cfn.config(motor_bus=pimm.cfg.hardware.motors.so101_leader)
def so101(motor_bus):
    from pimm.drivers.roboarm.so101.driver import Robot
    return Robot(motor_bus=motor_bus)
