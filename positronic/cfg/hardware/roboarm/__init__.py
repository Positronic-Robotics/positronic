import configuronic as cfn

import positronic.cfg.hardware.motors


@cfn.config(ip='172.168.0.2', relative_dynamics_factor=0.2, home_joints=[0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0])
def franka(ip: str, relative_dynamics_factor: float, home_joints: list[float]):
    from positronic.drivers.roboarm import franka  # noqa: F401

    return franka.Robot(ip=ip, relative_dynamics_factor=relative_dynamics_factor, home_joints=home_joints)


@cfn.config(ip='192.168.1.10', relative_dynamics_factor=0.5)
def kinova(ip, relative_dynamics_factor):
    from positronic.drivers.roboarm.kinova.driver import Robot

    return Robot(ip=ip, relative_dynamics_factor=relative_dynamics_factor)


@cfn.config(motor_bus=positronic.cfg.hardware.motors.so101_follower)
def so101(motor_bus):
    from positronic.drivers.roboarm.so101.driver import Robot

    return Robot(motor_bus=motor_bus)
