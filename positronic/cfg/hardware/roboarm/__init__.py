import configuronic as cfn

import positronic.cfg.hardware.motors

# The pose each arm is drawn around at the start of a trial, and — where its driver parks — the pose it is
# left at when the driver takes control and hands it back.
FRANKA_NOMINAL_JOINTS = [0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0]
YAM_NOMINAL_JOINTS = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]
SO101_NOMINAL_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0]
# How far, per joint, a start pose drawn around the Franka's nominal may sit from it.
FRANKA_JOINTS_SPREAD = [0.03, 0.05, 0.08, 0.08, 0.10, 0.10, 0.10]


@cfn.config(
    ip='172.168.0.2',
    relative_dynamics_factor=0.2,
    park_joints=FRANKA_NOMINAL_JOINTS,
    load=None,
    collision_coeff=2.0,
    manage_desk=True,
    reboot_on_safety_error=False,
)
def franka(
    ip: str,
    relative_dynamics_factor: float,
    park_joints: list[float],
    load: tuple | None = None,
    collision_coeff: float = 2.0,
    manage_desk: bool = True,
    reboot_on_safety_error: bool = False,
):
    from positronic.drivers.roboarm import franka  # noqa: F401

    return franka.Robot(
        ip=ip,
        relative_dynamics_factor=relative_dynamics_factor,
        park_joints=park_joints,
        load=load,
        collision_coeff=collision_coeff,
        manage_desk=manage_desk,
        reboot_on_safety_error=reboot_on_safety_error,
    )


franka_droid = franka.override(load=(0.9, [0.0, 0.0, 0.057], [0.002768, 0, 0, 0, 0.003149, 0, 0, 0, 0.000564]))


@cfn.config(ip='192.168.1.10', relative_dynamics_factor=0.5)
def kinova(ip, relative_dynamics_factor):
    from positronic.drivers.roboarm.kinova.driver import Robot

    return Robot(ip=ip, relative_dynamics_factor=relative_dynamics_factor)


@cfn.config(motor_bus=positronic.cfg.hardware.motors.so101_follower)
def so101(motor_bus):
    from positronic.drivers.roboarm.so101.driver import Robot

    return Robot(motor_bus=motor_bus)


@cfn.config(channel='can0', park_joints=YAM_NOMINAL_JOINTS, sim=False, base_pose=None)
def yam(channel: str, park_joints: list[float], sim: bool, base_pose):
    from positronic.drivers.roboarm.yam import Robot

    return Robot(channel, park_joints=park_joints, base_pose=base_pose, sim=sim)
