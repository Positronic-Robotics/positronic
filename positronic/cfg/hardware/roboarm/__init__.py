import configuronic as cfn

import positronic.cfg.hardware.motors
from positronic.drivers.roboarm import command

# The pose each arm is drawn around at the start of a trial. Where a driver parks is its own and lives with it.
FRANKA_NOMINAL_JOINTS = [0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0]
YAM_NOMINAL_JOINTS = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]
SO101_NOMINAL_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0]
# The Trossen rests on the lower limit of joints 1 and 2, where half the directions out of it have no
# solution at all. Its start pose is mid-range on every joint instead, end effector at [0.503, 0, 0.232].
TROSSEN_NOMINAL_JOINTS = [0.0, 1.571, 1.178, 0.0, 0.0, 0.0]
# Where the Trossen rests: every joint at zero, which is the lower limit of joints 1 and 2. The arm holds
# itself there without the controller, so it is where a session leaves it.
TROSSEN_PARK_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# How far, per joint, a start pose drawn around the Franka's nominal may sit from it.
FRANKA_JOINTS_SPREAD = [0.03, 0.05, 0.08, 0.08, 0.10, 0.10, 0.10]
# The gains DROID's Franka ran, which its pretrained checkpoints were trained under.
DROID_IMPEDANCE = command.Impedance(
    kq=(40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0),
    kqd=(4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0),
    kx=(750.0, 750.0, 750.0, 15.0, 15.0, 15.0),
    kxd=(37.0, 37.0, 37.0, 2.0, 2.0, 2.0),
)


def droid_start_pose() -> command.JointPosition:
    """The command a DROID trial opens with: joints drawn afresh around the Franka's nominal, under DROID's gains."""
    return command.sampled_joints(FRANKA_NOMINAL_JOINTS, FRANKA_JOINTS_SPREAD, DROID_IMPEDANCE)


@cfn.config(
    ip='172.168.0.2',
    relative_dynamics_factor=0.2,
    load=None,
    collision_coeff=2.0,
    manage_desk=True,
    reboot_on_safety_error=False,
)
def franka(
    ip: str,
    relative_dynamics_factor: float,
    load: tuple | None = None,
    collision_coeff: float = 2.0,
    manage_desk: bool = True,
    reboot_on_safety_error: bool = False,
):
    from positronic.drivers.roboarm import franka  # noqa: F401

    return franka.Robot(
        ip=ip,
        relative_dynamics_factor=relative_dynamics_factor,
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


@cfn.config(channel='can0', sim=False, base_pose=None)
def yam(channel: str, sim: bool, base_pose):
    from positronic.drivers.roboarm.yam import Robot

    return Robot(channel, base_pose=base_pose, sim=sim)


@cfn.config(ip='192.168.1.4')
def trossen(ip: str):
    from positronic.drivers.roboarm.trossen import Robot

    return Robot(ip=ip)


# What the controller cancels of the gripper's own friction on a leader the operator holds, in N. Felt at
# the rig on the leader at 192.168.1.2, which is calibrated at 5.77: at 10.02 the trigger moves under a
# finger and still holds where it is left. Another arm may want its own.
TROSSEN_LEADER_GRIPPER_FRICTION = 10.02


@cfn.config(ip='192.168.1.2', force_feedback_gain=0.0, gripper_friction_constant=TROSSEN_LEADER_GRIPPER_FRICTION)
def trossen_leader(ip: str, force_feedback_gain: float, gripper_friction_constant: float | None):
    from positronic.drivers.roboarm.trossen_leader import Leader

    return Leader(ip=ip, force_feedback_gain=force_feedback_gain, gripper_friction_constant=gripper_friction_constant)
