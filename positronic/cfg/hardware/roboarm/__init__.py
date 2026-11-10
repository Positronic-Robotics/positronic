import dataclasses
from typing import TYPE_CHECKING

import configuronic as cfn

import positronic.cfg.hardware.motors
from positronic.drivers.fakes.franka import FakeFranka
from positronic.drivers.roboarm import command
from positronic.drivers.roboarm.models import DROID_EE_FRAME

if TYPE_CHECKING:
    from positronic.drivers.roboarm.yam.compensation import Compensation, FrictionFade
    from positronic.drivers.roboarm.yam.settle import SettleTuning

# The pose each arm is drawn around at the start of a trial. Where a driver parks is its own and lives with it.
FRANKA_NOMINAL_JOINTS = [0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0]
YAM_NOMINAL_JOINTS = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]
SO101_NOMINAL_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0]
# The two CRON arms on one stand, with the tool axis level and the tool x axis vertical. The left pose is the mirror of
# the right pose, joint by joint. Of the tool positions near the first start pose, this one has the most even reach
# (2026-11-10, with the IK of the controller and a margin from each pose where joints turn with no limit): 0.53 m up,
# 0.31 m toward the base, 0.55 m away from it and 0.70 m or more outward, inward and down.
CRON_LEFT_NOMINAL_JOINTS = [-3.36325, -1.34565, -2.27067, 3.64774, 4.45932, 2.36322]
CRON_RIGHT_NOMINAL_JOINTS = [0.22166, -1.79594, 2.27067, -0.50615, 1.82387, 7.06156]
# How far, per joint, a start pose drawn around the Franka's nominal may sit from it.
FRANKA_JOINTS_SPREAD = [0.03, 0.05, 0.08, 0.08, 0.10, 0.10, 0.10]
# The law that DROID-pretrained checkpoints run under: the joint gains and the Cartesian damping of DROID's Franka,
# and a Cartesian stiffness at the flange, where DROID measures its poses.
DROID_IMPEDANCE = command.Impedance(
    kq=(40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0),
    kqd=(4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0),
    kx=(400.0, 400.0, 400.0, 15.0, 15.0, 15.0),
    kxd=(37.0, 37.0, 37.0, 2.0, 2.0, 2.0),
    frame=DROID_EE_FRAME,
)


@cfn.config(
    ip='172.168.0.2',
    relative_dynamics_factor=0.2,
    load=None,
    collision_coeff=2.0,
    manage_desk=True,
    brake_after_idle_s=None,
)
def franka(
    ip: str,
    relative_dynamics_factor: float,
    load: tuple | None = None,
    collision_coeff: float = 2.0,
    manage_desk: bool = True,
    brake_after_idle_s: float | None = None,
):
    from positronic.drivers.roboarm import franka  # noqa: F401

    return franka.Robot(
        ip=ip,
        relative_dynamics_factor=relative_dynamics_factor,
        load=load,
        collision_coeff=collision_coeff,
        manage_desk=manage_desk,
        brake_after_idle_s=brake_after_idle_s,
    )


franka_droid = franka.override(load=(0.9, [0.0, 0.0, 0.057], [0.002768, 0, 0, 0, 0.003149, 0, 0, 0, 0.000564]))
franka_fake = cfn.Config(FakeFranka)


@cfn.config(ip='192.168.1.10', relative_dynamics_factor=0.5)
def kinova(ip, relative_dynamics_factor):
    from positronic.drivers.roboarm.kinova.driver import Robot

    return Robot(ip=ip, relative_dynamics_factor=relative_dynamics_factor)


# tool_length: a DH Robotics AG-160-95 on its flange adapter, from the flange to the middle of the finger pads
@cfn.config(
    ip='192.168.68.63',
    relative_dynamics_factor=0.2,
    power_off_on_exit=True,
    tool_length=0.195,
    projects=(),
    tool_capsules=(),
)
def cron(ip, relative_dynamics_factor, power_off_on_exit, tool_length, projects, tool_capsules):
    from positronic.drivers.roboarm.cron.driver import Robot

    return Robot(
        ip=ip,
        relative_dynamics_factor=relative_dynamics_factor,
        power_off_on_exit=power_off_on_exit,
        tool_length=tool_length,
        projects=projects,
        tool_capsules=tool_capsules,
    )


# The DH AG-160-95 on its adapter in the collision model: 225 mm from the flange to the finger tips, on the axis of
# the flange; the open jaws are 162 mm across, along the y axis of the flange (horizontal in the start pose). The model
# of the controller (v1.10.11) reads a capsule as a sphere at `xyz` and ignores `length` and `rpy`, so the body is a
# row of spheres along the axis and the jaws are two pairs of spheres near the tips
_DH_GRIPPER_CAPSULES = [
    *({'length': 0.0, 'radius': 0.045, 'xyz': [0.0, 0.0, z], 'rpy': [0.0, 0.0, 0.0]} for z in (0.03, 0.07, 0.11, 0.15)),
    *(
        {'length': 0.0, 'radius': 0.025, 'xyz': [0.0, y, z], 'rpy': [0.0, 0.0, 0.0]}
        for z in (0.19, 0.215)
        for y in (-0.06, 0.06)
    ),
]

# A CRON arm with the DH gripper on its wrist: the arm starts the bridge project to the gripper on its control box
cron_with_gripper = cron.override(projects=['gripper_bridge'], tool_capsules=_DH_GRIPPER_CAPSULES)


@cfn.config(motor_bus=positronic.cfg.hardware.motors.so101_follower)
def so101(motor_bus):
    from positronic.drivers.roboarm.so101.driver import Robot

    return Robot(motor_bus=motor_bus)


@cfn.config()
def yam_park_tuning(**fields: float) -> 'SettleTuning':
    from positronic.drivers.roboarm.yam.settle import PARK_SETTLE

    return dataclasses.replace(PARK_SETTLE, **fields)


@cfn.config()
def yam_move_tuning(**fields: float) -> 'SettleTuning':
    from positronic.drivers.roboarm.yam.settle import MOVE_SETTLE

    return dataclasses.replace(MOVE_SETTLE, **fields)


@cfn.config()
def yam_compensation(**fields) -> 'Compensation':
    from positronic.drivers.roboarm.yam.compensation import I2RT_COMPENSATION

    return dataclasses.replace(I2RT_COMPENSATION, **fields)


@cfn.config()
def yam_friction_fade(zero_below_rad_s: float, full_from_rad_s: float) -> 'FrictionFade':
    from positronic.drivers.roboarm.yam.compensation import FrictionFade

    return FrictionFade(zero_below_rad_s=zero_below_rad_s, full_from_rad_s=full_from_rad_s)


@cfn.config(
    channel='can0',
    sim=False,
    base_pose=None,
    compensation=yam_compensation,
    park_after_idle_s=60.0,
    park_tuning=yam_park_tuning,
    move_tuning=yam_move_tuning,
)
def yam(
    channel: str,
    sim: bool,
    base_pose,
    compensation: 'Compensation',
    park_after_idle_s: float | None,
    park_tuning: 'SettleTuning',
    move_tuning: 'SettleTuning',
):
    from positronic.drivers.roboarm.yam.driver import Robot

    return Robot(
        channel,
        base_pose=base_pose,
        sim=sim,
        compensation=compensation,
        park_after_idle_s=park_after_idle_s,
        park_tuning=park_tuning,
        move_tuning=move_tuning,
    )
