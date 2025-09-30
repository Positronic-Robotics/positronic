from collections.abc import Sequence

import configuronic as cfn
import mujoco as mj
import numpy as np

import positronic.cfg.hardware.roboarm.motors
from positronic.drivers.roboarm.generate_urdf import MotorParameters, create_arm


def convert_urdf_to_mujoco(
    urdf_path: str, wall_mounted: bool = False, kp: float = 1000.0, kv: float = 100.0, actuator_type: str = 'position'
) -> mj.MjModel:
    """
    Convert a URDF file to a Mujoco model.

    Args:
        urdf_path: (str) Path to the URDF file to convert

    Returns:
        mj.Model: Compiled MuJoCo model with actuators, sensors, sites, and visuals
    """
    spec = mj.MjSpec.from_file(urdf_path)

    _add_actuators(spec, kp, kv, actuator_type)
    _add_sites(spec)
    _add_geoms(spec)
    _add_sensors(spec)
    _add_camera(spec)
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST

    if wall_mounted:
        spec.body('link1').quat = [0, 0.707107, 0, 0.707107]

    return spec


def _add_actuators(spec: mj.MjSpec, kp: float, kv: float, actuator_type: str) -> None:
    """Add position actuators for each joint."""
    for joint in spec.joints:
        if actuator_type == 'position':
            actuator = spec.add_actuator()
            actuator.name = f'actuator_{joint.name.split("_")[1]}'
            actuator.trntype = mj.mjtTrn.mjTRN_JOINT
            actuator.gainprm = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # kp
            actuator.biasprm = [0, -kp, -kv, 0, 0, 0, 0, 0, 0, 0]  # kv
            actuator.target = joint.name
            actuator.biastype = mj.mjtBias.mjBIAS_AFFINE
            actuator.forcerange = joint.actfrcrange
            actuator.ctrlrange = [-np.pi, np.pi]
        elif actuator_type == 'torque':
            actuator = spec.add_actuator()
            actuator.name = f'actuator_{joint.name.split("_")[1]}'
            actuator.trntype = mj.mjtTrn.mjTRN_JOINT
            actuator.gainprm = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # kp
            actuator.biasprm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # kv
            actuator.biastype = mj.mjtBias.mjBIAS_NONE
            actuator.gaintype = mj.mjtGain.mjGAIN_FIXED
            actuator.target = joint.name
            actuator.ctrlrange = joint.actfrcrange


def _add_sites(spec: mj.MjSpec) -> None:
    for joint in spec.joints:
        site = joint.parent.add_site()
        site.name = f'{joint.name}_site'
        site.pos = [0.0, 0.0, 0.0]

    # find the last link
    link_numbers = [int(body.name.replace('link', '')) for body in spec.bodies if 'link' in body.name]
    max_link = max(link_numbers)

    end_site = spec.body(f'link{max_link}').add_site()
    end_site.name = 'end_effector'
    end_site.pos = [0.0, 0.0, 0.0]


def _add_geoms(
    spec: mj.MjSpec,
    motor_height: float = 0.05,
    motor_radius: float = 0.05,
    link_radius: float = 0.025,
) -> None:
    for body in spec.bodies:
        if body.name and 'link' in body.name:
            half_height = motor_height / 2  # Cylinder is defined by half-height and radius in mujoco
            body.add_geom(
                type=mj.mjtGeom.mjGEOM_CYLINDER,
                size=[motor_radius, half_height, 0],
                pos=[0.0, 0.0, 0.0],
                rgba=[1.0, 1.0, 1.0, 1.0],
                density=0,
            )
            if len(body.bodies) == 0:
                continue

            offset = body.bodies[0].pos
            link_size = offset[2] - half_height - motor_radius  # TODO: here should be the radius of the next motor
            body.add_geom(
                type=mj.mjtGeom.mjGEOM_CYLINDER,
                size=[link_radius, link_size / 2, 0],
                pos=[0.0, 0.0, link_size / 2 + half_height],
                rgba=[1.0, 0.0, 1.0, 1.0],
                density=0,
            )


def _add_sensors(spec: mj.MjSpec) -> None:
    for site in spec.sites:
        if 'joint' in site.name:
            sensor = spec.add_sensor()
            sensor.name = f'torque_{site.name.replace("_site", "")}'
            sensor.type = mj.mjtSensor.mjSENS_TORQUE
            sensor.objname = site.name
            sensor.objtype = mj.mjtObj.mjOBJ_SITE


def _add_camera(spec: mj.MjSpec) -> None:
    spec.worldbody.add_camera(
        name='viewer', pos=[1.235, -0.839, 1.092], xyaxes=[0.712, 0.702, -0.000, -0.420, 0.425, 0.802]
    )


@cfn.config(
    wall_mounted=False,
    urdf_path='robot_urdf.xml',
    link_lengths=[0.05, 0.05, 0.2, 0.05, 0.2, 0.05],
    motors=[
        positronic.cfg.hardware.roboarm.motors.my_actuator_rmd_x10_p35_100,
        positronic.cfg.hardware.roboarm.motors.my_actuator_rmd_x10_p35_100,
        positronic.cfg.hardware.roboarm.motors.my_actuator_rmd_x6_v3,
        positronic.cfg.hardware.roboarm.motors.my_actuator_rmd_x6_v3,
        positronic.cfg.hardware.roboarm.motors.my_actuator_rmd_x6_v3,
        positronic.cfg.hardware.roboarm.motors.my_actuator_rmd_x6_v3,
        positronic.cfg.hardware.roboarm.motors.my_actuator_rmd_x6_v3,
    ],
    joint_rotations=[np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2],
    link_density=0.2,
    payload_mass=2.0,
    kp=100.0,
    kv=10.0,
    actuator_type='position',
)
def main(
    target_path: str,
    wall_mounted: bool,
    urdf_path: str,
    link_lengths: Sequence[float],
    motors: Sequence[MotorParameters],
    joint_rotations: Sequence[float],
    link_density: float,
    payload_mass: float,
    kp: float,
    kv: float,
    actuator_type: str,
):
    with open(urdf_path, 'w') as f:
        xml = create_arm(
            link_lengths=link_lengths,
            motors=motors,
            joint_rotations=joint_rotations,
            link_density=link_density,
            payload_mass=payload_mass,
        )
        f.write(xml)

    spec = convert_urdf_to_mujoco(urdf_path, wall_mounted=wall_mounted, kp=kp, kv=kv, actuator_type=actuator_type)
    spec.compile()
    with open(target_path, 'w') as f:
        f.write(spec.to_xml())


if __name__ == '__main__':
    cfn.cli(main)
