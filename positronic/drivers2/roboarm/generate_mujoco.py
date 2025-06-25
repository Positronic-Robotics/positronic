import mujoco as mj

def convert_urdf_to_mujoco(urdf_path: str) -> mj.MjModel:
    """
    Convert a URDF file to a Mujoco model.

    Args:
        urdf_path: (str) Path to the URDF file to convert

    Returns:
        mj.Model: Compiled MuJoCo model with actuators, sensors, sites, and visuals
    """
    spec = mj.MjSpec.from_file(urdf_path)

    _add_actuators(spec)
    _add_sites(spec)
    _add_visuals(spec)
    _add_sensors(spec)
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST

    return spec


def _add_actuators(spec: mj.MjSpec) -> None:
    """Add position actuators for each joint."""
    for joint in spec.joints:
        if joint.type != mj.mjtJoint.mjJNT_FREE:  # Skip free joints
            actuator = spec.add_actuator()
            actuator.name = f"actuator_{joint.name.split('_')[1]}"
            actuator.trntype = mj.mjtTrn.mjTRN_JOINT
            actuator.gainprm = [1000.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # kp
            actuator.biasprm = [0, -1000.0, -100.0, 0, 0, 0, 0, 0, 0, 0]  # kv
            actuator.target = joint.name
            actuator.biastype = mj.mjtBias.mjBIAS_AFFINE
            actuator.forcerange = [-100, 100]
            actuator.ctrlrange = [-3.1416, 3.1416]  # Full rotation for base



def _add_sites(spec: mj.MjSpec) -> None:
    """Add sites at joints for sensors and end effector."""
    # Based on the URDF structure, each link has a corresponding joint
    # joint_1 moves link1, joint_2 moves link2, etc.
    for joint in spec.joints:
        site = joint.parent.add_site()
        site.name = f"{joint.name}_site"
        site.pos = [0.0, 0.0, 0.0]

    # Add end effector site to the last link (link6)
    for body in spec.bodies:
        if body.name == "link6":
            end_site = body.add_site()
            end_site.name = "end_effector"
            end_site.pos = [0.0, 0.0, 0.0]
            break


def _add_visuals(spec: mj.MjSpec, motor_size: float = 0.05) -> None:
    """Add visual geometry to bodies."""
    for body in spec.bodies:
        if body.name and "link" in body.name:

            motor_geom = body.add_geom(
                type=mj.mjtGeom.mjGEOM_CYLINDER,
                size=[motor_size, 0.025, 0],
                pos=[0.0, 0.0, 0.0],
                rgba=[1.0, 1.0, 1.0, 1.0],
                density=0,

            )
            if len(body.bodies) == 0:
                continue

            offset = body.bodies[0].pos
            link_size = max(offset[2] - motor_size, 0.05)
            print('link_size', link_size)
            print('adding visual for', body.name)
            link_geom = body.add_geom(
                type=mj.mjtGeom.mjGEOM_CYLINDER,
                size=[0.025, link_size / 2, 0],
                pos=[0.0, 0.0, link_size / 2],
                rgba=[1.0, 0.0, 1.0, 1.0],
                density=0,
            )


def _add_sensors(spec: mj.MjSpec) -> None:
    """Add torque sensors at each joint site."""
    # Iterate through bodies to find their sites
    for site in spec.sites:
        if 'joint' in site.name:
            sensor = spec.add_sensor()
            sensor.name = f"torque_{site.name.replace('_site', '')}"
            sensor.type = mj.mjtSensor.mjSENS_TORQUE
            sensor.objname = site.name
            sensor.objtype = mj.mjtObj.mjOBJ_SITE