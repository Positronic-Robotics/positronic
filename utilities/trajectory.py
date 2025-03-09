import franky
import hydra
from omegaconf import DictConfig
import geom

@hydra.main(version_base=None, config_path=".", config_name="trajectory")
def main(cfg: DictConfig):
    robot = franky.Robot(cfg.ip, realtime_config=franky.RealtimeConfig.Ignore)
    robot.relative_dynamics_factor = cfg.relative_dynamics_factor
    robot.set_collision_behavior(
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    )

    robot.recover_from_errors()

    waypoints = []
    last_q = robot.state.q
    print(cfg.targets)
    for target in cfg.targets:
        if target is None:
            continue
        elif "joints" in target:
            robot.move(franky.JointMotion(target.joints))
        elif 'relative' in target:
            pos = geom.Transform3D(translation=target.relative.translation, rotation=geom.Rotation.from_quat(target.relative.quaternion))
            pos = franky.Affine(translation=pos.translation, quaternion=pos.rotation.as_quat)
            print("Current pose", robot.current_pose.end_effector_pose)
            robot.move(franky.CartesianMotion(pos, reference_type=franky.ReferenceType.Relative))
        elif 'absrel' in target:
            q = target.absrel.quaternion
            rel_transform = geom.Transform3D(
                translation=target.absrel.translation,
                rotation=geom.Rotation.from_quat_xyzw(q)
            )

            current_pose = robot.current_pose.end_effector_pose
            print("Current pose", current_pose.quaternion)
            current_transform = geom.Transform3D(
                translation=current_pose.translation,
                rotation=geom.Rotation.from_quat_xyzw(current_pose.quaternion)
            )

            combined_transform = current_transform * rel_transform

            # Convert to franky.Affine format
            pos = franky.Affine(
                translation=combined_transform.translation,
                quaternion=combined_transform.rotation.as_quat_xyzw
            )


            # Use ReferenceType.Absolute with the current pose as the frame
            # This will mimic ReferenceType.Relative behavior
            robot.move(franky.CartesianMotion(pos, reference_type=franky.ReferenceType.Absolute))
        elif "ik" in target:
            pos = franky.Affine(translation=target.ik.translation, rotation=geom.Rotation.from_quat(target.ik.quaternion))
            q = robot.inverse_kinematics(pos, last_q)
            robot.move(franky.JointMotion(q))
            last_q = q
        elif "impedance" in target:
            pos = franky.Affine(translation=target.impedance.translation, quaternion=target.impedance.quaternion)
            # robot.move(franky.ExponentialImpedanceMotion(pos))
            kwargs = {}
            if 'impedance' in target:
                kwargs['translational_stiffness'] = target.impedance.stiffness.translational
                kwargs['rotational_stiffness'] = target.impedance.stiffness.rotational
            robot.move(franky.CartesianImpedanceMotion(pos, duration=target.impedance.duration, return_when_finished=False, finish_wait_factor=10, **kwargs))
        elif "file" in target:
            from adhoc.validate_trajectory import get_umi_trajectory
            umi_relative_trajectory = get_umi_trajectory(target.file)
            registration_transform = geom.Transform3D(rotation=geom.Rotation.from_euler([1.36897958, -0.73992762, 1.39720004]))

            waypoints = []
            for pos in umi_relative_trajectory:
                pos = registration_transform.inv * pos * registration_transform
                q = pos.rotation.as_quat
                q = [q[1], q[2], q[3], q[0]]
                pos = franky.Affine(translation=pos.translation, quaternion=q)
                pos = franky.CartesianWaypoint(pos, reference_type=franky.ReferenceType.Relative)
                waypoints.append(pos)

            robot.move(franky.CartesianWaypointMotion(waypoints, ))
    print(robot.current_pose.end_effector_pose)
    print(robot.current_joint_state.position)


if __name__ == "__main__":
    main()
