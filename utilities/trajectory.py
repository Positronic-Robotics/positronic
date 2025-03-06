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
            print(f"Doing: {target.relative}")
            robot.move(franky.CartesianMotion(pos, reference_type=franky.ReferenceType.Relative))
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

    print(robot.current_pose.end_effector_pose)


if __name__ == "__main__":
    main()
