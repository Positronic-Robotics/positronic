from dataclasses import dataclass
from typing import Callable, List

import franky
import fire

import configuronic as cfgc


@dataclass
class JointMotion:
    joints: List[float]


@dataclass
class IKMotion:
    translation: List[float]
    quaternion: List[float]


@dataclass
class ImpedanceMotion:
    translation: List[float]
    quaternion: List[float]
    stiffness: List[float]


def main(ip: str, relative_dynamics_factor: float, targets: List[Callable[[], franky.Motion]]):
    robot = franky.Robot(ip, realtime_config=franky.RealtimeConfig.Ignore)
    robot.relative_dynamics_factor = relative_dynamics_factor
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

    last_q = robot.state.q
    for target in targets:
        if isinstance(target, JointMotion):
            robot.move(franky.JointMotion(target.joints))
        elif isinstance(target, IKMotion):
            affine_pose = franky.Affine(translation=target.translation, quaternion=target.quaternion)
            q = robot.inverse_kinematics(affine_pose, last_q)
            robot.move(franky.JointMotion(q))
            last_q = q
        elif isinstance(target, ImpedanceMotion):
            affine_pose = franky.Affine(translation=target.translation, quaternion=target.quaternion)
            stiffness = franky.CartesianStiffness(
                translational=target.stiffness,
                rotational=target.stiffness
            )
            # robot.move(franky.ExponentialImpedanceMotion(pos))
            kwargs = {}
            if stiffness is not None:
                kwargs['translational_stiffness'] = target.stiffness.translational
                kwargs['rotational_stiffness'] = target.stiffness.rotational
            motion = franky.CartesianImpedanceMotion(
                affine_pose,
                duration=target.duration,
                return_when_finished=False,
                finish_wait_factor=10,
                **kwargs
            )
            robot.move(motion)
            motion.wait_until_finished()

    print(robot.current_pose.end_effector_pose)


base_cfg = cfgc.Config(
    main,
    ip="172.168.0.2",
    relative_dynamics_factor=0.2,
    targets=[
        cfgc.Config(JointMotion, joints=[0, 0, 0, 0, 0, 0, 0]),
    ]
)


if __name__ == "__main__":
    fire.Fire(base_cfg.override_and_instantiate)
