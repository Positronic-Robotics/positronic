from typing import List, Optional

import ironic as ir


@ir.config(ip="172.168.0.2",
           relative_dynamics_factor=0.2,
           gripper_speed=0.02,
           realtime_config="Ignore",
           collision_behavior=None,
           home_joints_config=None,
           cartesian_mode="LIBFRANKA")
def franka_default(ip: str, relative_dynamics_factor: float, gripper_speed: float, realtime_config: str,
                   collision_behavior: Optional[List[List[float]]], home_joints_config: Optional[List[float]],
                   cartesian_mode: str):
    from positronic.drivers.roboarm.franka import Franka, CartesianMode
    from franky import RealtimeConfig

    realtime_config = getattr(RealtimeConfig, realtime_config)
    cartesian_mode = getattr(CartesianMode, cartesian_mode)
    return Franka(ip, relative_dynamics_factor, gripper_speed, realtime_config, collision_behavior, home_joints_config,
                  cartesian_mode)


franka_ik = franka_default.override(ip="172.168.0.2",
                                    relative_dynamics_factor=0.2,
                                    home_joints_config=[0.0, -0.31, 0.0, -1.53, 0.0, 1.522, 0.785],
                                    collision_behavior=[[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                                        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                                        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                                        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                                        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                                        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                                        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                                        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]],
                                    cartesian_mode="POSITRONIC")


@ir.config(ip="192.168.1.10", relative_dynamics_factor=0.5)
def kinova(ip: str, relative_dynamics_factor: float):
    from positronic.drivers.roboarm.kinova import Kinova
    kinova = Kinova(ip, relative_dynamics_factor)


    kinova = ir.extend(kinova, **{
        'ext_force_ee': ir.utils.const_property([0, 0, 0]),
        'ext_force_base': ir.utils.const_property([0, 0, 0]),
    })
    return kinova