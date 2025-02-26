from typing import List, Optional

from cfg import store, builds


def _franka(ip: str,
            relative_dynamics_factor: float = 0.2,
            gripper_speed: float = 0.02,
            realtime_config: str = "Ignore",
            collision_behavior: Optional[List[List[float]]] = None,
            home_joints_config: Optional[List[float]] = None,
            cartesian_mode: str = "LIBFRANKA"):
    from drivers.roboarm.franka import Franka, CartesianMode
    from franky import RealtimeConfig

    realtime_config = getattr(RealtimeConfig, realtime_config)
    cartesian_mode = getattr(CartesianMode, cartesian_mode)
    return Franka(ip, relative_dynamics_factor, gripper_speed, realtime_config, collision_behavior, home_joints_config,
                  cartesian_mode)


franka = builds(_franka)

roboarm_store = store(group="hardware/roboarms")

roboarm_store(franka(ip="172.168.0.2",
                     relative_dynamics_factor=0.2,
                     gripper_force=0.4,
                     home_joints_config=[0.0, -0.31, 0.0, -1.53, 0.0, 1.522, 0.785],
                     collision_behavior=[[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                         [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                         [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                         [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                         [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                         [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                         [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                                         [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]],
                     cartesian_mode="POSITRONIC"),
              name="franka")

roboarm_store.add_to_hydra_store()
