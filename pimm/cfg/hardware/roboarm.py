import ironic as ir


@ir.config(ip="172.168.0.2", relative_dynamics_factor=0.2, cartesian_mode="positronic")
def franka(ip: str, relative_dynamics_factor: float, cartesian_mode: str):
    from pimm.drivers.roboarm.franka import Robot  # noqa: F401
    return Robot(ip=ip, relative_dynamics_factor=relative_dynamics_factor, cartesian_mode=cartesian_mode)
