import ironic as ir


@ir.config(port="/dev/ttyUSB0")
def dh_gripper(port: str):
    from pimm.drivers.gripper.dh import DHGripper  # noqa: F401
    return DHGripper(port=port)


@ir.config(ip="172.168.0.2")
def franka(ip: str):
    from pimm.drivers.gripper.franka import Gripper  # noqa: F401
    return Gripper(ip=ip)
