import ironic as ir


@ir.config(port="/dev/ttyUSB0")
def dh_gripper(port: str):
    from pimm.drivers.gripper.dh import DHGripper  # noqa: F401
    return DHGripper(port=port)
