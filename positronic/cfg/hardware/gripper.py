import configuronic as cfn


@cfn.config(port='/dev/ttyUSB0')
def dh_gripper(port: str):
    from positronic.drivers.gripper.dh import DHGripper  # noqa: F401

    return DHGripper(port=port)


@cfn.config(port='/dev/ttyUSB0')
def robotiq(port: str):
    from positronic.drivers.gripper.robotiq import Robotiq2F  # noqa: F401

    return Robotiq2F(port=port)
