import configuronic as cfn


@cfn.config(port="/dev/ttyUSB0")
def dh(port: str):
    from positronic.drivers.gripper.dh import DHGripper

    return DHGripper(port)
