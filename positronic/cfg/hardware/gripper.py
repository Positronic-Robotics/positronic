import configuronic as cfgc


@cfgc.config(port="/dev/ttyUSB0")
def dh(port: str):
    from positronic.drivers.gripper.dh import DHGripper

    return DHGripper(port)
