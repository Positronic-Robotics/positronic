from positronic.drivers.gripper.dh import DHGripper

import ironic as ir


@ir.config(port="/dev/ttyUSB0")
def dh(port: str):
    return DHGripper(port)
