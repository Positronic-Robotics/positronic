import ironic as ir

from pimm.drivers.gripper.dh import DHGripper

dh_gripper = ir.Config(
    DHGripper,
    port="/dev/ttyUSB0",
)
