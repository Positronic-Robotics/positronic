import configuronic as cfn

from positronic.drivers.fakes.robotiq import FakeRobotiq2F


@cfn.config(port='/dev/ttyUSB0', force=100, speed=100, target_period_s=None, target_step=None)
def dh_gripper(port: str, force: int, speed: int, target_period_s: float | None, target_step: int | None):
    """``target_period_s`` and ``target_step`` left as None take the values of the driver."""
    from positronic.drivers.gripper import dh

    return dh.DHGripper(
        port=port,
        force=force,
        speed=speed,
        target_period_s=dh.TARGET_PERIOD_S if target_period_s is None else target_period_s,
        target_step=dh.TARGET_STEP if target_step is None else target_step,
    )


# The DH gripper on the wrist of a CRON arm, through the bridge project on its control box
# (box_projects/gripper_bridge in the station notes). The force also sets how fast the fingers move: a full stroke
# took 0.88 s at force 100 and 2.0 s at force 30 (2026-11-06)
cron_dh_gripper = dh_gripper.override(port='socket://192.168.68.67:7663')


@cfn.config(port='/dev/ttyUSB0')
def robotiq(port: str):
    from positronic.drivers.gripper.robotiq import Robotiq2F  # noqa: F401

    return Robotiq2F(port=port)


robotiq_fake = cfn.Config(FakeRobotiq2F)
