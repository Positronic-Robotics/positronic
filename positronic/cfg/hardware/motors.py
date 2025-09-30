import configuronic as cfn
import numpy as np

leader_calibration = {
    'mins': np.array([380.0, 928.0, 874.0, 721.0, 303.0, 2020.0]),
    'maxs': np.array([3094.0, 3303.0, 3084.0, 3034.0, 4144.0, 3308.0]),
}


follower_calibration = {
    'mins': np.array([887.0, 725.0, 979.0, 878.0, 17.0, 2016.0]),
    'maxs': np.array([3598.0, 3095.0, 3197.0, 3223.0, 3879.0, 3526.0]),
}


@cfn.config()
def feetech(port: str, calibration: dict[str, np.ndarray] | None = None, processing_freq: float = 1000.0):
    from positronic.drivers.motors.feetech import MotorBus

    return MotorBus(port, calibration, processing_freq)


so101_follower = feetech.override(port='/dev/ttyACM0', calibration=follower_calibration)
so101_leader = feetech.override(port='/dev/ttyACM1', calibration=leader_calibration)
