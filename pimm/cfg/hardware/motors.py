import configuronic as cfn
import numpy as np


leader_calibration = {
    "mins": np.array([ 380.,  928.,  874.,  721.,  303., 2020.]),
    "maxs": np.array([3094., 3303., 3084., 3034., 4144., 3308.])
}


follower_calibration = {
    "mins": np.array([ 887.,  725.,  979.,  878.,   17., 2016.]),
    "maxs": np.array([3598., 3095., 3197., 3223., 3879., 3526.])
}


@cfn.config()
def feetech(port: str, calibration: dict[str, np.ndarray] | None = None, processing_freq: float = 1000.0):
    from pimm.drivers.motors.feetech import MotorBus

    return MotorBus(port, calibration, processing_freq)


so101_follower = feetech.override(port="/dev/ttyACM0", calibration=follower_calibration)
so101_leader = feetech.override(port="/dev/ttyACM1", calibration=leader_calibration)
