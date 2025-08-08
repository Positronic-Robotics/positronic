import configuronic as cfn
import numpy as np



leader_calibration = {
    "mins": np.array([552.0, 927.0, 876.0, 729.0, 306.0, 2045.0]),
    "maxs": np.array([3081.0, 3286.0, 3089.0, 3032.0, 4131.0, 3299.0]),
}


follower_calibration = {
    "mins": np.array([920.0, 724.0, 963.0, 884.0, 29.0, 2034.0]),
    "maxs": np.array([3589.0, 3094.0, 3184.0, 3183.0, 3874.0, 3509.0]),
}

@cfn.config()
def feetech(port: str, calibration: dict[str, np.ndarray] | None = None, processing_freq: float = 1000.0):
    from pimm.drivers.motors.feetech import MotorBus

    return MotorBus(port, calibration, processing_freq)


so101_follower = feetech.override(port="/dev/ttyACM0", calibration=follower_calibration)
so101_leader = feetech.override(port="/dev/ttyACM1", calibration=leader_calibration)