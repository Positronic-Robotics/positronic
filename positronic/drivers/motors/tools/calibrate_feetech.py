import configuronic as cfn
import pimm
import numpy as np

import positronic.cfg.hardware.motors
from positronic.drivers.motors.feetech import MotorBus


def get_function(motor_bus: MotorBus):
    def calibrate_fn(should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        motor_bus.connect()
        mins = np.full(len(motor_bus.motor_indices), np.inf)
        maxs = np.full(len(motor_bus.motor_indices), -np.inf)

        print_limiter = pimm.RateLimiter(hz=30, clock=clock)
        print()
        print()

        while not should_stop.value:
            if motor_bus.position is None:
                yield pimm.Sleep(0.001)
            else:
                break

        motor_bus.set_torque_mode(False)

        while not should_stop.value:
            pos = motor_bus.position
            mins = np.minimum(mins, pos)
            maxs = np.maximum(maxs, pos)

            if print_limiter.wait_time() > 0:
                print(f"mins: {mins.tolist()}, maxs: {maxs.tolist()}", end='\r')

            yield pimm.Sleep(0.001)

        mins_str = np.array2string(mins, separator=', ', precision=1).strip()
        maxs_str = np.array2string(maxs, separator=', ', precision=1).strip()

        print("{")
        print(f'    "mins": np.array({mins_str}),')
        print(f'    "maxs": np.array({maxs_str})')
        print("}")

    return calibrate_fn


@cfn.config(motor_bus=positronic.cfg.hardware.motors.feetech)
def calibrate(motor_bus: MotorBus):
    with pimm.World() as w:
        calibrate_fn = get_function(motor_bus)
        w.start_in_subprocess(calibrate_fn)

        input("Move all joints to it's limit, then press ENTER...")


if __name__ == "__main__":
    cfn.cli(calibrate)
