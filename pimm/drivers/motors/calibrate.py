import configuronic as cfn
import ironic2 as ir
import numpy as np
import json

import pimm.cfg.hardware.motors
from pimm.drivers.motors.feetech import MotorBus


def get_function(position: ir.SignalReader, torque_mode: ir.SignalReader):
    def record_limits(should_stop: ir.SignalReader, clock: ir.Clock):
        mins = np.full(6, np.inf)
        maxs = np.full(6, -np.inf)

        print_limiter = ir.RateLimiter(hz=30, clock=clock)
        print()
        print()

        while not should_stop.value:
            if position.read() is None:
                yield ir.Sleep(0.001)
            else:
                break

        torque_mode.emit(False)

        while not should_stop.value:
            pos = position.value
            mins = np.minimum(mins, pos)
            maxs = np.maximum(maxs, pos)

            if print_limiter.wait_time() > 0:
                print(f"mins: {mins.tolist()}, maxs: {maxs.tolist()}", end='\r')

            yield ir.Sleep(0.001)

        print(json.dumps({"mins": mins.tolist(), "maxs": maxs.tolist()}, indent=4))

    return record_limits


@cfn.config(motor_bus=pimm.cfg.hardware.motors.feetech)
def calibrate(motor_bus: MotorBus):
    with ir.World() as w:
        motor_bus.position, motor_position = w.mp_pipe()
        torque_mode, motor_bus.torque_mode = w.mp_pipe()

        record_limits = get_function(motor_position, torque_mode)
        w.start_in_subprocess(record_limits, motor_bus.run)

        input("Move all joints to it's limit, then press ENTER...")

if __name__ == "__main__":
    cfn.cli(calibrate)