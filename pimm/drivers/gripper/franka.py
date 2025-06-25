import time
import franky
import ironic2 as ir


class Gripper:
    grip: ir.SignalEmitter = ir.NoOpEmitter()
    target_grip: ir.SignalReader = ir.NoOpReader()
    force: ir.SignalReader = ir.NoOpReader()
    speed: ir.SignalReader = ir.NoOpReader()

    def __init__(self, ip: str, close_threshold: float = 0.4, open_threshold: float = 0.6):
        assert 0 < close_threshold < open_threshold < 1
        self._ip = ip
        self._close_threshold = close_threshold
        self._open_threshold = open_threshold

    def run(self, should_stop: ir.SignalReader) -> None:
        gripper = franky.Gripper(self._ip)
        gripper.homing()

        is_open = True
        limiter = ir.RateLimiter(hz=100)

        while not ir.signal_value(should_stop):
            try:
                target_grip = ir.signal_value(self.target_grip)
                if is_open and target_grip < self._close_threshold:
                    gripper.grasp(width=0, speed=ir.signal_value(self.speed), force=ir.signal_value(self.force))
                    is_open = False
                elif not is_open and target_grip > self._open_threshold:
                    gripper.grasp(width=gripper.max_width,
                                  speed=ir.signal_value(self.speed),
                                  force=0)
                    is_open = True
            except ir.NoValueException:
                time.sleep(0.05)

            limiter.wait()
