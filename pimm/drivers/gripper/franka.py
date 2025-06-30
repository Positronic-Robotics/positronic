import time

import franky

import ironic2 as ir
from ironic.utils import RateLimiter


class Gripper:
    grip: ir.SignalEmitter = ir.NoOpEmitter()
    target_grip: ir.SignalReader = ir.NoOpReader()
    force: ir.SignalReader = ir.NoOpReader()
    speed: ir.SignalReader = ir.NoOpReader()

    def __init__(self, ip: str, close_threshold: float = 0.6, open_threshold: float = 0.4):
        """
        Franka's gripper does not allow to rewrite the executing command with the newer one,
        hence it is currently sequential. To avoid weird behavior, we support only two states –
        fully open and fully closed.

        :param ip: IP address of the robot (and gripper).
        :param close_threshold: If `target_grip` is less than this value, the gripper will close.
        :param open_threshold: If `target_grip` is greater than this value, the gripper will open.
        """
        assert 0 < open_threshold < close_threshold < 1
        self._ip = ip
        self._close_threshold = close_threshold
        self._open_threshold = open_threshold

    def run(self, should_stop: ir.SignalReader) -> None:
        gripper = franky.Gripper(self._ip)
        print(f"Connected to gripper at {self._ip}, homing...")
        gripper.homing()

        is_open = True
        limiter = RateLimiter(hz=100)
        force = ir.DefaultReader(self.force, 5.0)  # N
        speed = ir.DefaultReader(self.speed, 0.05)  # m/s

        while not should_stop.value:
            try:
                target_grip = self.target_grip.value
                if is_open and target_grip > self._close_threshold:
                    gripper.grasp_async(width=0, speed=speed.value, force=force.value)
                    is_open = False
                elif not is_open and target_grip < self._open_threshold:
                    gripper.open_async(speed=speed.value)
                    is_open = True
            except ir.NoValueException:
                time.sleep(0.05)

            limiter.wait()
            self.grip.emit(gripper.width / gripper.max_width)


if __name__ == "__main__":
    with ir.World() as world:
        gripper = Gripper(ip="172.168.0.2")
        target_grip, gripper.target_grip = world.pipe(1)
        gripper.grip, actual_grip = world.pipe(1)
        world.start(gripper.run)

        commands = [(1.0, 0.0), (0.0, 4.0), (0.65, 8.0), (0.35, 12.0)]

        start, i = time.monotonic(), 0
        j = 0
        while i < len(commands) and not world.should_stop:
            target_grip_value, duration = commands[i]
            if time.monotonic() > start + duration:
                target_grip.emit(target_grip_value)
                i += 1
            else:
                time.sleep(0.01)

            if j % 100 == 0:
                print(f"Actual grip: {actual_grip.read()}")
            j += 1

        time.sleep(3)
        print("Finishing")
