import logging
import time
from collections.abc import Iterator
from ctypes import c_uint16

import pimm
from positronic.drivers import vendor_import
from positronic.drivers.gripper import aperture
from positronic.drivers.utils import Moves, grip_setpoint

with vendor_import('pymodbus', 'Gripper support'):
    import pymodbus.client as ModbusClient
    from pymodbus.exceptions import ModbusException

logger = logging.getLogger(__name__)


_DEVICE = 1
# Holding registers: write 0xA5 to start the calibration; the target width, force and speed; the calibration state
# (1 when done) and the width the fingers read back. Widths are in permille of the stroke, 1000 is open
_INIT, _FORCE, _TARGET, _SPEED = 0x100, 0x101, 0x103, 0x104
_INIT_STATE, _WIDTH = 0x200, 0x202
_START_INIT = 0xA5
# The first start of the bridge after a reboot of the control box took 11 s, as it set up its Python environment
_CONNECT_TIMEOUT_S, _CONNECT_RETRY_S = 60.0, 0.5
# Each new target starts the motion of the fingers again from rest, so a stream of targets keeps them still. Measured on
# two AG-160-95 at force 100 (2026-11-06, experiments/gripper_stress.py, human-like presses with trigger noise of +-3):
# written at each change, the fingers missed 7 of 8 moves; written at most each 0.1 s and only on a change of 2 % or
# more, they were 7.6 % of the stroke behind on average, and covered 90 % of a 30 % press in 0.35 s. Other periods
# (0.07 to 0.2 s), steps, a debounce and a lead on the trigger speed did no better
TARGET_PERIOD_S, TARGET_STEP = 0.1, 20


class DHGripper(pimm.ControlSystem):
    """A DH Robotics gripper on Modbus RTU.

    ``port`` is a serial device, or a ``socket://host:port`` address of a bridge that passes the bytes to one.
    ``force`` and ``speed`` are in percent, until something sends others. A new width goes out at most each
    ``target_period_s``, and only ``target_step`` permille or more away from the last one, or when it is open or closed.
    """

    def __init__(
        self,
        port: str,
        force: int = 100,
        speed: int = 100,
        target_period_s: float = TARGET_PERIOD_S,
        target_step: int = TARGET_STEP,
    ):
        self.port = port
        self.target_period_s = target_period_s
        self.target_step = target_step
        self.grip = pimm.ControlSystemEmitter[float](self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self.sync_move = pimm.calls.ControlSystemHandler[float, None](self)
        self.force = pimm.DefaultingReceiver(self, default=force)
        self.speed = pimm.DefaultingReceiver(self, default=speed)
        # The registers as last written: each write is a bus round trip, so an unchanged value is not sent again
        self._written: dict[int, int] = {}
        self._target_written_at = float('-inf')

    @staticmethod
    def _read(client, register: int) -> int:
        response = client.read_holding_registers(register, count=1, device_id=_DEVICE)
        if response.isError():
            raise ModbusException(f'the gripper refused a read of {register:#x}: {response}')
        return response.registers[0]

    def _connect(self, client, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        """Connect ``client``, yielding until it is connected; then calibrate the gripper if it has no calibration."""
        # A bridge that the arm driver starts on its control box takes some seconds to open its port
        deadline = clock.now() + _CONNECT_TIMEOUT_S
        while not client.connect():
            if clock.now() > deadline:
                client.close()
                raise ConnectionError(f'no DH gripper at {self.port} after {_CONNECT_TIMEOUT_S} s')
            yield pimm.Sleep(_CONNECT_RETRY_S)
        self._written.clear()
        yield from self._initialize(client)

    def _initialize(self, client) -> Iterator[pimm.Sleep]:
        """Run the gripper's calibration if it has none, yielding until it reports itself ready."""
        if self._read(client, _INIT_STATE) != 1:
            client.write_register(_INIT, _START_INIT, device_id=_DEVICE)
            while self._read(client, _INIT_STATE) != 1:
                yield pimm.Sleep(0.1)

    def _width(self, client) -> float:
        """How closed the fingers read back."""
        return aperture.flip_grip(self._read(client, _WIDTH) / 1000)

    def _write(self, client, register: int, value: int) -> None:
        value = c_uint16(value).value
        if self._written.get(register) != value:
            client.write_register(register, value, device_id=_DEVICE)
            self._written[register] = value

    def _command(self, client, grip: float, now: float) -> None:
        """Put ``grip`` on the fingers, at the force and speed asked for."""
        self._write(client, _FORCE, self.force.value)
        self._write(client, _SPEED, self.speed.value)
        target = round(aperture.flip_grip(grip) * 1000)
        last = self._written.get(_TARGET)
        far = last is None or abs(target - last) >= self.target_step
        if target in (0, 1000) or (far and now - self._target_written_at >= self.target_period_s):
            if target != last:
                self._target_written_at = now
            self._write(client, _TARGET, target)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        client = ModbusClient.ModbusSerialClient(port=self.port, baudrate=115200, bytesize=8, parity='N', stopbits=1)
        yield from self._connect(client, clock)

        last_grip = 0.0

        # TODO: Should we translate these to physical units (N and m/s)?
        try:
            with Moves[float](self.sync_move, self.target_grip) as moves:
                while not should_stop.value:
                    try:
                        current_grip = self._width(client)
                        self.grip.emit(current_grip)

                        target = grip_setpoint(moves, current_grip, clock.now())
                        if target is not None:
                            last_grip = target
                        self._command(client, last_grip, clock.now())
                        moves.answer()  # the width a settled move is answered with is on the fingers
                    # A lost link ends in a serial error, an OSError: a bridge drops its client when another one comes
                    except (ModbusException, OSError) as e:
                        logger.warning(f'{self.port}: the link to the gripper broke ({e}), connecting again')
                        client.close()
                        yield from self._connect(client, clock)
                        continue

                    yield pimm.Sleep(0.001)  # Small delay to prevent busy-waiting

                if moves.active:  # the fingers push at the last width written to them, run or no run
                    self._write(client, _TARGET, round(aperture.flip_grip(self._width(client)) * 1000))
        finally:
            client.close()


if __name__ == '__main__':
    import numpy as np

    with pimm.World() as world:
        gripper = DHGripper('/dev/ttyUSB0')

        speed = world.pair(gripper.speed)
        force = world.pair(gripper.force)
        target_grip = world.pair(gripper.target_grip)
        grip = world.pair(gripper.grip)

        world.start([], background=gripper)

        print('Setting gripper to 20% speed and 100% force', flush=True)
        speed.emit(20)
        force.emit(100)

        for width in np.sin(np.linspace(0, 10 * np.pi, 60)) + 1:
            target_grip.emit(width)
            time.sleep(0.5)
            try:
                print(f'Real grip position: {grip.value}')
            except pimm.NoValueException:
                pass
