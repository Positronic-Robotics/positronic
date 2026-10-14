"""Robotiq 2F-85 (and 2F-140) Modbus RTU driver (RS-485)."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

import pimm
from positronic.drivers import vendor_import
from positronic.drivers.utils import Moves, RecoveryOutcome, grip_setpoint

with vendor_import('pymodbus', 'Gripper support'):
    import pymodbus.client as ModbusClient

logger = logging.getLogger(__name__)

_REG_CMD = 0x03E8
_REG_IN = 0x07D0
_SLAVE = 9
_BAUD_RATE = 115200
_BYTESIZE = 8
_PARITY = 'N'
_STOPBITS = 1
_ACTIVATION_TIMEOUT_S = 10.0


class Activation(Enum):
    """``gSTA``: how far the gripper is through its activation."""

    RESET = 0
    IN_PROGRESS = 1
    NOT_USED = 2
    COMPLETED = 3


class Fingers(Enum):
    """``gOBJ``: where the fingers stand against the position request. It means nothing while ``gGTO`` is 0."""

    MOVING = 0
    CONTACT_OPENING = 1
    CONTACT_CLOSING = 2
    AT_REQUEST = 3


@dataclass(frozen=True)
class Status:
    """The gripper's status and fault registers."""

    activated: bool  # gACT
    going: bool  # gGTO
    activation: Activation  # gSTA
    fingers: Fingers  # gOBJ
    fault: int  # kFLT in the high nibble, gFLT in the low one


@dataclass(frozen=True)
class Reading:
    """One read of the gripper's input registers, in the manual's terms."""

    status: Status
    requested: int  # gPR, the position request the gripper took
    position: int  # gPO
    current: int  # gCU

    @classmethod
    def decode(cls, registers: list[int]) -> 'Reading':
        """The three registers from ``0x07D0``: each holds two bytes of the manual, the first byte high."""
        flags, fault_and_request, position_and_current = registers
        status_byte = flags >> 8
        status = Status(
            activated=bool(status_byte & 0x01),
            going=bool(status_byte & 0x08),
            activation=Activation(status_byte >> 4 & 0x03),
            fingers=Fingers(status_byte >> 6),
            fault=fault_and_request >> 8,
        )
        return cls(status, fault_and_request & 0xFF, position_and_current >> 8, position_and_current & 0xFF)

    @property
    def width(self) -> float:
        """How closed the fingers read back."""
        return self.position / 255.0

    def __str__(self) -> str:
        s = self.status
        return (
            f'gACT={s.activated:d} gGTO={s.going:d} gSTA={s.activation.name} gOBJ={s.fingers.name} '
            f'gFLT=0x{s.fault:02X} gPR={self.requested} gPO={self.position} gCU={self.current}'
        )


class _Gripper:
    """A gripper on its Modbus client. Logs each change of the status it reports."""

    def __init__(self, client):
        self._client = client
        self._status: Status | None = None

    def read(self) -> Reading:
        reading = Reading.decode(self._client.read_input_registers(_REG_IN, count=3, device_id=_SLAVE).registers)
        if reading.status != self._status:
            self._status = reading.status
            level = logging.WARNING if reading.status.fault else logging.INFO
            logger.log(level, 'The Robotiq gripper reports %s', reading)
        return reading

    def activate(self) -> None:
        """Reset the gripper, then activate it. The rising edge of ``rACT`` clears a major fault."""
        self._client.write_registers(_REG_CMD, [0x0000, 0x0000, 0x0000], device_id=_SLAVE)
        self._client.write_registers(_REG_CMD, [0x0100, 0x0000, 0x0000], device_id=_SLAVE)

    def go_to(self, grip: float, speed: int, force: int) -> None:
        self._client.write_registers(_REG_CMD, [0x0900, int(grip * 255), (force << 8) | speed], device_id=_SLAVE)


class Robotiq2F(pimm.ControlSystem):
    def __init__(self, port: str):
        self._port = port
        self.grip = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self.sync_move = pimm.calls.ControlSystemHandler[float, None](self)
        # Re-activates the gripper: the fingers stroke fully open and closed, and drop what they hold.
        self.recover = pimm.calls.ControlSystemHandler[None, RecoveryOutcome](self)
        self.force = pimm.DefaultingReceiver(self, default=255)  # device scale 0..255
        self.speed = pimm.DefaultingReceiver(self, default=255)  # device scale 0..255

    def _observe(self, gripper: _Gripper) -> Reading:
        """Read the gripper and publish how closed the fingers are."""
        reading = gripper.read()
        self.grip.emit(reading.width)
        return reading

    def _command(self, gripper: _Gripper, grip: float) -> None:
        """Put ``grip`` on the fingers, at the force and speed asked for."""
        gripper.go_to(grip, int(max(0, min(255, self.speed.value))), int(max(0, min(255, self.force.value))))

    def _reactivate(
        self,
        gripper: _Gripper,
        asked: list[pimm.calls.Call[None, RecoveryOutcome]],
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
        limiter: pimm.RateLimiter,
    ) -> Iterator[pimm.Command]:
        """Reset and activate the gripper, then answer every call in ``asked``.

        CLEARED needs the gripper to leave its completed activation, to come back to it and to hold no fault.
        """
        logger.info('A console asked to re-activate the Robotiq gripper')
        gripper.activate()
        deadline = clock.now() + _ACTIVATION_TIMEOUT_S
        left_completed = False
        outcome = RecoveryOutcome.NOT_CLEARED
        while not should_stop.value and clock.now() < deadline:
            yield limiter.wait()
            status = self._observe(gripper).status
            if status.activation is not Activation.COMPLETED:
                left_completed = True
            elif left_completed:
                outcome = RecoveryOutcome.NOT_CLEARED if status.fault else RecoveryOutcome.CLEARED
                break
        logger.info(f'The Robotiq gripper re-activation returned {outcome.name}')
        for call in asked:
            call.set_result(outcome)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        client = ModbusClient.ModbusSerialClient(
            port=self._port, baudrate=_BAUD_RATE, bytesize=_BYTESIZE, parity=_PARITY, stopbits=_STOPBITS
        )
        assert client.connect(), f'Failed to connect to Robotiq gripper at {self._port}'
        gripper = _Gripper(client)

        try:
            limiter = pimm.RateLimiter(clock, hz=200)  # According to the manual, the gripper can handle 200Hz
            gripper.activate()

            with Moves[float](self.sync_move, self.target_grip) as moves:
                while not should_stop.value:
                    grip = self._observe(gripper).width
                    if asked := list(self.recover.incoming()):
                        yield from self._reactivate(gripper, asked, should_stop, clock, limiter)
                        continue

                    target = grip_setpoint(moves, grip, clock.now())
                    if target is not None:
                        self._command(gripper, target)
                    moves.answer()  # the width a settled move is answered with is on the fingers

                    yield limiter.wait()

                if moves.active:  # the fingers push at the last width written to them, run or no run
                    self._command(gripper, gripper.read().width)
        finally:
            client.close()


if __name__ == '__main__':
    import time

    with pimm.World() as world:
        gr = Robotiq2F(port='/dev/ttyUSB0')

        spd = world.pair(gr.speed)
        frc = world.pair(gr.force)
        tgt = world.pair(gr.target_grip)
        grip = world.pair(gr.grip)

        world.start([], background=gr)

        spd.emit(128)
        frc.emit(128)

        start = time.time()
        waypoints = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0]
        i = 0

        while True:
            if time.time() - start > i * 1.0:
                tgt.emit(waypoints[i])
                i += 1
                if i >= len(waypoints):
                    break
            time.sleep(0.1)
            try:
                print(f'[{i}] Grip: {grip.value:.2f}')
            except pimm.NoValueException:
                pass
