"""Robotiq 2F-85 (and 2F-140) Modbus RTU driver (RS-485)."""

import logging
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from enum import Enum

import pimm
from positronic.drivers import vendor_import
from positronic.drivers.utils import DriverRun, RecoveryOutcome, grip_setpoint

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

    def claims_arrival_farther_than(self, tolerance: int) -> bool:
        """The gripper reports its fingers at the position request, and their position is more than ``tolerance``
        counts from it.

        A gripper that does not obey reports this, with ``gFLT`` at zero.
        """
        s = self.status
        far = abs(self.position - self.requested) > tolerance
        return s.going and s.fingers is Fingers.AT_REQUEST and far

    def __str__(self) -> str:
        s = self.status
        return (
            f'gACT={s.activated:d} gGTO={s.going:d} gSTA={s.activation.name} gOBJ={s.fingers.name} '
            f'gFLT=0x{s.fault:02X} gPR={self.requested} gPO={self.position} gCU={self.current}'
        )


@dataclass(frozen=True)
class RobotiqTuning:
    """How long one gripper's activation may take, and how far from a request its fingers may stop."""

    activation_timeout_s: float  # the reset and the activation together
    arrival_tolerance: int  # in position counts


TUNING_2F85 = RobotiqTuning(
    # A stuck gripper pushed at full current for about 9 s before its activation stroke broke free.
    activation_timeout_s=30.0,
    # A 2F-85 shut on nothing reads 229 against a request of 255.
    arrival_tolerance=50,
)


class _Gripper(DriverRun[float]):
    """The gripper the driver drives: its Modbus client, and the status it last reported.

    Logs each change of that status, and warns when the fingers claim the request far from it.
    """

    def __init__(
        self,
        client,
        sync_move: pimm.calls.ControlSystemHandler[float, None],
        async_move: pimm.SignalReceiver[float],
        out: pimm.SignalEmitter[float],
        tuning: RobotiqTuning,
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
    ):
        super().__init__(sync_move, async_move, should_stop, clock, hz=200)  # the manual allows 200 Hz
        self._client = client
        self._out = out
        self._tuning = tuning
        self._status: Status | None = None
        self._disobeys = False

    def observe(self) -> Reading:
        """Read the gripper, publish how closed the fingers are, and log what changed."""
        reading = Reading.decode(self._client.read_input_registers(_REG_IN, count=3, device_id=_SLAVE).registers)
        if reading.status != self._status:
            self._status = reading.status
            level = logging.WARNING if reading.status.fault else logging.INFO
            logger.log(level, 'The Robotiq gripper reports %s', reading)
        disobeys = reading.claims_arrival_farther_than(self._tuning.arrival_tolerance)
        if disobeys and not self._disobeys:
            logger.warning('The Robotiq gripper reports its fingers at the request, far from it: %s', reading)
        self._disobeys = disobeys
        self._out.emit(reading.width)
        return reading

    def go_to(self, grip: float, speed: int, force: int) -> None:
        self._write([0x0900, int(grip * 255), (speed << 8) | force])

    def activate(self) -> Generator[pimm.Command, None, RecoveryOutcome]:
        """Reset the gripper, activate it once it reports the reset, and yield until the activation ends.

        CLEARED once the activation completes with no fault. The rising edge of ``rACT`` clears a major fault,
        and the fingers stroke fully open and closed.
        """
        deadline = self.clock.now() + self._tuning.activation_timeout_s
        self._write([0x0000, 0x0000, 0x0000])
        outcome = RecoveryOutcome.NOT_CLEARED
        if (yield from self._await(Activation.RESET, deadline)) is not None:
            self._write([0x0100, 0x0000, 0x0000])
            status = yield from self._await(Activation.COMPLETED, deadline)
            if status is not None and not status.fault:
                outcome = RecoveryOutcome.CLEARED
        level = logging.INFO if outcome is RecoveryOutcome.CLEARED else logging.WARNING
        logger.log(level, 'The Robotiq gripper activation returned %s', outcome.name)
        return outcome

    def _await(self, activation: Activation, deadline: float) -> Generator[pimm.Command, None, Status | None]:
        """Yield until the gripper reports ``activation``. None when the deadline passes or the world stops."""
        while not self.should_stop.value and self.clock.now() < deadline:
            yield self.limiter.wait()
            if (status := self.observe().status).activation is activation:
                return status
        return None

    def _write(self, registers: list[int]) -> None:
        self._client.write_registers(_REG_CMD, registers, device_id=_SLAVE)


class Robotiq2F(pimm.ControlSystem):
    def __init__(self, port: str, tuning: RobotiqTuning = TUNING_2F85):
        self._port = port
        self._tuning = tuning
        self.grip = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self.sync_move = pimm.calls.ControlSystemHandler[float, None](self)
        # Re-activates the gripper: the fingers stroke fully open and closed, and drop what they hold.
        self.recover = pimm.calls.ControlSystemHandler[None, RecoveryOutcome](self)
        self.force = pimm.DefaultingReceiver(self, default=255)  # device scale 0..255
        self.speed = pimm.DefaultingReceiver(self, default=255)  # device scale 0..255

    def _command(self, gripper: _Gripper, grip: float) -> None:
        """Put ``grip`` on the fingers, at the force and speed asked for."""
        gripper.go_to(grip, int(max(0, min(255, self.speed.value))), int(max(0, min(255, self.force.value))))

    def _reactivate(
        self, gripper: _Gripper, asked: list[pimm.calls.Call[None, RecoveryOutcome]]
    ) -> Iterator[pimm.Command]:
        """Activate the gripper again, then answer every call in ``asked`` with how the activation ended.

        A throw reaches every call in ``asked``, then ends the run.
        """
        logger.info('A console asked to re-activate the Robotiq gripper')
        try:
            outcome = yield from gripper.activate()
        except Exception as exc:
            for call in asked:
                call.set_exception(exc)
            raise
        for call in asked:
            call.set_result(outcome)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        client = ModbusClient.ModbusSerialClient(
            port=self._port, baudrate=_BAUD_RATE, bytesize=_BYTESIZE, parity=_PARITY, stopbits=_STOPBITS
        )
        assert client.connect(), f'Failed to connect to Robotiq gripper at {self._port}'
        gripper = _Gripper(client, self.sync_move, self.target_grip, self.grip, self._tuning, should_stop, clock)

        try:
            yield from gripper.activate()

            with gripper.moves as moves:
                while not should_stop.value:
                    grip = gripper.observe().width
                    if asked := list(self.recover.incoming()):
                        yield from self._reactivate(gripper, asked)
                        continue

                    target = grip_setpoint(moves, grip, clock.now())
                    if target is not None:
                        self._command(gripper, target)
                    moves.answer()  # the width a settled move is answered with is on the fingers

                    yield gripper.limiter.wait()

                if moves.active:  # the fingers push at the last width written to them, run or no run
                    self._command(gripper, gripper.observe().width)
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
