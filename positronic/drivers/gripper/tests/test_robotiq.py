"""What the Robotiq driver logs of the gripper's status, and how it re-activates the gripper when asked."""

import logging
from types import SimpleNamespace

import pytest

import pimm
from pimm.tests.testing import MockClock, wire_call
from positronic.drivers.gripper import robotiq
from positronic.drivers.gripper.robotiq import Activation, Fingers
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.drivers.utils import RecoveryOutcome
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

ACTIVATION_READS = 3
STEP = 40
# Where the fingers stop when they close on nothing, and where they close on the object, in position counts.
SHUT = 229
OBJECT = 200
TICK_S = 0.005


class FakeBus:
    """A Robotiq 2F on its Modbus registers, in memory. Each read is one step of the device.

    A fall of ``rACT`` resets the gripper. A rise starts an activation of ``activation_reads`` reads, which
    leaves the fingers open, obeying again, and with the fault byte ``fault_after_activation``. With
    ``completes`` off the activation never ends. A position request moves the fingers ``STEP`` counts a read,
    to no more than ``SHUT``, and closing stops at ``object_at``. With ``obeying`` off the fingers stay where
    they stand, and the gripper reports them at the request.
    """

    def __init__(self, object_at: int | None = None):
        self.writes: list[list[int]] = []
        self.object_at = object_at
        self.obeying = True
        self.completes = True
        self.activation_reads = ACTIVATION_READS
        self.fault_after_activation = 0
        self.fault = 0
        self.position = 0
        self._act = False
        self._go = False
        self._requested = 0
        self._activation = Activation.RESET
        self._activation_left = 0
        self._fingers = Fingers.AT_REQUEST

    def connect(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def write_registers(self, address: int, values: list[int], device_id: int) -> None:
        assert (address, device_id) == (0x03E8, 9)
        self.writes.append(list(values))
        action = values[0] >> 8
        act, self._go = bool(action & 0x01), bool(action & 0x08)
        if act and not self._act:
            self._activation, self._activation_left = Activation.IN_PROGRESS, self.activation_reads
        elif not act:
            self._activation = Activation.RESET
        self._act = act
        if self._go:
            self._requested = values[1] & 0xFF
            self._fingers = Fingers.MOVING

    def read_input_registers(self, address: int, count: int, device_id: int) -> SimpleNamespace:
        assert (address, count, device_id) == (0x07D0, 3, 9)
        self._step()
        status = self._fingers.value << 6 | self._activation.value << 4 | self._go << 3 | self._act
        return SimpleNamespace(registers=[status << 8, self.fault << 8 | self._requested, self.position << 8])

    def _step(self) -> None:
        if self._activation is Activation.IN_PROGRESS:
            if self.completes:
                self._activation_left -= 1
                if self._activation_left == 0:
                    self._activation, self.position, self.obeying = Activation.COMPLETED, 0, True
                    self.fault = self.fault_after_activation
        elif self._fingers is Fingers.MOVING and not self.obeying:
            self._fingers = Fingers.AT_REQUEST
        elif self._fingers is Fingers.MOVING:
            goal = min(self._requested, SHUT)
            step = max(-STEP, min(STEP, goal - self.position))
            self.position += step
            if self.object_at is not None and step > 0 and self.position >= self.object_at:
                self.position, self._fingers = self.object_at, Fingers.CONTACT_CLOSING
            elif self.position == goal:
                self._fingers = Fingers.AT_REQUEST


@pytest.fixture
def world():
    with pimm.World() as w:
        yield w


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger=robotiq.__name__)
    return caplog


def _driven(monkeypatch, bus: FakeBus, clock: MockClock):
    """A driver over ``bus``, the receiver its grip requests arrive on, and its loop past the activation."""
    monkeypatch.setattr(robotiq, 'ModbusClient', SimpleNamespace(ModbusSerialClient=lambda **_kwargs: bus))
    driver = robotiq.Robotiq2F(port='/dev/null')
    driver.grip._bind(RecordingEmitter(), clock=clock)
    requests = ManualCommandReceiver()
    driver.target_grip._bind(requests)
    loop = driver.run(StopFlag(), clock)
    _pump(loop, clock, ACTIVATION_READS + 2)  # the reset, its report, then the activation
    return driver, requests, loop


def _pump(loop, clock: MockClock, ticks: int, seconds: float = TICK_S) -> None:
    for _ in range(ticks):
        next(loop)
        clock.advance(seconds)


def _recoverer(world: pimm.World, driver: robotiq.Robotiq2F) -> pimm.calls.Caller[None, RecoveryOutcome]:
    """A caller on ``driver.recover``, wired for a test that pumps the driver's generator itself."""
    caller = pimm.calls.ControlSystemCaller[None, RecoveryOutcome](driver)
    wire_call(world, caller, driver.recover)
    return caller


def _stop_obeying_on_the_object(bus: FakeBus, requests: ManualCommandReceiver, loop, clock: MockClock) -> None:
    """Close the fingers on the object, then ask them open after they stopped obeying."""
    requests.push(1.0)
    _pump(loop, clock, 10)
    bus.obeying = False
    requests.push(0.0)
    _pump(loop, clock, 3)


def _warnings(logs) -> list[str]:
    return [record.getMessage() for record in logs.records if record.levelno == logging.WARNING]


def test_fingers_that_claim_the_request_far_from_it_log_a_warning(monkeypatch, logs):
    """The fingers hold an object and take an open request without moving. The gripper reports them at the
    request with no fault, and the log warns once, with the request and the position."""
    bus, clock = FakeBus(object_at=OBJECT), MockClock()
    _, requests, loop = _driven(monkeypatch, bus, clock)

    _stop_obeying_on_the_object(bus, requests, loop, clock)
    _pump(loop, clock, 10)

    assert len(_warnings(logs)) == 1
    assert f'gOBJ=AT_REQUEST gFLT=0x00 gPR=0 gPO={OBJECT}' in _warnings(logs)[0]


def test_fingers_shut_on_nothing_or_stopped_on_an_object_log_no_warning(monkeypatch, logs):
    """A full close stops short of the request, and an object stops the fingers far from it: neither is a fault."""
    bus, clock = FakeBus(), MockClock()
    _, requests, loop = _driven(monkeypatch, bus, clock)

    requests.push(1.0)
    _pump(loop, clock, 10)
    assert f'gOBJ=AT_REQUEST gFLT=0x00 gPR=255 gPO={SHUT}' in logs.text
    bus.object_at = OBJECT
    requests.push(0.0)
    _pump(loop, clock, 10)
    requests.push(1.0)
    _pump(loop, clock, 10)

    assert f'gOBJ=CONTACT_CLOSING gFLT=0x00 gPR=255 gPO={OBJECT}' in logs.text
    assert _warnings(logs) == []


def test_the_log_takes_a_line_per_status_and_none_per_position(monkeypatch, logs):
    bus, clock = FakeBus(), MockClock()
    _, requests, loop = _driven(monkeypatch, bus, clock)
    logs.clear()

    requests.push(1.0)
    _pump(loop, clock, 20)

    lines = [record.getMessage() for record in logs.records]
    assert len(lines) == 2
    assert 'gOBJ=MOVING' in lines[0]
    assert 'gOBJ=AT_REQUEST' in lines[1] and f'gPO={SHUT}' in lines[1]


def test_a_fault_is_logged_as_a_warning(monkeypatch, logs):
    bus, clock = FakeBus(), MockClock()
    _, _, loop = _driven(monkeypatch, bus, clock)

    bus.fault = 0x0E
    _pump(loop, clock, 2)

    assert len(_warnings(logs)) == 1 and 'gFLT=0x0E' in _warnings(logs)[0]


def test_recover_raises_ract_only_once_the_gripper_reports_the_reset(monkeypatch, world):
    bus, clock = FakeBus(object_at=OBJECT), MockClock()
    driver, requests, loop = _driven(monkeypatch, bus, clock)
    _stop_obeying_on_the_object(bus, requests, loop, clock)
    before = len(bus.writes)

    answer = _recoverer(world, driver)(None)
    _pump(loop, clock, 1)
    assert bus.writes[before:] == [[0x0000, 0x0000, 0x0000]]
    _pump(loop, clock, 1)
    assert bus.writes[before:] == [[0x0000, 0x0000, 0x0000], [0x0100, 0x0000, 0x0000]]

    assert not answer.done()
    _pump(loop, clock, ACTIVATION_READS)
    assert answer.result() is RecoveryOutcome.CLEARED

    requests.push(0.0)
    _pump(loop, clock, 10)
    assert bus.position == 0, 'the fingers did not open after the re-activation'


def test_recover_waits_out_an_activation_that_takes_seconds(monkeypatch, world):
    """A stuck gripper pushes for seconds before its activation stroke breaks free."""
    bus, clock = FakeBus(object_at=OBJECT), MockClock()
    driver, requests, loop = _driven(monkeypatch, bus, clock)
    _stop_obeying_on_the_object(bus, requests, loop, clock)
    bus.activation_reads = 120

    answer = _recoverer(world, driver)(None)
    _pump(loop, clock, 2 + bus.activation_reads, seconds=0.1)

    assert answer.result() is RecoveryOutcome.CLEARED


def test_recover_answers_not_cleared_when_the_activation_does_not_complete(monkeypatch, world):
    bus, clock = FakeBus(), MockClock()
    driver, _, loop = _driven(monkeypatch, bus, clock)
    bus.completes = False
    timeout = int(robotiq._ACTIVATION_TIMEOUT_S)

    answer = _recoverer(world, driver)(None)
    _pump(loop, clock, timeout - 1, seconds=1.0)
    assert not answer.done(), 'the answer came before the activation had its time'
    _pump(loop, clock, 2, seconds=1.0)

    assert answer.result() is RecoveryOutcome.NOT_CLEARED


def test_recover_answers_not_cleared_when_the_activation_ends_in_a_fault(monkeypatch, world):
    bus, clock = FakeBus(), MockClock()
    driver, _, loop = _driven(monkeypatch, bus, clock)
    bus.fault_after_activation = 0x0D

    answer = _recoverer(world, driver)(None)
    _pump(loop, clock, ACTIVATION_READS + 2)

    assert answer.result() is RecoveryOutcome.NOT_CLEARED
