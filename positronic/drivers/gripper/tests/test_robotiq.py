"""What the Robotiq driver logs of the gripper's status, how it re-activates the gripper when asked, and how it
holds a lost gripper until a call activates it again."""

import logging
import os
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import pytest

import pimm
from pimm.tests.testing import MockClock, wire_call
from positronic.drivers.faults import DeviceFault, FaultKind
from positronic.drivers.gripper import robotiq, robotiq_grip_probe
from positronic.drivers.gripper.robotiq import Activation, Fingers
from positronic.drivers.gripper.robotiq_grip_probe import GripRecord, RobotiqGripProbe
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
    they stand, and the gripper reports them at the request. A read raises ``raises`` once it is set, and
    answers with an exception response while ``refuses`` is on, and a write while ``refuses_writes`` is on.
    ``connect`` answers ``answers``. Each read reports ``current`` as ``gCU``.
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
        self.current = 0
        self.raises: Exception | None = None
        self.refuses = False
        self.refuses_writes = False
        self.answers = True
        self._act = False
        self._go = False
        self._requested = 0
        self._activation = Activation.RESET
        self._activation_left = 0
        self._fingers = Fingers.AT_REQUEST

    def connect(self) -> bool:
        return self.answers

    def close(self) -> None:
        pass

    def write_registers(self, address: int, values: list[int], device_id: int) -> SimpleNamespace:
        assert (address, device_id) == (0x03E8, 9)
        if self.refuses_writes:
            return SimpleNamespace(isError=lambda: True)
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
        return SimpleNamespace(isError=lambda: False)

    def read_input_registers(self, address: int, count: int, device_id: int) -> SimpleNamespace:
        assert (address, count, device_id) == (0x07D0, 3, 9)
        if self.raises is not None:
            raise self.raises
        if self.refuses:
            return SimpleNamespace(registers=[], isError=lambda: True)
        self._step()
        status = self._fingers.value << 6 | self._activation.value << 4 | self._go << 3 | self._act
        registers = [status << 8, self.fault << 8 | self._requested, self.position << 8 | self.current]
        return SimpleNamespace(registers=registers, isError=lambda: False)

    def _step(self) -> None:
        if self._activation is Activation.IN_PROGRESS:
            if self.completes:
                self._activation_left -= 1
                if self._activation_left == 0:
                    self._activation, self.fault = Activation.COMPLETED, self.fault_after_activation
                    self.position, self.obeying = 0, True
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


def _driven(monkeypatch, bus: FakeBus, clock: MockClock, widths: RecordingEmitter | None = None):
    """A driver over ``bus``, the receiver its grip requests arrive on, and its loop past the activation.

    The driver emits its widths to ``widths``.
    """
    monkeypatch.setattr(robotiq, 'ModbusClient', SimpleNamespace(ModbusSerialClient=lambda **_kwargs: bus))
    driver = robotiq.Robotiq2F(port='/dev/null')
    driver.grip._bind(widths or RecordingEmitter(), clock=clock)
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
    timeout = int(robotiq.TUNING_2F85.activation_timeout_s)

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


def test_speed_and_force_reach_their_own_bytes(monkeypatch):
    """``rSP`` is the high byte of the third command register, and ``rFR`` the low one."""
    bus, clock = FakeBus(), MockClock()
    driver, requests, loop = _driven(monkeypatch, bus, clock)
    speed, force = ManualCommandReceiver(), ManualCommandReceiver()
    driver.speed._bind(speed)
    driver.force._bind(force)

    speed.push(10)
    force.push(200)
    requests.push(1.0)
    _pump(loop, clock, 1)

    assert bus.writes[-1][2] == 10 << 8 | 200


def _readier(world: pimm.World, driver: robotiq.Robotiq2F) -> pimm.calls.Caller[None, None]:
    """A caller on ``driver.ready``, wired for a test that pumps the driver's generator itself."""
    caller = pimm.calls.ControlSystemCaller[None, None](driver)
    wire_call(world, caller, driver.ready)
    return caller


def _mover(world: pimm.World, driver: robotiq.Robotiq2F) -> pimm.calls.Caller[float, None]:
    """A caller on ``driver.sync_move``, wired for a test that pumps the driver's generator itself."""
    caller = pimm.calls.ControlSystemCaller[float, None](driver)
    wire_call(world, caller, driver.sync_move)
    return caller


def _bind_shut(bus: FakeBus, requests: ManualCommandReceiver, loop, clock: MockClock) -> None:
    """Close the fingers on nothing, then stop them obeying, as a gripper bound shut does."""
    requests.push(1.0)
    _pump(loop, clock, 10)
    bus.obeying = False


def test_a_move_on_fingers_bound_shut_holds_a_gripper_fault_and_answers_the_move_with_it(monkeypatch, world):
    """A gripper bound shut takes an open request and reports the fingers at it, with no fault, while they stay
    shut. A move a caller waits on puts a GRIPPER fault on the width and answers with it, and activates nothing."""
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, requests, loop = _driven(monkeypatch, bus, clock, widths)
    _bind_shut(bus, requests, loop, clock)
    before = len(bus.writes)

    move = _mover(world, driver)(0.0)
    _pump(loop, clock, 5)

    with pytest.raises(DeviceFault, match='reports its fingers at the request') as raised:
        move.result()
    assert raised.value.kind is FaultKind.GRIPPER
    assert _errors(widths) == [raised.value]
    assert [0x0100, 0x0000, 0x0000] not in bus.writes[before:]


@pytest.mark.parametrize(('ask', 'cleared'), [(_readier, None), (_recoverer, RecoveryOutcome.CLEARED)])
def test_a_ready_or_recover_call_reactivates_a_gripper_held_for_fingers_bound_shut(monkeypatch, world, ask, cleared):
    bus, clock = FakeBus(), MockClock()
    driver, requests, loop = _driven(monkeypatch, bus, clock)
    _bind_shut(bus, requests, loop, clock)
    mover = _mover(world, driver)
    move = mover(0.0)
    _pump(loop, clock, 5)
    with pytest.raises(DeviceFault):
        move.result()

    answer = ask(world, driver)(None)
    _pump(loop, clock, ACTIVATION_READS + 3)
    assert answer.result() is cleared

    move = mover(0.0)
    _pump(loop, clock, 20)
    assert move.result() is None
    assert bus.position == 0


def test_a_streamed_width_on_fingers_bound_shut_holds_no_fault(monkeypatch):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    _, requests, loop = _driven(monkeypatch, bus, clock, widths)
    _bind_shut(bus, requests, loop, clock)
    before = len(bus.writes)

    requests.push(0.0)
    _pump(loop, clock, 40, seconds=0.1)

    assert bus.position == SHUT
    assert _errors(widths) == []
    actions = [write[0] for write in bus.writes[before:]]
    assert actions and set(actions) == {0x0900}, f'a streamed width started an activation: {actions}'


@pytest.mark.parametrize('object_at', [None, OBJECT])
def test_a_move_that_closes_on_nothing_or_on_an_object_holds_no_fault(monkeypatch, world, object_at):
    """A full close stops short of the request, and an object stops the fingers far from it. The move gives up
    when its time runs out, and the gripper holds no fault."""
    bus, clock, widths = FakeBus(object_at=object_at), MockClock(), RecordingEmitter()
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)

    move = _mover(world, driver)(1.0)
    _pump(loop, clock, 40, seconds=0.1)

    with pytest.raises(TimeoutError, match='short of 1.0'):
        move.result()
    assert _errors(widths) == []
    assert bus.position == (SHUT if object_at is None else OBJECT)


def _last(widths: RecordingEmitter):
    return widths.emitted[-1][1]


def _errors(widths: RecordingEmitter) -> list[pimm.SignalError]:
    return [data for _, data in widths.emitted if isinstance(data, pimm.SignalError)]


class BusFault(Enum):
    """How the bus fails the driver."""

    NO_ANSWER = auto()
    PORT_GONE = auto()
    EXCEPTION_RESPONSE = auto()


def _lose(bus: FakeBus, fault: BusFault) -> None:
    """Make every read fail the way ``fault`` names."""
    match fault:
        case BusFault.NO_ANSWER:
            bus.raises = robotiq.ModbusException('No response received after 3 retries')
        case BusFault.PORT_GONE:
            bus.raises = OSError('[Errno 5] Input/output error')
        case BusFault.EXCEPTION_RESPONSE:
            bus.refuses = True


def _restore(bus: FakeBus) -> None:
    bus.raises, bus.refuses = None, False


@pytest.mark.parametrize('fault', list(BusFault))
def test_a_lost_bus_puts_an_error_on_the_width_and_the_run_goes_on(monkeypatch, world, fault):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)
    move = _mover(world, driver)(1.0)
    _pump(loop, clock, 1)

    _lose(bus, fault)
    _pump(loop, clock, 5)

    assert len(_errors(widths)) == 1
    error = _last(widths)
    assert isinstance(error, DeviceFault) and error.kind is FaultKind.GRIPPER
    with pytest.raises(DeviceFault, match='The Robotiq gripper is lost') as raised:
        move.result()
    assert raised.value.kind is FaultKind.GRIPPER


def test_a_write_the_gripper_refuses_puts_an_error_on_the_width_and_answers_the_move(monkeypatch, world):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)
    _pump(loop, clock, 1)

    bus.refuses_writes = True
    move = _mover(world, driver)(1.0)
    _pump(loop, clock, 3)

    assert len(_errors(widths)) == 1
    error = _last(widths)
    assert isinstance(error, DeviceFault) and error.kind is FaultKind.GRIPPER
    with pytest.raises(DeviceFault, match='refused a write') as raised:
        move.result()
    assert raised.value.kind is FaultKind.GRIPPER


def test_a_move_on_a_lost_gripper_is_answered_with_its_error_and_reaches_no_register(monkeypatch, world):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, requests, loop = _driven(monkeypatch, bus, clock, widths)
    _lose(bus, BusFault.NO_ANSWER)
    _pump(loop, clock, 1)
    before = len(bus.writes)

    move = _mover(world, driver)(1.0)
    requests.push(0.5)
    _pump(loop, clock, 3)

    with pytest.raises(pimm.SignalError, match='The Robotiq gripper is lost'):
        move.result()
    assert bus.writes[before:] == []


@pytest.mark.parametrize('fault', list(BusFault))
def test_a_ready_call_activates_a_lost_gripper_and_the_width_comes_back(monkeypatch, world, fault):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, requests, loop = _driven(monkeypatch, bus, clock, widths)
    _lose(bus, fault)
    _pump(loop, clock, 3)
    _restore(bus)

    answer = _readier(world, driver)(None)
    _pump(loop, clock, ACTIVATION_READS + 3)

    assert answer.result() is None
    assert not isinstance(_last(widths), pimm.SignalError)
    requests.push(1.0)
    _pump(loop, clock, 10)
    assert bus.position == SHUT, 'the fingers do not move after the repair'


def test_a_ready_call_on_a_bus_that_stays_down_answers_the_error_and_the_next_one_repairs(monkeypatch, world):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)
    _lose(bus, BusFault.NO_ANSWER)
    _pump(loop, clock, 1)

    ready = _readier(world, driver)
    first = ready(None)
    _pump(loop, clock, 2)

    with pytest.raises(pimm.SignalError, match='No response received'):
        first.result()
    assert isinstance(_last(widths), pimm.SignalError)

    _restore(bus)
    second = ready(None)
    _pump(loop, clock, ACTIVATION_READS + 3)
    assert second.result() is None


def test_a_ready_call_on_a_lost_gripper_that_activates_with_a_fault_answers_an_error(monkeypatch, world):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)
    _lose(bus, BusFault.NO_ANSWER)
    _pump(loop, clock, 1)
    _restore(bus)
    bus.fault_after_activation = 0x0D

    answer = _readier(world, driver)(None)
    _pump(loop, clock, ACTIVATION_READS + 3)

    with pytest.raises(pimm.SignalError, match='did not activate'):
        answer.result()
    assert isinstance(_last(widths), pimm.SignalError)


def test_a_ready_call_on_a_gripper_that_answers_is_answered_at_once(monkeypatch, world):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)
    before = len(bus.writes)

    answer = _readier(world, driver)(None)
    _pump(loop, clock, 1)

    assert answer.result() is None
    assert bus.writes[before:] == [], 'a ready call re-activated a gripper that answers'


def test_a_gripper_absent_at_start_holds_an_error_until_a_ready_call_opens_it(monkeypatch, world):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    bus.answers = False
    monkeypatch.setattr(robotiq, 'ModbusClient', SimpleNamespace(ModbusSerialClient=lambda **_kwargs: bus))
    driver = robotiq.Robotiq2F(port='/dev/null')
    driver.grip._bind(widths, clock=clock)
    loop = driver.run(StopFlag(), clock)
    _pump(loop, clock, 3)

    assert [str(error) for error in _errors(widths)] == [
        'The Robotiq gripper is lost: no Robotiq gripper answers at /dev/null'
    ]

    bus.answers = True
    answer = _readier(world, driver)(None)
    _pump(loop, clock, ACTIVATION_READS + 3)
    assert answer.result() is None


def test_a_bus_error_during_recover_reaches_the_caller_and_the_gripper_holds_it(monkeypatch, world):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)

    answer = _recoverer(world, driver)(None)
    _pump(loop, clock, 1)
    _lose(bus, BusFault.PORT_GONE)
    _pump(loop, clock, 2)

    with pytest.raises(DeviceFault) as answered:
        answer.result()
    assert answered.value.kind is FaultKind.GRIPPER
    assert isinstance(answered.value.__cause__, OSError)
    assert len(_errors(widths)) == 1


def test_a_recover_call_on_a_lost_gripper_activates_it(monkeypatch, world):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)
    _lose(bus, BusFault.NO_ANSWER)
    _pump(loop, clock, 1)
    _restore(bus)

    answer = _recoverer(world, driver)(None)
    _pump(loop, clock, ACTIVATION_READS + 3)

    assert answer.result() is RecoveryOutcome.CLEARED
    assert not isinstance(_last(widths), pimm.SignalError)


def test_a_gripper_that_activates_with_a_fault_at_start_holds_an_error_until_a_ready_call(monkeypatch, world):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    bus.fault_after_activation = 0x0D
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)
    move = _mover(world, driver)(1.0)
    _pump(loop, clock, 2)

    [error] = _errors(widths)
    assert isinstance(error, DeviceFault) and error.kind is FaultKind.GRIPPER
    assert 'did not activate' in str(error)
    with pytest.raises(DeviceFault, match='did not activate'):
        move.result()

    bus.fault_after_activation = 0
    answer = _readier(world, driver)(None)
    _pump(loop, clock, ACTIVATION_READS + 3)
    assert answer.result() is None
    assert not isinstance(_last(widths), pimm.SignalError)


def test_a_gripper_whose_activation_does_not_complete_at_start_holds_an_error(monkeypatch):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    bus.completes = False
    _, _, loop = _driven(monkeypatch, bus, clock, widths)
    _pump(loop, clock, int(robotiq.TUNING_2F85.activation_timeout_s) + 1, seconds=1.0)

    [error] = _errors(widths)
    assert isinstance(error, DeviceFault) and error.kind is FaultKind.GRIPPER
    assert 'did not activate' in str(error)


def test_a_recover_call_whose_activation_ends_in_a_fault_holds_an_error_until_a_ready_call(monkeypatch, world):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)
    bus.fault_after_activation = 0x0D

    recovered = _recoverer(world, driver)(None)
    _pump(loop, clock, ACTIVATION_READS + 2)

    assert recovered.result() is RecoveryOutcome.NOT_CLEARED
    [error] = _errors(widths)
    assert isinstance(error, DeviceFault) and error.kind is FaultKind.GRIPPER
    assert 'did not activate' in str(error)

    bus.fault_after_activation = 0
    answer = _readier(world, driver)(None)
    _pump(loop, clock, ACTIVATION_READS + 3)
    assert answer.result() is None


def test_a_recover_call_that_activates_a_gripper_that_answers_puts_no_error_on_the_width(monkeypatch, world):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)

    recovered = _recoverer(world, driver)(None)
    _pump(loop, clock, ACTIVATION_READS + 2)

    assert recovered.result() is RecoveryOutcome.CLEARED
    assert _errors(widths) == []


def test_a_fault_the_gripper_reports_puts_no_error_on_the_width(monkeypatch):
    """``gFLT`` is a status the gripper reports over a bus that answers, so the width stays a width."""
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    driver, _, loop = _driven(monkeypatch, bus, clock, widths)

    bus.fault = 0x0E
    _pump(loop, clock, 5)

    assert _errors(widths) == []


def test_an_error_that_is_not_the_bus_ends_the_run(monkeypatch):
    bus, clock, widths = FakeBus(), MockClock(), RecordingEmitter()
    _, _, loop = _driven(monkeypatch, bus, clock, widths)

    bus.raises = ValueError('a bug in the driver')

    with pytest.raises(ValueError, match='a bug in the driver'):
        _pump(loop, clock, 1)


PROBE_RECORD_S = 0.5


def _probed(monkeypatch, bus: FakeBus, out, enters: int) -> GripRecord:
    """Run the grip probe over ``bus`` to its end, with ``enters`` pushes of Enter and then the end of the input."""
    monkeypatch.setattr(robotiq, 'ModbusClient', SimpleNamespace(ModbusSerialClient=lambda **_kwargs: bus))
    record = GripRecord(label='pen', speed=255, force=128, record_s=PROBE_RECORD_S)
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b'\n' * enters)
    os.close(write_fd)
    clock = MockClock()
    try:
        for command in RobotiqGripProbe('/dev/null', record, 3, out, read_fd).run(StopFlag(), clock):
            clock.advance(command.seconds if isinstance(command, pimm.Sleep) else 0.0)
    finally:
        os.close(read_fd)
    return record


def test_the_grip_probe_records_each_close_on_an_object_and_writes_its_files(monkeypatch, tmp_path):
    bus = FakeBus(object_at=OBJECT)
    bus.current = 40
    record = _probed(monkeypatch, bus, tmp_path, enters=3)

    assert [close.repeat for close in record.closes] == [1, 2, 3]
    for close in record.closes:
        o = close.outcome
        assert (o.fingers, o.position, o.current, o.detected) == (Fingers.CONTACT_CLOSING, OBJECT, 40, True)
        assert o.fingers_seen == [Fingers.MOVING, Fingers.CONTACT_CLOSING]
        assert o.stopped_s is not None and 0 < o.stopped_s == o.still_s < PROBE_RECORD_S
    assert bus.writes[-1][1] == 0, 'the probe left the fingers closed'

    assert GripRecord.model_validate_json((tmp_path / 'pen.json').read_text()) == record
    csv_lines = (tmp_path / 'pen.csv').read_text().splitlines()
    assert csv_lines[0] == 'repeat,t_s,fingers,requested,position,current'
    assert len(csv_lines) == 1 + sum(len(close.samples) for close in record.closes)
    assert csv_lines[-1].endswith(f',{Fingers.CONTACT_CLOSING.value},255,{OBJECT},40')
    table = (tmp_path / 'pen.txt').read_text().splitlines()
    assert len(table) == 1 + 3 + 1
    assert table[-1] == 'pen: gOBJ is CONTACT_CLOSING at the end of 3 of 3 closes'


def test_the_grip_probe_reports_no_object_for_a_close_on_nothing(monkeypatch, tmp_path):
    record = _probed(monkeypatch, FakeBus(), tmp_path, enters=2)

    for close in record.closes:
        o = close.outcome
        assert (o.fingers, o.position, o.detected) == (Fingers.AT_REQUEST, SHUT, False)
    assert (tmp_path / 'pen.txt').read_text().splitlines()[-1].endswith('at the end of 0 of 2 closes')


def test_the_grip_probe_refuses_a_port_that_does_not_open(monkeypatch, tmp_path):
    bus = FakeBus()
    bus.answers = False

    with pytest.raises(SystemExit, match='A station or a console can hold the port'):
        _probed(monkeypatch, bus, tmp_path, enters=1)
    assert bus.writes == []


def test_the_grip_probe_ends_when_the_input_ends(monkeypatch, tmp_path):
    record = _probed(monkeypatch, FakeBus(object_at=OBJECT), tmp_path, enters=1)

    assert len(record.closes) == 1


def test_the_grip_probe_opens_the_fingers_when_a_write_of_its_files_fails(monkeypatch, tmp_path):
    bus = FakeBus(object_at=OBJECT)

    with pytest.raises(FileNotFoundError):
        _probed(monkeypatch, bus, tmp_path / 'absent', enters=3)
    assert bus.writes[-1][1] == 0, 'the probe left the fingers closed on the object'


def test_the_grip_probe_opens_the_fingers_before_it_writes_its_files(monkeypatch, tmp_path):
    bus = FakeBus(object_at=OBJECT)
    write = robotiq_grip_probe.write
    requested_at_write = []

    def recording_write(record, out):
        requested_at_write.append(bus.writes[-1][1])
        write(record, out)

    monkeypatch.setattr(robotiq_grip_probe, 'write', recording_write)
    _probed(monkeypatch, bus, tmp_path, enters=2)

    assert requested_at_write == [0, 0], 'a slow write kept the fingers closed on the object'


def test_a_stop_at_the_prompt_ends_the_grip_probe_with_the_fingers_open(monkeypatch, tmp_path):
    """A Ctrl-C sets ``should_stop`` while the probe waits for Enter. The probe ends, and the fingers open."""
    bus = FakeBus(object_at=OBJECT)
    monkeypatch.setattr(robotiq, 'ModbusClient', SimpleNamespace(ModbusSerialClient=lambda **_kwargs: bus))
    record = GripRecord(label='pen', speed=255, force=128, record_s=PROBE_RECORD_S)
    read_fd, write_fd = os.pipe()
    stop, clock = StopFlag(), MockClock()
    try:
        for command in RobotiqGripProbe('/dev/null', record, 3, tmp_path, read_fd).run(stop, clock):
            clock.advance(command.seconds if isinstance(command, pimm.Sleep) else 0.0)
            stop.stopped = clock.now() > 10.0
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert record.closes == []
    assert bus.writes[-1][1] == 0


def _stopped_during_the_activation(monkeypatch, bus: FakeBus, out, stop_after_s: float) -> GripRecord:
    """Run the grip probe over ``bus``, and stop it once the clock passes ``stop_after_s``."""
    monkeypatch.setattr(robotiq, 'ModbusClient', SimpleNamespace(ModbusSerialClient=lambda **_kwargs: bus))
    record = GripRecord(label='pen', speed=255, force=128, record_s=PROBE_RECORD_S)
    read_fd, write_fd = os.pipe()
    stop, clock = StopFlag(), MockClock()
    stop.stopped = stop_after_s < 0
    try:
        for command in RobotiqGripProbe('/dev/null', record, 3, out, read_fd).run(stop, clock):
            clock.advance(command.seconds if isinstance(command, pimm.Sleep) else 0.0)
            stop.stopped = clock.now() > stop_after_s
    finally:
        os.close(read_fd)
        os.close(write_fd)
    return record


def test_a_stop_during_the_activation_ends_the_grip_probe_without_an_activation_error(monkeypatch, tmp_path):
    """A Ctrl-C while the gripper activates ends the probe as a stop, not as a gripper that did not activate."""
    bus = FakeBus()
    bus.completes = False
    record = _stopped_during_the_activation(monkeypatch, bus, tmp_path, stop_after_s=1.0)

    assert record.closes == []
    assert bus.writes[-1] == [0x0100, 0, 0], 'the probe asked a move of a gripper that did not complete its activation'


def test_a_stop_before_the_gripper_reports_its_reset_keeps_the_gripper_reset(monkeypatch, tmp_path):
    """A move command raises ``rACT``, so a move after a stop before the reset would start an activation stroke."""
    bus = FakeBus()
    record = _stopped_during_the_activation(monkeypatch, bus, tmp_path, stop_after_s=-1.0)

    assert record.closes == []
    assert bus.writes == [[0, 0, 0]]


@pytest.mark.parametrize(
    ('flag', 'value', 'refusal'),
    [
        ('--record-s', '0', 'is not a positive number of seconds'),
        ('--record-s', '-1', 'is not a positive number of seconds'),
        ('--record-s', 'nan', 'is not a positive number of seconds'),
        ('--repeats', '0', 'is not a positive count'),
        ('--repeats', '-2', 'is not a positive count'),
        ('--speed', '256', 'is not in 0..255'),
        ('--force', '-1', 'is not in 0..255'),
        ('--label', '../pen', 'is not a file name'),
        ('--label', '', 'is not a file name'),
    ],
)
def test_the_grip_probe_refuses_an_argument_out_of_range(monkeypatch, capsys, tmp_path, flag, value, refusal):
    argv = ['probe', '--port', '/dev/null', '--label', 'pen', '--out', str(tmp_path / 'out'), flag, value]
    monkeypatch.setattr('sys.argv', argv)

    with pytest.raises(SystemExit):
        robotiq_grip_probe.main()
    assert refusal in capsys.readouterr().err
    assert not (tmp_path / 'out').exists(), 'the probe started before it checked its arguments'


def test_a_failed_write_keeps_the_files_of_the_closes_before_it(monkeypatch, tmp_path):
    record = _probed(monkeypatch, FakeBus(object_at=OBJECT), tmp_path, enters=1)
    before = {name: (tmp_path / name).read_bytes() for name in ('pen.json', 'pen.csv', 'pen.txt')}
    write_text = Path.write_text

    def torn(path: Path, text: str) -> int:
        write_text(path, text[: len(text) // 2])
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(Path, 'write_text', torn)
    record.closes.append(record.closes[0])
    with pytest.raises(OSError):
        robotiq_grip_probe.write(record, tmp_path)

    assert {name: (tmp_path / name).read_bytes() for name in before} == before


def test_the_grip_probe_prints_each_close_once(monkeypatch, capsys, tmp_path):
    bus = FakeBus(object_at=OBJECT)
    monkeypatch.setattr(robotiq, 'ModbusClient', SimpleNamespace(ModbusSerialClient=lambda **_kwargs: bus))
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b'\n\n')
    os.close(write_fd)
    monkeypatch.setattr('sys.stdin', SimpleNamespace(fileno=lambda: read_fd))
    argv = ['probe', '--port', '/dev/null', '--label', 'pen', '--out', str(tmp_path), '--repeats', '2']
    monkeypatch.setattr('sys.argv', [*argv, '--record-s', '0.05'])
    try:
        robotiq_grip_probe.main()
    finally:
        os.close(read_fd)

    lines = capsys.readouterr().out.splitlines()
    assert lines.count(robotiq_grip_probe.HEADER) == 1
    assert sum('MOVING>CONTACT_CLOSING' in line for line in lines) == 2
    assert lines[-1] == 'pen: gOBJ is CONTACT_CLOSING at the end of 2 of 2 closes'


@pytest.mark.parametrize('name', ['pen.json', 'pen.csv', 'pen.txt'])
def test_the_grip_probe_refuses_to_overwrite_any_of_its_files(monkeypatch, tmp_path, name):
    (tmp_path / name).write_text('earlier')
    bus = FakeBus()
    monkeypatch.setattr(robotiq, 'ModbusClient', SimpleNamespace(ModbusSerialClient=lambda **_kwargs: bus))
    monkeypatch.setattr('sys.argv', ['probe', '--port', '/dev/null', '--label', 'pen', '--out', str(tmp_path)])

    with pytest.raises(SystemExit, match=f'{name} exists'):
        robotiq_grip_probe.main()
    assert bus.writes == []
    assert (tmp_path / name).read_text() == 'earlier'
