"""What the DH driver writes to the gripper: the calibration only when the gripper has none, and each register only
when its value changes."""

from types import SimpleNamespace

import pytest

from pimm.tests.testing import MockClock
from positronic.drivers.gripper import dh
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

CALIBRATION_READS = 3
TICK_S = 0.005


class FakeBus:
    """A DH gripper on its Modbus holding registers, in memory.

    A write of 0xA5 to the calibration register starts a calibration of ``CALIBRATION_READS`` reads of its state.
    A target width puts the fingers there at once. Each call checks that it goes to device 1.
    """

    def __init__(self, calibrated: bool):
        self.registers = {dh._INIT_STATE: int(calibrated), dh._WIDTH: 1000}
        self.writes: list[tuple[int, int]] = []
        self._calibration_left = 0
        # A bridge that is not up yet refuses the first connects
        self.refused_connects = 0
        self.connects = 0
        # A bridge that drops its client: the read after this many reads fails once, as a lost socket does
        self.drop_after_reads: int | None = None
        self._reads = 0

    def reboot(self) -> None:
        """The gripper loses its power for a moment: it forgets its calibration and the values written to it."""
        self.registers = {dh._INIT_STATE: 0, dh._WIDTH: self.registers[dh._WIDTH]}

    def connect(self) -> bool:
        if self.refused_connects:
            self.refused_connects -= 1
            return False
        self.connects += 1
        return True

    def close(self) -> None:
        pass

    def read_holding_registers(self, address: int, count: int, device_id: int) -> SimpleNamespace:
        assert (count, device_id) == (1, 1)
        self._reads += 1
        if self._reads == self.drop_after_reads:
            raise OSError('read failed: socket disconnected')
        if address == dh._INIT_STATE and self._calibration_left:
            self._calibration_left -= 1
            self.registers[dh._INIT_STATE] = int(not self._calibration_left)
        return SimpleNamespace(registers=[self.registers[address]], isError=lambda: False)

    def write_register(self, address: int, value: int, device_id: int) -> SimpleNamespace:
        assert device_id == 1
        self.writes.append((address, value))
        if address == dh._INIT and value == dh._START_INIT:
            self.registers[dh._INIT_STATE], self._calibration_left = 0, CALIBRATION_READS
        if address == dh._TARGET:
            self.registers[dh._WIDTH] = value
        return SimpleNamespace(isError=lambda: False)


def _driven(monkeypatch, bus: FakeBus):
    """A driver over ``bus``, the receiver its grip requests arrive on, its loop, and its clock."""
    monkeypatch.setattr(dh, 'ModbusClient', SimpleNamespace(ModbusSerialClient=lambda **_kwargs: bus))
    clock = MockClock()
    driver = dh.DHGripper(port='socket://bridge:7663')
    driver.grip._bind(RecordingEmitter(), clock=clock)
    requests = ManualCommandReceiver()
    driver.target_grip._bind(requests)
    return requests, driver.run(StopFlag(), clock), clock


def _pump(loop, clock: MockClock, ticks: int) -> None:
    for _ in range(ticks):
        next(loop)
        clock.advance(TICK_S)


def test_the_driver_waits_for_a_bridge_that_is_not_up_yet(monkeypatch):
    bus = FakeBus(calibrated=True)
    bus.refused_connects = 3
    _, loop, clock = _driven(monkeypatch, bus)

    for _ in range(5):
        next(loop)
        clock.advance(dh._CONNECT_RETRY_S)

    assert dh._TARGET in [address for address, _ in bus.writes]


def test_a_bridge_that_never_comes_up_fails_the_run(monkeypatch):
    bus = FakeBus(calibrated=True)
    bus.refused_connects = 10**6
    _, loop, clock = _driven(monkeypatch, bus)

    with pytest.raises(ConnectionError, match='socket://bridge:7663'):
        for _ in range(10**4):
            next(loop)
            clock.advance(dh._CONNECT_RETRY_S)


def test_a_link_that_breaks_is_connected_again_and_the_grip_goes_on(monkeypatch):
    """A bridge drops its client when another one connects. The driver connects again, and the run goes on, so the
    arms of the session do not stop with it."""
    bus = FakeBus(calibrated=True)
    requests, loop, clock = _driven(monkeypatch, bus)
    _pump(loop, clock, 5)
    bus.drop_after_reads = bus._reads + 1
    requests.push(1.0)

    _pump(loop, clock, 20)

    assert bus.connects == 2
    assert bus.registers[dh._WIDTH] == 0, 'the grip asked for after the break did not reach the fingers'


def _targets(bus: FakeBus) -> list[int]:
    return [value for address, value in bus.writes if address == dh._TARGET]


def test_a_grip_that_changes_at_each_tick_is_written_at_most_once_a_period(monkeypatch):
    """Each new target starts the fingers from rest, so a stream of them keeps the fingers still."""
    bus = FakeBus(calibrated=True)
    requests, loop, clock = _driven(monkeypatch, bus)
    _pump(loop, clock, 5)
    ticks = round(1.0 / TICK_S)
    for tick in range(ticks):  # from 0.1 to 0.9 in one second, a new value at each tick
        requests.push(0.1 + 0.8 * tick / ticks)
        _pump(loop, clock, 1)

    written = _targets(bus)[1:]
    assert 3 <= len(written) <= round(1.0 / dh.TARGET_PERIOD_S) + 1


def test_a_grip_that_moves_less_than_a_step_is_not_written(monkeypatch):
    bus = FakeBus(calibrated=True)
    requests, loop, clock = _driven(monkeypatch, bus)
    requests.push(0.5)
    _pump(loop, clock, 100)
    requests.push(0.5 + 0.5 * dh.TARGET_STEP / 1000)

    _pump(loop, clock, 100)

    assert _targets(bus)[-1] == 500


def test_open_and_closed_are_written_at_once(monkeypatch):
    bus = FakeBus(calibrated=True)
    requests, loop, clock = _driven(monkeypatch, bus)
    requests.push(0.5)
    _pump(loop, clock, 2)
    requests.push(1.0)

    _pump(loop, clock, 1)

    assert _targets(bus)[-1] == 0


def test_a_calibrated_gripper_is_not_calibrated_again(monkeypatch):
    """A calibration moves the fingers through their whole stroke, so a driver that restarts must not ask for one."""
    bus = FakeBus(calibrated=True)
    _, loop, clock = _driven(monkeypatch, bus)

    _pump(loop, clock, 5)

    assert (dh._INIT, dh._START_INIT) not in bus.writes


def test_a_gripper_with_no_calibration_is_calibrated_before_it_is_commanded(monkeypatch):
    bus = FakeBus(calibrated=False)
    _, loop, clock = _driven(monkeypatch, bus)

    _pump(loop, clock, CALIBRATION_READS + 3)

    assert bus.writes[0] == (dh._INIT, dh._START_INIT)
    assert dh._TARGET in [address for address, _ in bus.writes[1:]]


def test_a_grip_that_does_not_change_is_written_once(monkeypatch):
    """Each write is a round trip on the bus: one sent at each tick would slow the loop down to the bus."""
    bus = FakeBus(calibrated=True)
    requests, loop, clock = _driven(monkeypatch, bus)
    _pump(loop, clock, 5)
    requests.push(1.0)

    _pump(loop, clock, 20)

    assert [value for address, value in bus.writes if address == dh._TARGET] == [1000, 0]
    assert len([address for address, _ in bus.writes if address in (dh._FORCE, dh._SPEED)]) == 2
    assert bus.registers[dh._WIDTH] == 0


def test_a_gripper_that_loses_its_calibration_while_the_link_stays_up_is_calibrated_and_commanded_again(monkeypatch):
    """A gripper on the wrist of a CRON arm lost its calibration and its force as the arm powered up."""
    bus = FakeBus(calibrated=True)
    requests, loop, clock = _driven(monkeypatch, bus)
    requests.push(0.5)
    _pump(loop, clock, 20)
    del bus.writes[:]

    bus.reboot()
    _pump(loop, clock, round(dh._CALIBRATION_CHECK_S / TICK_S) + CALIBRATION_READS + 5)

    assert bus.writes[0] == (dh._INIT, dh._START_INIT)
    assert {dh._FORCE, dh._SPEED, dh._TARGET} <= {address for address, _ in bus.writes[1:]}
    assert bus.registers[dh._WIDTH] == 500
