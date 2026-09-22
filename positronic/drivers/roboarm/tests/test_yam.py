import logging
import types

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock, wire_call
from positronic.drivers.roboarm import RobotStatus, command, yam
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

PARK = np.zeros(6)
RAISED = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])


class FakeYam(yam._FakeYam):
    """A ``_FakeYam`` that can miss what it is asked for, the two ways the yambox station's chain does.

    ``bias`` offsets every joint by a fixed amount, whatever it is asked for. ``gives_back`` is the fraction
    of the way from ``floats_at`` to the command that the chain actually travels, so a correction only
    partly lands and the rest of the gap survives into the next one.
    """

    def __init__(self):
        super().__init__()
        self.bias = np.zeros(6)
        self.gives_back = 1.0
        self.floats_at = np.zeros(6)
        self.stuck = False
        self.targets = []
        self.released_at = []
        self.closed = False

    def command_joint_pos(self, joint_pos):
        self.targets.append(joint_pos.copy())
        if not self.stuck:
            position = joint_pos.copy()
            position[:6] += self.bias
            position[:6] = self.floats_at + self.gives_back * (position[:6] - self.floats_at)
            position[1:3] = np.maximum(position[1:3], 0.0)
            super().command_joint_pos(position)

    def zero_torque_mode(self):
        self.released_at.append(self._pos.copy())
        super().zero_torque_mode()

    def close(self):
        assert self.released_at, 'connection closed before releasing torque'
        self.closed = True


class Rig:
    def __init__(self):
        self.vendor = FakeYam()
        # positronic#772 widened the factory with the station's gravity-compensation factors; this rig
        # names none, so the vendor keeps i2rt's own.
        self.driver = yam.Robot(connect=lambda channel, sim, gravity_comp_factor: self.vendor, park_after_idle_s=1.0)
        self.commands = ManualCommandReceiver()
        self.grip = ManualCommandReceiver()
        self.states = RecordingEmitter()
        self.driver.commands._bind(self.commands)
        self.driver.target_grip._bind(self.grip)
        self.driver.state._bind(self.states)
        self.clock = MockClock()
        self.stop = StopFlag()
        self.loop = self.driver.run(self.stop, self.clock)

    def tick(self, seconds=0.01):
        deadline = self.clock.now() + seconds
        while self.clock.now() < deadline:
            wait = next(self.loop)
            if isinstance(wait, pimm.Sleep):
                self.clock.advance(wait.seconds)

    def finish(self):
        self.stop.stopped = True
        for wait in self.loop:
            if isinstance(wait, pimm.Sleep):
                self.clock.advance(wait.seconds)

    def raise_arm(self):
        while not self.states.emitted or self.states.emitted[-1][1].status == RobotStatus.BUSY:
            self.tick()
        self.commands.push(command.JointPosition(RAISED))
        self.tick(0.5)
        np.testing.assert_allclose(self.vendor._pos[:6], RAISED, atol=0.005)


@pytest.fixture
def rig():
    result = Rig()
    yield result
    # The teardown stow yields from the run body's finally, so the loop is drained, never closed.
    result.finish()


def test_idle_parks_once_and_a_new_command_moves_the_arm(rig, caplog):
    caplog.set_level(logging.INFO, logger=yam.__name__)
    rig.raise_arm()
    rig.tick(0.5)
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=0.005)
    rig.tick(5)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)
    parked = caplog.text.count('Arm parked')
    rig.tick(10)
    assert caplog.text.count('Arm parked') == parked
    assert not rig.vendor.released_at
    rig.raise_arm()


def test_identical_grip_updates_do_not_prevent_parking(rig):
    rig.raise_arm()
    for _ in range(600):
        rig.grip.push(0.0)
        rig.tick()
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)


def test_arm_commands_and_grip_changes_reset_idle_time(rig):
    rig.raise_arm()
    for _ in range(5):
        rig.commands.push(command.JointPosition(RAISED))
        rig.tick(0.5)
    rig.grip.push(0.7)
    rig.tick(0.8)
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=0.005)
    rig.tick(5)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)
    assert rig.vendor._pos[6] == pytest.approx(0.3, abs=0.005)


def test_motion_resets_idle_time(rig):
    rig.raise_arm()
    rig.vendor._vel[:6] = 0.1
    rig.vendor.stuck = True
    rig.tick(5)
    np.testing.assert_allclose(rig.vendor.targets[-1][:6], RAISED)
    rig.vendor._vel[:] = 0
    rig.vendor.stuck = False
    rig.tick(5)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)


def test_disabling_idle_parking_still_parks_on_exit(rig):
    rig.driver._park_after_idle_s = None
    rig.raise_arm()
    rig.tick(60)
    np.testing.assert_allclose(rig.vendor._pos[:6], RAISED, atol=0.005)
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed


def test_startup_and_shutdown_preserve_the_gripper(rig):
    rig.vendor._pos[:6] = RAISED
    rig.vendor._pos[6] = 0.4
    rig.tick(4)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)
    assert rig.vendor._pos[6] == pytest.approx(0.4)
    rig.raise_arm()
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0], np.append(PARK, 0.4), atol=0.005)


def test_parking_corrects_servo_bias_before_releasing_torque(rig):
    rig.raise_arm()
    rig.vendor.bias = np.array([0.0, 0.01, 0.025, 0.025, 0.0, 0.0])
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)
    assert rig.vendor.closed


# What the yambox station measured of its own chain: a correction lands about two thirds of the way, and
# the chain floats 66 mrad above its stops on the three joints that carry the arm's weight.
GIVES_BACK = 0.65
FLOATS_AT = np.array([0.0, 0.066, 0.066, 0.066, 0.0, 0.0])


def test_a_chain_that_gives_back_part_of_a_correction_still_lands_on_its_stops(rig):
    """Torque is cut right after the stow, so the gap the settle leaves is the height the chain falls.

    Corrections that do not add up never close this gap: they swing about three fifths of the first one,
    which on this chain is the centimetre of height the arm was dropping from. The gap is named here rather
    than read from ``_PARK_TOL``, because the gate is half of what is under test: 5 mrad is below anything
    the swinging correction reaches on this chain, whatever gate lets it stop.
    """
    rig.raise_arm()
    rig.vendor.gives_back = GIVES_BACK
    rig.vendor.floats_at = FLOATS_AT
    rig.finish()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)


def test_a_settle_that_never_lands_still_bounds_what_it_asks_for(rig):
    """The corrections add up, so a chain that gives nothing back would march the reference away with no
    bound. ``_PARK_MAX_CORRECTION`` is what keeps the settle off a mechanical stop."""
    rig.raise_arm()
    rig.vendor.stuck = True
    rig.finish()
    asked = np.array([target[:6] for target in rig.vendor.targets])
    assert np.min(asked) >= -yam._Chain._PARK_MAX_CORRECTION


def test_failed_parking_is_bounded_and_reported_then_closes(rig, caplog):
    rig.raise_arm()
    rig.vendor.stuck = True
    started = rig.clock.now()
    rig.finish()
    # Derived, not a constant: each pass spends its own ramp plus the move's settle, then settles again
    # before it measures. The ramp is paced by distance, so the bound follows how far the arm was raised.
    ramp_s = max(yam._Chain._MOVE_TIME_S, float(np.max(np.abs(RAISED - PARK))) / yam._Chain._MAX_JOINT_SPEED)
    bound_s = yam._Chain._PARK_ATTEMPTS * (ramp_s + 2 * yam._Chain._SETTLE_S) + 5
    assert rig.clock.now() - started < bound_s
    assert 'did not reach the park pose' in caplog.text
    assert rig.states.emitted[-1][1].status == RobotStatus.ERROR
    assert rig.vendor.closed


def test_stop_during_a_move_answers_the_caller_and_parks(world, rig):
    rig.tick(2)
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](rig.driver)
    wire_call(world, caller, rig.driver.sync_move)
    answer = caller(command.JointPosition(RAISED))
    rig.tick(0.5)
    rig.finish()
    with pytest.raises(yam.MoveAbandoned):
        answer.result()
    np.testing.assert_allclose(rig.vendor.released_at[0][:6], PARK, atol=0.005)


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
def test_invalid_idle_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match='park_after_idle_s'):
        yam.Robot(park_after_idle_s=timeout)


def test_starting_at_zero_still_settles_under_servo_bias(rig):
    rig.vendor.bias = np.array([0.0, 0.01, 0.025, 0.025, 0.0, 0.0])
    rig.tick(20)
    np.testing.assert_allclose(rig.vendor._pos[:6], PARK, atol=0.005)
    assert rig.states.emitted[-1][1].status == RobotStatus.AVAILABLE
    assert np.min(rig.vendor.targets) >= -0.05


STATION_GRAVITY_COMP = [1.0, 1.1, 1.4, 1.4, 1.0, 1.0]


def _opened_with(**kwargs) -> dict:
    """Start a ``Robot`` built with ``kwargs`` and return what it asked its vendor factory for."""
    seen = {}

    def connect(channel, sim, gravity_comp_factor):
        seen.update(channel=channel, sim=sim, gravity_comp_factor=gravity_comp_factor)
        return yam._FakeYam()

    robot = yam.Robot('can0', connect=connect, **kwargs)
    with pimm.World() as world:
        loop = world.start([robot])
        next(loop)  # the chain is opened before the driver yields for the first time
    return seen


def test_a_station_hands_its_gravity_compensation_to_the_chain():
    """i2rt holds a joint against a gravity model of its own, and a joint that model reads short settles below
    where it is sent. The factors a station measured are no use to it unless the driver passes them on."""
    passed = _opened_with(gravity_comp_factor=STATION_GRAVITY_COMP)['gravity_comp_factor']
    np.testing.assert_array_equal(passed, STATION_GRAVITY_COMP)


def test_a_station_that_measured_none_leaves_the_vendor_its_own():
    """Every YAM shares i2rt's factors until a station measures better ones; naming none has to mean that,
    rather than a vector of ones that would turn the compensation off."""
    assert _opened_with()['gravity_comp_factor'] is None


_DT = 0.01  # seconds per pump; matches the driver's 100 Hz tick so the ramp behaves as it does on the rig

# Event tags the recording fake logs, so the fake and the assertions agree on one spelling.
_CMD, _ZERO_TORQUE, _CLOSE, _MOTOR_OFF = 'cmd', 'zero_torque', 'close', 'motor_off'


class _RunLoopCrash(RuntimeError):
    """Stands for the vendor dropping the chain mid-run, so a test can raise it from inside the loop."""


class _RecordingYam(yam._FakeYam):
    """A ``_FakeYam`` that records the order of vendor calls, so a test can prove the arm reached the safe
    pose before it was cut limp. Setting ``raise_next`` makes the following joint command raise once."""

    def __init__(self, alpha: float = 0.3):
        super().__init__(alpha)
        self.events: list[str] = []
        self.raise_next = False
        self.pos_at_zero_torque: np.ndarray | None = None

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        if self.raise_next:
            self.raise_next = False
            raise _RunLoopCrash('the vendor dropped the chain mid-run')
        super().command_joint_pos(joint_pos)
        self.events.append(_CMD)

    def zero_torque_mode(self) -> None:
        self.pos_at_zero_torque = self._pos[:6].copy()
        self.events.append(_ZERO_TORQUE)
        super().zero_torque_mode()

    def close(self) -> None:
        self.events.append(_CLOSE)
        super().close()


class _StatusSpy(pimm.SignalEmitter):
    """Records the status of each emitted state. The driver re-emits one mutable state object, so each status
    is snapshotted at emit time."""

    def __init__(self):
        self.statuses: list[RobotStatus] = []

    def emit(self, data, ts: int = -1):
        self.statuses.append(data.status)


class _Sink(pimm.SignalEmitter):
    def emit(self, data, ts: int = -1):
        pass


def _driven(fake: _RecordingYam, stop: StopFlag, clock: MockClock):
    """A YAM ``Robot`` over ``fake``, its emitters bound and its loop ready to pump."""
    driver = yam.Robot(connect=lambda channel, sim, gravity_comp_factor: fake)
    status = _StatusSpy()
    driver.state._bind(status)
    driver.grip._bind(_Sink())
    driver.robot_meta._bind(_Sink())
    driver.target_grip._bind(ManualCommandReceiver())
    driver.commands._bind(ManualCommandReceiver())
    return driver, status, driver.run(stop, clock)


def _pump_until(loop, clock: MockClock, cond, *, max_steps: int = 4000) -> None:
    """Pump the loop, advancing the clock ``_DT`` each step, until ``cond`` reads True."""
    for _ in range(max_steps):
        if cond():
            return
        next(loop)
        clock.advance(_DT)
    raise AssertionError('condition not met within the step budget')


def _pump_to_end(loop, clock: MockClock, *, max_steps: int = 4000) -> None:
    """Pump the loop to ``StopIteration``, advancing the clock ``_DT`` each step. A run that ends by raising
    lets that exception propagate."""
    for _ in range(max_steps):
        try:
            next(loop)
        except StopIteration:
            return
        clock.advance(_DT)
    raise AssertionError('the loop did not finish within the step budget')


def _in_run_loop(status: _StatusSpy) -> bool:
    """The startup park has finished and the main loop is running. The park publishes BUSY while it settles
    and one AVAILABLE as it lands; the main loop publishes AVAILABLE every tick, so a run of them marks it."""
    return len(status.statuses) >= 3 and all(s is RobotStatus.AVAILABLE for s in status.statuses[-3:])


def _assert_stowed_then_limp(fake: _RecordingYam) -> None:
    """The arm read the stow pose when torque was cut, no command went out afterwards, and the handle was
    then given back. Only the motor disable follows that, and it has to: it needs the control thread joined."""
    assert fake.pos_at_zero_torque is not None, 'the chain was never cut limp'
    np.testing.assert_allclose(fake.pos_at_zero_torque, yam.YAM_STOW_JOINTS, atol=yam._Chain._PARK_TOL)
    cut = fake.events.index(_ZERO_TORQUE)
    assert _CMD not in fake.events[cut + 1 :], 'a joint command went out after the chain was cut limp'
    given_back = fake.events.index(_CLOSE)
    assert given_back > cut, 'the handle was given back before the chain was cut limp'
    assert set(fake.events[given_back + 1 :]) <= {_MOTOR_OFF}, 'the run did more than disable motors at the end'


def test_a_normal_stop_stows_the_arm_before_it_goes_limp():
    """The brakeless chain is laid onto its joint stops before it loses torque. Without that, a chain going
    limp from the ready pose drops ~0.4 m."""
    fake = _RecordingYam()
    stop, clock = StopFlag(), MockClock()
    _, status, loop = _driven(fake, stop, clock)

    _pump_until(loop, clock, lambda: _in_run_loop(status))  # let the startup park finish and the loop begin

    stop.stopped = True
    _pump_to_end(loop, clock)

    _assert_stowed_then_limp(fake)


class _FakeMotorInterface:
    """Stands for i2rt's single-motor CAN interface, the only thing that can switch a motor off."""

    def __init__(self, events: list[str], **opened_with):
        self.events = events
        self.opened_with = opened_with
        self.refuses: set[int] = set()
        self.off: list[int] = []
        self.closed = False

    def motor_off(self, motor_id: int) -> None:
        if motor_id in self.refuses:
            raise OSError(f'motor {motor_id} did not answer')
        self.off.append(motor_id)
        self.events.append(_MOTOR_OFF)

    def close(self) -> None:
        self.closed = True


class _FakeMotorChain:
    """The CAN chain i2rt hangs off a real YAM: seven motors, the interface that drives them, and a name."""

    def __init__(self):
        self.channel = 'can_follower_l'
        self.motor_list = [(motor_id, 'DM4310') for motor_id in range(1, 8)]
        self.motor_interface = types.SimpleNamespace(control_mode='mit')


class _ChainedYam(_RecordingYam):
    """A ``_RecordingYam`` carrying i2rt's CAN chain, so the driver can reach the motors behind it."""

    def __init__(self):
        super().__init__()
        self.motor_chain = _FakeMotorChain()


def _interfaces_opened(monkeypatch, events: list[str], refuses: tuple[int, ...] = ()) -> list[_FakeMotorInterface]:
    """Stand in for i2rt's CAN interface and collect every one the driver opens. Motors named in ``refuses``
    never answer."""
    opened: list[_FakeMotorInterface] = []

    def open_interface(**kwargs):
        interface = _FakeMotorInterface(events, **kwargs)
        interface.refuses = set(refuses)
        opened.append(interface)
        return interface

    monkeypatch.setattr(yam, 'DMSingleMotorCanInterface', open_interface)
    return opened


def _run_to_the_end(fake: _RecordingYam) -> None:
    """Start the driver over ``fake``, let the startup park finish, then stop it and drain the teardown."""
    stop, clock = StopFlag(), MockClock()
    _, status, loop = _driven(fake, stop, clock)
    _pump_until(loop, clock, lambda: _in_run_loop(status))
    stop.stopped = True
    _pump_to_end(loop, clock)


def test_a_stopped_run_switches_the_motors_off_after_it_gives_the_handle_back(monkeypatch):
    """A chain left limp keeps its motors enabled, so each reaches its own 8 s command timeout and latches an
    error. The disable has to follow ``close()``, which is what stops and joins i2rt's control thread: one
    that beats it makes that thread fail on a motor it is still driving."""
    fake = _ChainedYam()
    opened = _interfaces_opened(monkeypatch, fake.events)

    _run_to_the_end(fake)

    assert len(opened) == 1, 'the teardown opened more than one interface'
    assert opened[0].off == [1, 2, 3, 4, 5, 6, 7]
    assert opened[0].closed, 'the interface that switched the motors off was not closed'
    assert opened[0].opened_with['channel'] == fake.motor_chain.channel
    assert fake.events.index(_CLOSE) < fake.events.index(_MOTOR_OFF), 'a motor was disabled before close()'


def test_a_chain_with_no_motors_behind_it_is_left_alone(monkeypatch):
    """i2rt's own sim and the fakes carry no CAN motors, so the teardown has nothing to open or switch off."""
    fake = _RecordingYam()
    opened = _interfaces_opened(monkeypatch, fake.events)

    _run_to_the_end(fake)

    assert not opened
    _assert_stowed_then_limp(fake)


def test_a_motor_that_will_not_answer_stops_neither_the_others_nor_the_teardown(monkeypatch, caplog):
    """The disable runs in the teardown's own ``finally``, so it may not raise, and one dead motor may not
    leave the other six drawing current."""
    caplog.set_level(logging.WARNING, logger=yam.__name__)
    fake = _ChainedYam()
    opened = _interfaces_opened(monkeypatch, fake.events, refuses=(3,))

    _run_to_the_end(fake)

    assert opened[0].off == [1, 2, 4, 5, 6, 7]
    assert '[3]' in caplog.text
    assert opened[0].closed
    _assert_stowed_then_limp(fake)


def test_a_crash_in_the_run_loop_still_stows_the_arm_before_it_goes_limp():
    """The stow sits in the run body's outermost ``finally``, so an exception thrown out of the loop reaches
    the joint stops as a normal stop does."""
    fake = _RecordingYam()
    stop, clock = StopFlag(), MockClock()
    _, status, loop = _driven(fake, stop, clock)

    _pump_until(loop, clock, lambda: _in_run_loop(status))
    fake.raise_next = True  # the next in-loop joint command raises, standing for a driver crash

    with pytest.raises(_RunLoopCrash):
        _pump_to_end(loop, clock)

    _assert_stowed_then_limp(fake)
