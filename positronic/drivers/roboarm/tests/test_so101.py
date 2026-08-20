"""What the SO-101 driver puts on its bus, and what a caller waiting on a move hears."""

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock, wire_call
from positronic import geom
from positronic.drivers.motors.feetech import MotorBus
from positronic.drivers.roboarm import RobotStatus, command
from positronic.drivers.roboarm.so101 import driver as so101
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.drivers.utils import MoveAbandoned
from positronic.tests.testing_coutils import ManualCommandReceiver, RecordingEmitter

# Mid-range on every joint: a posture the arm can be commanded away from in either direction. Five arm
# joints and the gripper, in the bus's normalized units — which ``_Kinematics`` below makes equal radians.
MIDDLE = np.full(6, 0.5)
JOGGED = np.full(5, 0.7)


class _Kinematics:
    """Kinematics with the vendor solver taken out: the end effector sits at the first three joints.

    Every joint spans 0..1 radians, so the bus's normalized units and radians are the same number.
    """

    joint_limits = np.tile(np.array([0.0, 1.0]), (6, 1))

    def __init__(self, urdf_path: str, target_frame_name: str):
        pass

    def forward(self, joint_positions: np.ndarray) -> geom.Transform3D:
        return geom.Transform3D(translation=np.asarray(joint_positions[:3], dtype=np.float64))

    def inverse(self, current_joint_pos, target_ee_pose: geom.Transform3D, n_iter: int = 10) -> np.ndarray:
        reach = np.asarray(target_ee_pose.translation, dtype=np.float64)
        if np.any(reach < 0.0) or np.any(reach > 1.0):
            raise ValueError(f'{target_ee_pose} is out of reach')
        return np.concatenate([reach, np.asarray(current_joint_pos, dtype=np.float64)[3:]])


@pytest.fixture(autouse=True)
def _kinematics(monkeypatch):
    monkeypatch.setattr(so101, 'Kinematics', _Kinematics)


class FakeBus(MotorBus):
    """In-memory ``MotorBus``: the servos latch the goal they are given and are at it by the next read.

    Position is in the bus's normalized units, five arm joints and the gripper. ``blocked`` holds the
    servos where they stand; ``raises``, once set, is what reading the bus raises.
    """

    def __init__(self, position=MIDDLE):
        # No ``super().__init__``: the base opens a serial port, and there is none
        self._position = np.asarray(position, dtype=np.float64)
        self.targets: list[np.ndarray] = []
        self.blocked = False
        self.raises: Exception | None = None
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    @property
    def position(self) -> np.ndarray:
        if self.raises is not None:
            raise self.raises
        if not self.blocked and self.targets:
            self._position = self.targets[-1].copy()
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return np.zeros_like(self._position)

    def set_target_position(self, positions: np.ndarray) -> None:
        if self.raises is not None:
            raise self.raises
        self.targets.append(np.asarray(positions, dtype=np.float64))


class WatchingEmitter(pimm.SignalEmitter):
    """Records each published status alongside whether ``answer`` had already come back when it went out."""

    def __init__(self, answer):
        self.seen: list[tuple[RobotStatus, bool]] = []
        self._answer = answer

    def emit(self, data, ts: int = -1) -> None:
        self.seen.append((data.status, self._answer.done()))


def _mover(world: pimm.World, driver: so101.Robot) -> pimm.calls.Caller[command.CommandType, None]:
    """A caller on ``driver.sync_move``, for a test that pumps its generator rather than running a World."""
    caller = pimm.calls.ControlSystemCaller[command.CommandType, None](driver)
    wire_call(world, caller, driver.sync_move)
    return caller


def _driven(bus: FakeBus, clock: MockClock | None = None, stop: StopFlag | None = None):
    """A driver over ``bus`` with its state recorded, and its loop ready to pump."""
    driver = so101.Robot(bus)
    states = RecordingEmitter()
    driver.state._bind(states)
    return driver, states, driver.run(stop or StopFlag(), clock or MockClock())


def _pump(loop, answer, clock: MockClock | None = None, steps: int = 10) -> None:
    """Run the loop until ``answer`` comes back, moving ``clock`` on if one was given."""
    for _ in range(steps):
        if answer.done():
            return
        if clock is not None:
            clock.advance(1.0)
        next(loop)


def test_the_bus_is_written_only_once_something_has_asked_for_a_setpoint():
    """Until something asks, the arm holds whatever the bus was left holding."""
    bus = FakeBus()
    driver, _, loop = _driven(bus)
    grip = ManualCommandReceiver()
    driver.target_grip._bind(grip)

    for _ in range(3):  # nothing has asked for anything
        next(loop)
    assert bus.targets == []

    grip.push(0.25)
    next(loop)
    assert len(bus.targets) == 1

    next(loop)  # and nothing has asked since
    assert len(bus.targets) == 1


def test_closing_the_fingers_reaches_the_bus_from_an_arm_that_starts_open():
    """Zero is a width like any other, and a caller asking for it is asking for a move."""
    bus = FakeBus(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.8]))
    driver, _, loop = _driven(bus)
    grip = ManualCommandReceiver()
    driver.target_grip._bind(grip)

    grip.push(0.0)
    next(loop)

    assert bus.targets[-1][-1] == pytest.approx(0.0)


def test_the_gripper_and_the_arm_reach_the_bus_as_one_setpoint():
    """They share one bus setpoint, but arrive as two channels."""
    bus = FakeBus()
    driver, _, loop = _driven(bus)
    grip = ManualCommandReceiver()
    driver.target_grip._bind(grip)

    grip.push(0.8)
    next(loop)

    assert bus.targets[-1][-1] == pytest.approx(0.8)
    np.testing.assert_allclose(bus.targets[-1][:-1], MIDDLE[:-1])


def test_a_streamed_joint_command_reaches_the_bus():
    bus = FakeBus()
    driver, _, loop = _driven(bus)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)

    commands.push(command.JointPosition(JOGGED))
    next(loop)

    np.testing.assert_allclose(bus.targets[-1][:-1], JOGGED)


def test_a_streamed_arm_command_leaves_the_fingers_where_the_bus_found_them():
    """The whole setpoint goes out on every write, so the half nobody commanded carries what is held."""
    bus = FakeBus(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.8]))
    driver, _, loop = _driven(bus)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)

    commands.push(command.JointPosition(JOGGED))
    next(loop)

    assert bus.targets[-1][-1] == pytest.approx(0.8)


def test_a_streamed_command_the_arm_cannot_be_put_at_leaves_it_where_it_is():
    """A command stream cannot end the run: the next command supersedes one that could not be applied."""
    bus = FakeBus()
    driver, _, loop = _driven(bus)
    commands = ManualCommandReceiver()
    driver.commands._bind(commands)

    commands.push(command.CartesianPosition(geom.Transform3D(translation=np.full(3, 100.0))))
    next(loop)
    assert bus.targets == [], 'a pose the arm cannot reach is not a setpoint'

    commands.push(command.JointPosition(JOGGED))
    next(loop)
    np.testing.assert_allclose(bus.targets[-1][:-1], JOGGED)


def test_a_sync_move_answers_once_the_bus_reads_back_at_its_target(world):
    bus = FakeBus()
    driver, _, loop = _driven(bus)
    answer = _mover(world, driver)(command.JointPosition(JOGGED))

    _pump(loop, answer)

    assert answer.result() is None
    np.testing.assert_allclose(bus.targets[-1][:-1], JOGGED)


def test_an_arm_serving_a_move_reads_busy_and_takes_commands_once_it_lands(world):
    """A move owns the arm: while it runs the driver reads no command, and the wire says so."""
    bus = FakeBus()
    bus.blocked = True  # the arm does not follow its setpoint, so the move cannot land yet
    driver, states, loop = _driven(bus)
    answer = _mover(world, driver)(command.JointPosition(JOGGED))

    next(loop)
    assert states.emitted[-1][1].status is RobotStatus.BUSY
    assert not answer.done()

    bus.blocked = False
    _pump(loop, answer)

    answer.result()
    assert states.emitted[-1][1].status is RobotStatus.AVAILABLE


def test_the_state_that_answers_a_move_is_published_before_the_answer(world):
    """A caller that learns its move landed reads the arm next, and must not find the mid-travel sample."""
    bus = FakeBus()
    driver = so101.Robot(bus)
    answer = _mover(world, driver)(command.JointPosition(JOGGED))
    watch = WatchingEmitter(answer)
    driver.state._bind(watch)
    loop = driver.run(StopFlag(), MockClock())

    _pump(loop, answer)

    answer.result()
    assert (RobotStatus.AVAILABLE, False) in watch.seen, 'the arrival went out after its asker was told'


def test_the_state_answering_a_move_carries_the_pose_the_arm_reached(world):
    bus = FakeBus()
    driver, states, loop = _driven(bus)
    answer = _mover(world, driver)(command.JointPosition(JOGGED))

    _pump(loop, answer)

    answer.result()
    arrived = states.emitted[-1][1]
    assert arrived.status is RobotStatus.AVAILABLE
    np.testing.assert_allclose(arrived.q, JOGGED, atol=1e-6)


def test_a_sync_move_that_never_arrives_times_out_and_holds_where_the_arm_stopped(world):
    """An arm left on the target it missed would resume the move once whatever blocked it goes away."""
    bus = FakeBus()
    bus.blocked = True
    clock = MockClock()
    driver, states, loop = _driven(bus, clock)
    answer = _mover(world, driver)(command.JointPosition(JOGGED))

    _pump(loop, answer, clock)

    with pytest.raises(TimeoutError, match='stopped at'):
        answer.result()
    np.testing.assert_allclose(bus.targets[-1][:-1], MIDDLE[:-1])
    assert states.emitted[-1][1].status is RobotStatus.ERROR


def test_an_arm_that_stopped_short_reads_error_until_a_move_lands(world):
    """ERROR stands until a move genuinely lands, so a caller cannot read the arm as available in between."""
    bus = FakeBus()
    bus.blocked = True
    clock = MockClock()
    driver, states, loop = _driven(bus, clock)
    move = _mover(world, driver)

    failed = move(command.JointPosition(JOGGED))
    _pump(loop, failed, clock)
    with pytest.raises(TimeoutError):
        failed.result()

    bus.blocked = False
    landed = move(command.JointPosition(np.full(5, 0.6)))
    _pump(loop, landed, clock)

    landed.result()
    assert states.emitted[-1][1].status is RobotStatus.AVAILABLE


def test_a_move_the_world_stops_under_is_handed_back_to_its_asker(world):
    """A stop ends the loop with no arrival to report, and silence would hold the asker for good."""
    bus = FakeBus()
    bus.blocked = True
    stop = StopFlag()
    driver, _, loop = _driven(bus, stop=stop)
    answer = _mover(world, driver)(command.JointPosition(JOGGED))

    next(loop)
    assert not answer.done()

    stop.stopped = True
    with pytest.raises(StopIteration):
        next(loop)

    with pytest.raises(MoveAbandoned):
        answer.result()


def test_a_move_the_world_stops_under_leaves_the_bus_holding_where_the_arm_is(world):
    """The servos chase the goal they were last given, so a move cut short must not leave the far end on
    the bus for them to keep driving at once the world is gone."""
    bus = FakeBus()
    bus.blocked = True
    stop = StopFlag()
    driver, _, loop = _driven(bus, stop=stop)
    answer = _mover(world, driver)(command.JointPosition(JOGGED))

    next(loop)
    np.testing.assert_allclose(bus.targets[-1][:-1], JOGGED)  # the move is on the bus

    stop.stopped = True
    with pytest.raises(StopIteration):
        next(loop)

    np.testing.assert_allclose(bus.targets[-1][:-1], MIDDLE[:-1])
    assert answer.done()


def test_a_bus_that_dies_as_the_run_ends_still_answers_the_move(world):
    """The hold is the last thing the arm is asked for; a bus that refuses it must not swallow the answer."""
    bus = FakeBus()
    bus.blocked = True
    stop = StopFlag()
    driver, _, loop = _driven(bus, stop=stop)
    answer = _mover(world, driver)(command.JointPosition(JOGGED))

    next(loop)
    stop.stopped = True
    bus.raises = RuntimeError('the bus went away')
    with pytest.raises(RuntimeError, match='the bus went away'):
        next(loop)

    with pytest.raises(MoveAbandoned):
        answer.result()


def test_a_run_that_ends_gives_the_bus_back(world):
    """The port is held for the run, so an arm that never builds must not keep it."""
    bus = FakeBus()
    stop = StopFlag()
    _, _, loop = _driven(bus, stop=stop)

    next(loop)
    assert bus.connected

    stop.stopped = True
    with pytest.raises(StopIteration):
        next(loop)

    assert not bus.connected


def test_a_run_that_dies_mid_move_hands_what_killed_it_to_the_asker(world):
    """The asker is blocked on an answer, and a driver that stops looping will never produce one."""
    bus = FakeBus()
    bus.blocked = True
    driver, _, loop = _driven(bus)
    answer = _mover(world, driver)(command.JointPosition(JOGGED))

    next(loop)
    assert not answer.done()

    bus.raises = RuntimeError('the bus went away')
    with pytest.raises(RuntimeError, match='the bus went away'):
        next(loop)

    with pytest.raises(RuntimeError, match='the bus went away'):
        answer.result()
