"""The sim's arm under a synchronous move: it travels through the physics, and answers once it is there."""

import numpy as np
import pytest

import pimm
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm import command as roboarm_command
from positronic.geom import Transform3D
from positronic.simulator.mujoco.sim import MujocoSim
from positronic.tests.testing_coutils import drive_scheduler

MODEL = 'positronic/assets/mujoco/franka_table.xml'


def _turns(sim: MujocoSim, seconds: float) -> int:
    """How many run-loop turns cover ``seconds`` of sim time; a turn is one physics step."""
    return int(seconds / sim.model.opt.timestep) + 10


def _at_rest(scheduler, state, sim: MujocoSim) -> np.ndarray:
    """Pump the world until the arm reports itself, and hand back the joints it rests at."""
    drive_scheduler(scheduler, steps=_turns(sim, 0.1))
    return state.value.q


def _answered(scheduler, answer, turns: int):
    """Pump the world until ``answer`` comes back, and hand it over."""
    for _ in range(turns):
        if answer.done():
            return answer
        next(scheduler)
    raise AssertionError('the move never answered')


def test_a_sync_move_travels_the_arm_to_the_joints_it_asks_for():
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        state = world.pair(sim.state)
        scheduler = world.start([sim])
        target = _at_rest(scheduler, state, sim) + 0.2

        answer = move(roboarm_command.JointPosition(target))

        assert _answered(scheduler, answer, _turns(sim, 5.0)).result() is None
        np.testing.assert_allclose(state.value.q, target, atol=sim._MOVE_TOL)


def test_a_sync_move_that_asks_for_nothing_puts_the_arm_home():
    """Readying a rig is this call with nothing in it, so where it goes is the sim's own home pose."""
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        commands = world.pair(sim.commands)
        state = world.pair(sim.state)
        scheduler = world.start([sim])
        home = _at_rest(scheduler, state, sim)

        commands.emit(roboarm_command.JointPosition(home + 0.3))
        drive_scheduler(scheduler, steps=_turns(sim, 2.0))
        assert np.max(np.abs(state.value.q - home)) > sim._MOVE_TOL, 'the arm never left home'

        _answered(scheduler, move(None), _turns(sim, 5.0)).result()
        np.testing.assert_allclose(state.value.q, home, atol=sim._MOVE_TOL)


def test_a_streamed_command_leaves_the_arm_alone_while_a_move_is_in_flight():
    """A setpoint applied mid-travel fights the move, and its asker is owed the arrival it was promised."""
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        commands = world.pair(sim.commands)
        state = world.pair(sim.state)
        scheduler = world.start([sim])
        home = _at_rest(scheduler, state, sim)

        answer = move(roboarm_command.JointPosition(home + 0.2))
        drive_scheduler(scheduler, steps=2)
        commands.emit(roboarm_command.JointPosition(home - 0.5))

        _answered(scheduler, answer, _turns(sim, 5.0)).result()
        np.testing.assert_allclose(state.value.q, home + 0.2, atol=sim._MOVE_TOL)


def test_a_move_the_arm_never_finishes_hands_its_asker_the_timeout():
    """The joints stop at their limits, so a target beyond them is one the arm can be held short of forever."""
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        state = world.pair(sim.state)
        scheduler = world.start([sim])

        answer = move(roboarm_command.JointPosition(np.full(7, 10.0)))

        with pytest.raises(TimeoutError):
            _answered(scheduler, answer, _turns(sim, sim._MOVE_TIMEOUT_S + 1.0)).result()
        assert state.value.status is RobotStatus.ERROR


def test_a_move_to_a_pose_the_arm_cannot_reach_hands_its_asker_the_reason():
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        scheduler = world.start([sim])

        answer = move(roboarm_command.CartesianPosition(Transform3D([10.0, 10.0, 10.0])))

        with pytest.raises(ValueError, match='out of reach'):
            _answered(scheduler, answer, _turns(sim, 1.0)).result()


def test_frame_zero_waits_for_the_arm_a_trial_readies():
    """The recorder opens on the last prepare's answer and drains as it opens, so frame-0 comes after it."""
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        draw = world.pair(sim.env_reset)
        move = world.pair(sim.sync_move)
        state = world.pair(sim.state)
        meta = world.pair(sim.robot_meta)
        scheduler = world.start([sim])
        home = _at_rest(scheduler, state, sim)
        meta.read()  # the run start emits it too; frame-0 is the next one

        drawn, readied = draw({}), move(roboarm_command.JointPosition(home + 0.3))

        for _ in range(_turns(sim, 5.0)):
            next(scheduler)
            if pimm.value_updated(meta) is not None:
                assert drawn.done() and readied.done(), 'frame-0 went out while the trial was still being readied'
                return
        pytest.fail('frame-0 never went out')
