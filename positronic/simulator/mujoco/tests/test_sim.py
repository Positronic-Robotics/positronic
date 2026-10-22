"""The sim's arm under a synchronous move: it travels through the physics, and answers once it is there."""

import re

import mujoco as mj
import numpy as np
import pytest

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.ik import _prepare_spec
from positronic.drivers.roboarm.models import DEFAULT_FRAME, FLANGE_LINK, bundled_panda_model
from positronic.geom import Transform3D
from positronic.simulator.mujoco.sim import SCENE_XML, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
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


def test_a_sync_move_takes_the_arm_over_from_the_command_stream():
    """Between moves the stream owns the arm, so a move has to retarget what the stream last set."""
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        commands = world.pair(sim.commands)
        state = world.pair(sim.state)
        scheduler = world.start([sim])
        home = _at_rest(scheduler, state, sim)

        commands.emit(roboarm_command.JointPosition(home + 0.3))
        drive_scheduler(scheduler, steps=_turns(sim, 2.0))
        assert np.max(np.abs(state.value.q - home)) > sim._MOVE_TOL, 'the streamed command never moved the arm'

        _answered(scheduler, move(roboarm_command.JointPosition(home)), _turns(sim, 5.0)).result()
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


def test_a_move_that_ends_publishes_the_arm_before_it_answers():
    """The state stream can run slower than the physics, and its asker still reads where the arm got to."""
    sim = MujocoSim(MODEL, loaders=(), state_fps=1.0)

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        state = world.pair(sim.state)
        scheduler = world.start([sim])
        target = _at_rest(scheduler, state, sim) + 0.2

        answer = move(roboarm_command.JointPosition(target))

        _answered(scheduler, answer, _turns(sim, 5.0)).result()
        np.testing.assert_allclose(state.value.q, target, atol=sim._MOVE_TOL)


def test_a_move_that_is_already_there_publishes_before_it_answers():
    """The immediate answer owes its asker a state as much as one that travelled does."""
    sim = MujocoSim(MODEL, loaders=(), state_fps=1.0)

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        state = world.pair(sim.state)
        scheduler = world.start([sim])
        home = _at_rest(scheduler, state, sim)
        pimm.value_updated(state)  # drained, so what is read below is what the move published

        answer = move(roboarm_command.JointPosition(home))

        _answered(scheduler, answer, _turns(sim, 0.05)).result()
        assert pimm.value_updated(state) is not None


def test_a_move_naming_the_wrong_joints_leaves_the_arm_alone():
    """The vector is paired with the actuators before any is written, so a refused move retargets none."""
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        state = world.pair(sim.state)
        scheduler = world.start([sim])
        home = _at_rest(scheduler, state, sim)

        answer = move(roboarm_command.JointPosition(np.full(6, 1.0)))

        with pytest.raises(ValueError):
            _answered(scheduler, answer, _turns(sim, 1.0)).result()
        drive_scheduler(scheduler, steps=_turns(sim, 1.0))
        np.testing.assert_allclose(state.value.q, home, atol=sim._MOVE_TOL)


def test_a_move_to_joints_that_are_no_target_is_refused_rather_than_applied():
    """NaN controls step the sim to NaN, and no comparison against NaN ever reports the arm arrived."""
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        state = world.pair(sim.state)
        scheduler = world.start([sim])
        home = _at_rest(scheduler, state, sim)

        answer = move(roboarm_command.JointPosition(np.full(7, float('nan'))))

        with pytest.raises(ValueError, match='not a joint target'):
            _answered(scheduler, answer, _turns(sim, 1.0)).result()
        drive_scheduler(scheduler, steps=_turns(sim, 0.5))
        np.testing.assert_allclose(state.value.q, home, atol=sim._MOVE_TOL)


def test_a_move_by_a_delta_that_names_fewer_joints_is_refused():
    """NumPy broadcasts a one-element delta across every joint, so arity alone would accept it."""
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        state = world.pair(sim.state)
        scheduler = world.start([sim])
        home = _at_rest(scheduler, state, sim)

        answer = move(roboarm_command.JointDelta(np.full(1, 0.3)))

        with pytest.raises(ValueError, match='does not name every joint'):
            _answered(scheduler, answer, _turns(sim, 1.0)).result()
        drive_scheduler(scheduler, steps=_turns(sim, 0.5))
        np.testing.assert_allclose(state.value.q, home, atol=sim._MOVE_TOL)


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


def test_the_arm_reads_busy_while_a_move_owns_it():
    """BUSY is what an unread command stream looks like, and a move in flight is why it goes unread."""
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        state = world.pair(sim.state)
        scheduler = world.start([sim])
        target = _at_rest(scheduler, state, sim) + 0.3

        answer = move(roboarm_command.JointPosition(target))
        drive_scheduler(scheduler, steps=5)

        assert state.value.status is RobotStatus.BUSY
        _answered(scheduler, answer, _turns(sim, 5.0)).result()
        assert state.value.status is RobotStatus.AVAILABLE


def test_a_move_that_gives_up_leaves_the_arm_where_it_stopped():
    """The actuators keep driving at the last target they were given, so a move that fails hands them back."""
    sim = MujocoSim(MODEL, loaders=())
    sim._MOVE_TIMEOUT_S = 0.05  # the arm is still travelling when the move gives up

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        state = world.pair(sim.state)
        scheduler = world.start([sim])
        target = _at_rest(scheduler, state, sim) + 0.5

        answer = move(roboarm_command.JointPosition(target))

        with pytest.raises(TimeoutError):
            _answered(scheduler, answer, _turns(sim, 1.0)).result()
        stopped = state.value.q
        drive_scheduler(scheduler, steps=_turns(sim, 1.0))
        np.testing.assert_allclose(state.value.q, stopped, atol=sim._MOVE_TOL)


def test_a_move_to_a_pose_the_arm_cannot_reach_hands_its_asker_the_reason():
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        move = world.pair(sim.sync_move)
        scheduler = world.start([sim])

        answer = move(roboarm_command.CartesianPosition(Transform3D([10.0, 10.0, 10.0])))

        with pytest.raises(ValueError, match='out of reach'):
            _answered(scheduler, answer, _turns(sim, 1.0)).result()


def test_a_redraw_publishes_the_scene_before_it_answers():
    """A trial opening on the redraw's answer finds the scene it drew already on the channels."""
    sim = MujocoSim(MODEL, loaders=())

    with pimm.World(virtual_time=True) as world:
        draw = world.pair(sim.env_reset)
        state = world.pair(sim.state)
        meta = world.pair(sim.robot_meta)
        scheduler = world.start([sim])
        _at_rest(scheduler, state, sim)
        meta.read()  # the run start emits it too; the redraw's is the next one

        drawn = draw({})

        for _ in range(_turns(sim, 5.0)):
            next(scheduler)
            if drawn.done():
                assert pimm.value_updated(meta) is not None, 'the redraw answered before it published the scene'
                return
        pytest.fail('the redraw never answered')


JOINT_SETS = [
    np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0]),
    np.array([0.5, 0.3, -0.4, -1.2, 0.8, 1.0, -0.3]),
    np.array([-0.8, -1.0, 0.6, -2.5, -0.3, 2.5, 0.9]),
]


class _MoveTheArmBase(MujocoSceneTransform):
    def apply(self, spec):
        base = spec.body('world_ph')
        base.pos = np.array([0.3, -0.2, 0.1])
        base.quat = np.array([np.cos(0.35), 0.0, 0.0, np.sin(0.35)])
        return spec


def _model_pose(urdf: str, q: np.ndarray, frame: str) -> geom.Transform3D:
    """The pose of ``frame`` at joints ``q`` in a URDF, in the arm base."""
    model = _prepare_spec(urdf, frame).compile()
    data = mj.MjData(model)
    data.qpos[[model.joint(f'joint{i}').qposadr.item() for i in range(1, 8)]] = q
    mj.mj_forward(model, data)
    site = data.site(frame)
    return geom.Transform3D(site.xpos.copy(), geom.Rotation.from_rotation_matrix(site.xmat.reshape(3, 3)))


def _assert_reports_default_of_the_recorded_model(sim: MujocoSim) -> None:
    urdf = bundled_panda_model()[roboarm_keys.URDF]
    for q in JOINT_SETS:
        sim.data.qpos[[sim.model.joint(f'joint{i}_ph').qposadr.item() for i in range(1, 8)]] = q
        mj.mj_forward(sim.model, sim.data)
        base = sim.data.body('link0_ph')
        base_pose = geom.Transform3D(base.xpos.copy(), geom.Rotation.from_rotation_matrix(base.xmat.reshape(3, 3)))
        got = base_pose.inv * sim._ee_pose
        want = _model_pose(urdf, q, DEFAULT_FRAME)
        np.testing.assert_allclose(got.translation, want.translation, atol=1e-6)
        assert (want.rotation.inv * got.rotation).angle < 1e-6


@pytest.mark.parametrize('loaders', [(), (_MoveTheArmBase(),)], ids=['scene-base', 'moved-base'])
def test_the_reported_pose_is_default_of_the_recorded_model(loaders):
    _assert_reports_default_of_the_recorded_model(MujocoSim(MODEL, loaders=loaders))


def test_a_recorded_scene_with_no_flange_site_reports_at_the_flange():
    sim = MujocoSim(MODEL, loaders=())
    unnamed = re.sub(f'<site name="{FLANGE_LINK}_ph"[^>]*/>', '', sim.scene_xml)
    assert unnamed != sim.scene_xml, 'the scene carries no flange site to delete'
    sim.load_state({**sim.save_state(), SCENE_XML: unnamed})
    _assert_reports_default_of_the_recorded_model(sim)
