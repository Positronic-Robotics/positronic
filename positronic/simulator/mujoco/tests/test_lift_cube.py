import pimm
from positronic.eval import keys as eval_keys
from positronic.simulator.mujoco.lift_cube import CUBE_HALF, CUBE_POSE, LIFT_HEIGHT, TABLE_TOP, LiftCubeDone
from positronic.tests.testing_coutils import drive_scheduler

UP = TABLE_TOP + CUBE_HALF + LIFT_HEIGHT


def test_the_done_of_lift_cube_comes_once_for_each_lift_of_the_cube_past_the_lift_height():
    done = LiftCubeDone()
    with pimm.World(virtual_time=True) as world:
        truth = world.pair(done.gt_state)
        ended = world.pair(done.done)
        scheduler = world.start([done])
        emitted = []
        for z in (TABLE_TOP + CUBE_HALF, UP + 0.001, UP + 0.01, TABLE_TOP + CUBE_HALF, UP + 0.001):
            truth.emit({CUBE_POSE: [0.5, 0.0, z, 1.0, 0.0, 0.0, 0.0]})
            drive_scheduler(scheduler, steps=2)
            emitted.append(pimm.value_updated(ended))
    assert emitted == [None, {eval_keys.SUCCESS: True}, None, None, {eval_keys.SUCCESS: True}]
