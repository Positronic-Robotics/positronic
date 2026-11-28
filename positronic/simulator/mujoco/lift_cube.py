"""The ``lift_cube`` task of ``MujocoSim``: one red cube on the table of ``franka_table.xml``, and the ``done`` signal
that ends an episode with success when the cube is ``LIFT_HEIGHT`` above the table top::

    sim = MujocoSim(model_path, lift_cube_loaders(), objects={CUBE_NAME: CUBE.body_name})
    done = LiftCubeDone()
    world.connect(sim.gt_state, done.gt_state)
    wire.wire_embodiment(world, harness, embodiment, privileged={GT_STATE_SIGNAL: sim.gt_state}, done=done.done)
"""

from collections.abc import Iterator

from eval_vocabulary.episode import GT_STATE_OBJECTS, GT_STATE_POSE

import pimm
from positronic.eval import keys as eval_keys
from positronic.simulator.mujoco.transforms import AddBox

CUBE_NAME = 'cube'
CUBE_HALF = 0.02  # m, half the edge of the cube
CUBE = AddBox(
    name=CUBE_NAME,
    size=(CUBE_HALF, CUBE_HALF, CUBE_HALF),
    pos=(0.0, 0.0, CUBE_HALF),
    rgba=(1.0, 0.0, 0.0, 1.0),
    freejoint=True,
)
TABLE_TOP = 0.3  # m, the height of the table top of ``franka_table.xml``
LIFT_HEIGHT = 0.04  # m, how far the bottom of the cube must be above the table top for success
# The key of the pose of the cube in the ``gt_state`` of ``MujocoSim``.
CUBE_POSE = f'.{GT_STATE_OBJECTS}.{CUBE_NAME}.{GT_STATE_POSE}'


class LiftCubeDone(pimm.ControlSystem):
    """Emit ``done`` with success when the cube gets ``LIFT_HEIGHT`` above the table top. It emits once for each lift,
    so a cube that stays up after the episode ends does not end the next episode."""

    def __init__(self):
        self.gt_state = pimm.ControlSystemReceiver[dict](self)
        self.done = pimm.ControlSystemEmitter[dict](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        lifted = False
        while not should_stop.value:
            if (truth := pimm.value_updated(self.gt_state)) is not None:
                up = float(truth[CUBE_POSE][2]) - CUBE_HALF >= TABLE_TOP + LIFT_HEIGHT
                if up and not lifted:
                    self.done.emit({eval_keys.SUCCESS: True})
                lifted = up
            yield pimm.Sleep(0.01)
