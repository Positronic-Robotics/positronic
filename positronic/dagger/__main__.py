"""``python -m positronic.dagger run`` runs DAgger rounds from the config that its arguments give. ``python -m
positronic.dagger mujoco_lift_cube`` runs them from ``positronic.cfg.dagger.mujoco_lift_cube``, and
``robolab_banana_in_bowl`` from ``positronic.cfg.dagger.robolab_banana_in_bowl``."""

import configuronic as cfn

from pimm.logging import init_logging
from positronic.cfg import dagger
from positronic.dagger.rounds import run

if __name__ == '__main__':
    init_logging()
    cfn.cli({
        'run': run,
        'mujoco_lift_cube': dagger.mujoco_lift_cube,
        'robolab_banana_in_bowl': dagger.robolab_banana_in_bowl,
    })
