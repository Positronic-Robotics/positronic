"""Positronic datasets for tasks ran in simulation."""

from . import local, transform
from .internal import SIM_ROBOT_TRANSFORM

# Simulated cube stacking dataset
# Migrated from: @positronic.cfg.ds.internal.sim_stack
# Size: 499MB, 317 episodes with transforms baked in (ee_pose, robot_joints, task)
sim_stack_cubes = transform.override(
    base=local.override(path='s3://PUBLIC@positronic-public/datasets/sim-stack-cubes/'),
    transforms=[SIM_ROBOT_TRANSFORM],
)

# Simulated pick-and-place dataset
# Migrated from: @positronic.cfg.ds.internal.sim_pnp
# Size: 1.3GB, 214 episodes with transforms baked in
sim_pick_place = transform.override(
    base=local.override(path='s3://PUBLIC@positronic-public/datasets/sim-pick-place/'), transforms=[SIM_ROBOT_TRANSFORM]
)

# Ten demonstrations by a person on the MolmoSpaces pick mini benchmark (suite v2), recorded with
# `pick_mini_collect` on episodes that no competition eval scores. Every episode is a success.
# Size: 30MB, 10 episodes, three cameras
molmo_pick_demos = local.override(path='s3://PUBLIC@positronic-public/datasets/molmo-pick-demos/v1/')
