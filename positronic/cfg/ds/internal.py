"""
Internal PhAIL datasets with model-specific observation/action encodings.

These datasets remain on the private s3://raw/ bucket and include:
- Raw datasets from internal data collection
- Transformed variants (cubes_sim, pnp_sim with signal transforms)
- OpenPI-encoded variants (*_openpi_ft)
- GR00T-encoded variants (*_groot_ft)
- Full combined datasets for multi-task training
"""

import configuronic as cfn
import pos3

from positronic.cfg.policy import action, observation
from positronic.dataset.local_dataset import load_all_datasets
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.transforms.episode import Concat, Derive, FromValue, Group, Identity, Rename

from . import concat_ds, local, transform

# Task constants
TOWELS_TASK = 'Pick all the towels one by one from transparent tote and place them into the large grey tote.'
SPOONS_TASK = 'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.'
SCISSORS_TASK = 'Pick all the scissors one by one from transparent tote and place them into the large grey tote.'


@cfn.config(path='s3://raw/droid/', recovery_all=True, recovery_towels=True)
def droid_ds(path, recovery_all, recovery_towels):
    """Internal DROID dataset with task label transforms."""
    root = pos3.download(path)

    datasets = [
        TransformedDataset(load_all_datasets(root / 'towels'), Group(Derive(task=FromValue(TOWELS_TASK)), Identity())),
        TransformedDataset(load_all_datasets(root / 'spoons'), Group(Derive(task=FromValue(SPOONS_TASK)), Identity())),
        TransformedDataset(
            # TODO: This is typo, it was done when the we started collecting the dataset, and now we are not fixing it.
            load_all_datasets(root / 'scisors'),
            Group(Derive(task=FromValue(SCISSORS_TASK)), Identity()),
        ),
    ]
    if recovery_towels:
        datasets.append(
            TransformedDataset(
                load_all_datasets(root / 'recovery_towels/'), Group(Derive(task=FromValue(TOWELS_TASK)), Identity())
            )
        )
    if recovery_all:
        for task in [TOWELS_TASK, SPOONS_TASK, SCISSORS_TASK]:
            datasets.append(
                TransformedDataset(
                    load_all_datasets(root / 'recovery/'), Group(Derive(task=FromValue(task)), Identity())
                )
            )
    return concat_ds.override(datasets=datasets)


# Signal transformations for sim datasets
old_to_new = Group(
    Derive(**{
        'controller_positions.right': Concat('right_controller_translation', 'right_controller_quaternion'),
        'robot_commands.pose': Concat('target_robot_position_translation', 'target_robot_position_quaternion'),
        'robot_state.ee_pose': Concat('robot_position_translation', 'robot_position_quaternion'),
        'task': FromValue('Pick up the green cube and place it on the red cube.'),
    }),
    Rename(**{
        'robot_state.q': 'robot_joints',
        'robot_state.dq': 'robot_joints_velocity',
        'image.wrist': 'image.handcam_left',
        'image.exterior': 'image.back_view',
    }),
    Identity(select=['grip', 'target_grip', 'mjSTATE_FULLPHYSICS', 'mjSTATE_INTEGRATION', 'mjSTATE_WARMSTART']),
)

cubes_sim_raw = local.override(path='s3://raw/sim-cubes/luzan/')
cubes_sim = transform.override(base=cubes_sim_raw, transforms=[old_to_new])

pnp_sim_raw = local.override(path='s3://raw/sim_pnp/')
pnp_sim = transform.override(
    base=pnp_sim_raw,
    transforms=[
        Group(
            Derive(task=FromValue('Pick up objects from the red tote and place them in the green tote.')),
            Rename(**{'image.exterior': 'image.back_view'}),
            Identity(),
        )
    ],
)


# Encoded helper (inline to avoid circular deps)
def _encoded(base, observation_cfg, action_cfg, task=None):
    tfs = [observation_cfg, action_cfg]
    if task is not None:
        tfs.append(Derive(task=lambda _ep: task))
    else:
        tfs.append(Identity(select=['task']))
    return TransformedDataset(base, Group(*tfs))


# OpenPI encoded variants (based on INTERNAL datasets)
@cfn.config()
def droid_openpi_ft():
    return _encoded(droid_ds, observation.eepose, action.absolute_position)


@cfn.config()
def sim_stack_openpi_ft():
    return _encoded(cubes_sim, observation.eepose, action.absolute_position)


@cfn.config()
def sim_pnp_openpi_ft():
    return _encoded(pnp_sim, observation.eepose, action.absolute_position)


@cfn.config()
def full_openpi_ft():
    return _encoded(
        concat_ds.override(datasets=[droid_ds, cubes_sim, pnp_sim]), observation.eepose, action.absolute_position
    )


# GR00T encoded variants (based on INTERNAL datasets)
@cfn.config()
def droid_groot_ft():
    return _encoded(droid_ds, observation.groot_ee_absolute, action.groot)


@cfn.config()
def sim_stack_groot_ft():
    return _encoded(cubes_sim, observation.groot_ee_absolute, action.groot)


@cfn.config()
def sim_pnp_groot_ft():
    return _encoded(pnp_sim, observation.groot_ee_absolute, action.groot)


@cfn.config()
def full_groot_ft():
    return _encoded(
        concat_ds.override(datasets=[droid_ds, cubes_sim, pnp_sim]), observation.groot_ee_absolute, action.groot
    )
