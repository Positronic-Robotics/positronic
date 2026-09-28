"""Utility for exporting a transformed dataset to disk.

The CLI defined here reads a configured source dataset—most often a
``TransformedDataset`` that reshapes signals lazily—and streams it into a new
``LocalDataset`` directory. The default configuration targets
``update_v0_1_0`` transformation around specified local dataset,
but you are welcome to use your own transformations with ``original_ds`` config entry.

Stored Parquet and video signals are copied without decoding. Transformed
signals are materialized and must share one primary timeline.

Example:

    python -m utilities.convert_ds --original_ds.path /path/to/transformed_source \
        --output_path /path/to/export_root
"""

from pathlib import Path

import configuronic as cfn
import pos3

from positronic import keys
from positronic.dataset import Dataset
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.transforms.episode import Concat, Derive, FromValue, Group, Identity, Rename
from positronic.dataset.utilities.migrate_remote import migrate_dataset


@cfn.config()
def update_v0_1_0(path: str):
    return TransformedDataset(
        LocalDataset(Path(path)),
        Group(
            Derive(**{
                'controller_positions.right': Concat(
                    'right_controller_translation', 'right_controller_quaternion', timeline=RECORDED_TIME
                ),
                'robot_commands.pose': Concat(
                    'target_robot_position_translation', 'target_robot_position_quaternion', timeline=RECORDED_TIME
                ),
                keys.EE_POSE: Concat('robot_position_translation', 'robot_position_quaternion', timeline=RECORDED_TIME),
                'task': FromValue('Pick up the green cube and place it on the red cube.'),
            }),
            Rename(**{
                keys.JOINTS: 'robot_state.joints',
                keys.JOINT_VEL: 'robot_state.joints_velocity',
                keys.WRIST_IMAGE: 'image.handcam_left',
                keys.EXTERIOR_IMAGE: 'image.back_view',
            }),
            Identity(
                select=[keys.GRIP, keys.TARGET_GRIP, 'mjSTATE_FULLPHYSICS', 'mjSTATE_INTEGRATION', 'mjSTATE_WARMSTART']
            ),
        ),
    )


@cfn.config(original_ds=update_v0_1_0)
@pos3.with_mirror()
def main(output_path: str, original_ds: Dataset):
    migrate_dataset(original_ds, output_path)


if __name__ == '__main__':
    cfn.cli(main)
