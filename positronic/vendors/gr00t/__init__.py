"""What positronic states about GR00T: the nested observation it takes, and the modality configs it serves."""

from dataclasses import dataclass
from pathlib import Path

# Keys of the nested observation a GR00T session takes, which the codec writes and a warmup rebuilds.
VIDEO = 'video'
STATE = 'state'
LANGUAGE = 'language'

WRIST_IMAGE = 'wrist_image'
EXTERIOR_IMAGE = 'exterior_image_1'

# The state/action fields GR00T's data config declares and its model emits. ``GRIP`` shares a value with
# ``keys.GRIP`` by vocabulary, not by contract — renaming the positronic wire key must not rename the modality.
GRIP = 'grip'
EE_POSE = 'ee_pose'
JOINT_POSITION = 'joint_position'

TASK = 'annotation.language.language_instruction'

# The frame geometry GR00T is served at, as ``(width, height)``: what the rig is bounded to, what the codec
# resizes to, and what a warmup fills.
IMAGE_SIZE = (224, 224)


@dataclass(frozen=True)
class ModalityConfig:
    """One GR00T modality config: the fork module registering it, and the observation it declares.

    ``path`` is relative to the gr00t checkout, which is where the subprocess runs. The rest is positronic's
    own statement of what a checkpoint served under this config takes. The fork's config module is the other
    statement, and only gr00t's venv can import it, so nothing reconciles the two but the test pairing each
    config with the codec that feeds it.

    ``cameras`` and ``task_key`` default to what every config shipped here declares; gr00t's other embodiments
    name their cameras and their language field differently, so a config of your own states its own.
    """

    path: Path
    state: dict[str, int]
    cameras: tuple[str, ...] = (WRIST_IMAGE, EXTERIOR_IMAGE)
    task_key: str = TASK


_CONFIG_DIR = Path('gr00t/configs/data')

_EE_QUAT = {GRIP: 1, EE_POSE: 7}
_EE_QUAT_JOINTS = {GRIP: 1, EE_POSE: 7, JOINT_POSITION: 7}
_EE_ROT6D = {GRIP: 1, EE_POSE: 9}
_EE_ROT6D_JOINTS = {GRIP: 1, EE_POSE: 9, JOINT_POSITION: 7}

# A ``_rel`` config differs from its twin in the action space it trains, not in the observation it takes.
MODALITY_CONFIGS = {
    # 7D xyz+quat configs (absolute actions)
    'ee': ModalityConfig(_CONFIG_DIR / 'positronic_ee.py', _EE_QUAT),
    'ee_q': ModalityConfig(_CONFIG_DIR / 'positronic_ee_joints.py', _EE_QUAT_JOINTS),
    # 9D xyz+rot6d configs (supports both absolute and relative actions)
    'ee_rot6d': ModalityConfig(_CONFIG_DIR / 'positronic_ee_rot6d.py', _EE_ROT6D),
    'ee_rot6d_rel': ModalityConfig(_CONFIG_DIR / 'positronic_ee_rot6d_rel.py', _EE_ROT6D),
    'ee_rot6d_q': ModalityConfig(_CONFIG_DIR / 'positronic_ee_rot6d_joints.py', _EE_ROT6D_JOINTS),
    'ee_rot6d_q_rel': ModalityConfig(_CONFIG_DIR / 'positronic_ee_rot6d_joints_rel.py', _EE_ROT6D_JOINTS),
    # Joint-space action config (for IK-derived targets)
    'joints': ModalityConfig(_CONFIG_DIR / 'positronic_joints.py', {GRIP: 1, JOINT_POSITION: 7}),
}
