"""GR00T DROID model and wire vocabulary."""

VIDEO = 'video'
STATE = 'state'
LANGUAGE = 'language'
WRIST_IMAGE = 'wrist_image_left'
EXTERIOR_IMAGE = 'exterior_image_1_left'
EXTERIOR_IMAGE_2 = 'exterior_image_2_left'
GRIP = 'gripper_position'
EE_POSE = 'eef_9d'
JOINT_POSITION = 'joint_position'
TASK = 'annotation.language.language_instruction'
EMBODIMENT = 'oxe_droid_relative_eef_relative_joint'
BASE_MODEL = 'nvidia/GR00T-N1.7-DROID'
VENV = '/opt/gr00t-venv'

# Width, height at the DROID robot-client boundary; the checkpoint processor owns subsequent resizing/cropping.
IMAGE_SIZE = (320, 180)
STATE_DIMS = {EE_POSE: 9, GRIP: 1, JOINT_POSITION: 7}
