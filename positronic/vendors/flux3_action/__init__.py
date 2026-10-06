"""FLUX 3 Action DROID, served over positronic's session protocol. NOTICE states its licence."""

# The request and reply fields of Black Forest Labs' RoboLab server (`flux_action.serving.robolab`), which
# `backend.py` runs. The server stacks the wrist view above the two exterior views.
JOINT_POSITION = 'observation/joint_position'
GRIPPER_POSITION = 'observation/gripper_position'
WRIST_IMAGE = 'observation/wrist_image_left'
EXTERIOR_IMAGE_1 = 'observation/exterior_image_1_left'
EXTERIOR_IMAGE_2 = 'observation/exterior_image_2_left'
PROMPT = 'prompt'
ACTIONS = 'action'
