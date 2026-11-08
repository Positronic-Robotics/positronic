"""Dream-1 of Dream Labs, a hosted bimanual YAM policy, served over positronic's session protocol. NOTICE states the
terms of its API."""

from positronic import keys

# The YAM station observation of DreamAPI, as `examples/yam_mujoco/dream_control.py` of the Dream SDK sends it. The
# serve computes each hand's pose from these joints, so the client needs no IK.
JOINT_POS = 'state.{slot}.joint_pos'
GRIPPER_POS = 'state.{slot}.gripper_pos'
FRAME_WINDOW = 'image_window.{camera}'
PROMPT = 'prompt'
# The station slot of each arm.
ARM_SLOTS = {keys.LEFT_ARM: 'arm_l', keys.RIGHT_ARM: 'arm_r'}
# The station cameras: the top view and the two wrist views.
TOP_CAMERA = 'top'
LEFT_CAMERA = 'left'
RIGHT_CAMERA = 'right'
CAMERAS = (TOP_CAMERA, LEFT_CAMERA, RIGHT_CAMERA)
# The station robot fields. The serve executes its joint head, adds no joint speed cap of its own, and reads 1.0 as an
# open gripper.
STATION_ROBOT = {
    'robot.arm': 'yam',
    'robot.exec_head': 'joint',
    'robot.max_joint_vel_rad_s': 0.0,
    'robot.gripper_open': 1.0,
}
# The reply fields: per arm, rows of 6 joints and then the gripper opening, 0 closed to 1 open.
CHUNKS = {keys.LEFT_ARM: 'q_l_chunk', keys.RIGHT_ARM: 'q_r_chunk'}

# The fields of the request that the codec builds for `server.py`: the observation that DreamAPI reads, and the frame
# window of each camera, oldest frame first.
OBSERVATION = 'observation'
FRAMES = 'frames'
