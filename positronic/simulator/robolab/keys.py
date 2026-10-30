"""The keys of a RoboLab task record, of the trial params the eval config owns, of a reset token, and the env's
camera names.

The env server runs in RoboLab's own interpreter and cannot import positronic, so it reads this module as a
top-level ``keys`` from its own directory. The eval config reads it as ``positronic.simulator.robolab.keys``.
"""

# The fields of a task record that the env answers ``tasks`` with.
RECORD_NAME = 'name'
RECORD_EPISODE_LENGTH_S = 'episode_length_s'

# The seconds RoboLab gives an episode of this task; without an override, the trial's deadline follows them.
EPISODE_LENGTH = 'robolab.episode_length'
# The seconds that replace ``EPISODE_LENGTH`` in the env, and the deadline follows them.
# Absent when RoboLab's budget holds.
EPISODE_LENGTH_OVERRIDE = 'robolab.episode_length_override'
# The phrasing of the instruction, which the eval config owns and ``_reset_token`` reads back.
INSTRUCTION_TYPE = 'robolab.instruction_type'

# The fields of a reset token. The episode length is absent when the task sets it, the state when the scene randomizes.
TOKEN_TASK = 'task'
TOKEN_INSTRUCTION_TYPE = 'instruction_type'
TOKEN_EPISODE_LENGTH_S = 'episode_length_s'
TOKEN_STATE = 'state'

# [status, completed, total, score] for the task's subtask chain, beside the arm fields of ``protocol``.
SUBTASK = 'subtask'
# The scene state, as a flat dict from a suffix to its value. The env server cannot import ``eval_vocabulary``. So
# the words below restate the layout of ``eval_vocabulary.episode.GT_STATE_SIGNAL``, and a test keeps them the same.
GT_STATE = 'gt_state'
GT_STATE_OBJECTS = 'objects'
GT_STATE_POSE = 'pose'
GT_STATE_GRIPPER_CONTACT = 'gripper_contact'
GT_STATE_PROGRESS = 'progress'

# The cameras RoboLab renders, named as the env's ``image_obs`` group names them.
OVER_SHOULDER_LEFT_CAMERA = 'over_shoulder_left_camera'
OVER_SHOULDER_RIGHT_CAMERA = 'over_shoulder_right_camera'
WRIST_CAMERA = 'wrist_cam'

# The camera sets a run picks from. The names follow RoboLab's own presets.
WRIST_LEFT_RIGHT = 'wrist_left_right'
WRIST_LEFT = 'wrist_left'
WRIST_RIGHT = 'wrist_right'
CAMERA_SETS = {
    WRIST_LEFT_RIGHT: (OVER_SHOULDER_LEFT_CAMERA, OVER_SHOULDER_RIGHT_CAMERA, WRIST_CAMERA),
    WRIST_LEFT: (OVER_SHOULDER_LEFT_CAMERA, WRIST_CAMERA),
    WRIST_RIGHT: (OVER_SHOULDER_RIGHT_CAMERA, WRIST_CAMERA),
}
