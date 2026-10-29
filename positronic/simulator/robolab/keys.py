"""The keys of a RoboLab task record, of the trial params the eval config owns, and the env's camera names.

The env server runs in RoboLab's own interpreter and cannot import positronic, so it reads this module as a
top-level ``keys`` from its own directory. The eval config reads it as ``positronic.simulator.robolab.keys``.
"""

# The seconds RoboLab gives an episode of this task; the eval config sets the trial's deadline from it.
EPISODE_LENGTH = 'robolab.episode_length'
# The phrasing of the instruction, which the eval config owns and ``_reset_token`` reads back.
INSTRUCTION_TYPE = 'robolab.instruction_type'

# [status, completed, total, score] for the task's subtask chain, beside the arm fields of ``protocol``.
SUBTASK = 'subtask'
# The scene state, as a flat dict from a suffix to its value. The recorder writes each entry as ``gt_state<suffix>``,
# and ``eval_vocabulary.episode.GT_STATE_SIGNAL`` states the layout that a labeler reads.
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
