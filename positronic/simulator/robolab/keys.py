"""What the eval config, the launcher and the env server must spell the same way.

The keys of a RoboLab task record and of the trial params, the env's camera names and sets, and the flag
that carries the set to the server.

The env server runs in RoboLab's own interpreter and cannot import positronic, so it reads this module as a
top-level ``keys`` from its own directory. The eval config reads it as ``positronic.simulator.robolab.keys``.
"""

# The seconds RoboLab gives an episode of this task; the eval config sets the trial's deadline from it.
EPISODE_LENGTH = 'eval.episode_length'
# The phrasing of the instruction, which the eval config owns and ``_reset_token`` reads back.
INSTRUCTION_TYPE = 'eval.instruction_type'

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

# The flag the launcher spells and the server parses to carry the set name.
CAMERAS_FLAG = '--cameras'
