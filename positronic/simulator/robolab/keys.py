"""The keys of a RoboLab task record, of the trial params the eval config owns, of a reset token, the robots, and the
env's camera names.

The env server runs in RoboLab's own interpreter and cannot import positronic, so it reads this module as a
top-level ``keys`` from its own directory. The eval config reads it as ``positronic.simulator.robolab.keys``.
"""

# In a selection: the phrasing of the instruction that each task record states.
SELECT_INSTRUCTION_TYPE = 'instruction_type'

# The fields of a task record that the env answers ``tasks`` with.
RECORD_NAME = 'name'
RECORD_EPISODE_LENGTH_S = 'episode_length_s'
# The phrasing that the record's instruction states. The reset reads it back, so the reset builds that instruction.
RECORD_INSTRUCTION_TYPE = 'instruction_type'

# The seconds RoboLab gives an episode of this task; without an override, the trial's deadline follows them.
EPISODE_LENGTH = 'robolab.episode_length'
# The seconds that replace ``EPISODE_LENGTH`` in the env, and the deadline follows them.
# Absent when RoboLab's budget holds.
EPISODE_LENGTH_OVERRIDE = 'robolab.episode_length_override'
# The phrasing of the instruction, which the task record states and ``_reset_token`` reads back.
INSTRUCTION_TYPE = 'robolab.instruction_type'
# The camera set of the run, in ``ROBOT_CAMERA_SETS``, which ``_reset_token`` reads back.
CAMERAS = 'robolab.cameras'

# A step frame's field when the env ended a clone's episode because the clone cannot go on. Its value is the fault.
FRAME_FAULT = 'fault'
# The fault of an episode that the env ended because the clone could not go on. Absent for any other episode.
FAULT = 'robolab.fault'
# The fault of a clone whose arm joints or joint velocities hold a value that is not finite.
FAULT_NONFINITE_ARM = 'nonfinite_arm_state'
# The fault of a clone whose finite arm state is far outside what the arm can have (``arm_fault.arm_faults``).
FAULT_ARM_OUT_OF_RANGE = 'arm_state_out_of_range'

# The fields of a reset token. The episode length is absent when the task sets it, the state when the scene randomizes.
TOKEN_TASK = 'task'
TOKEN_INSTRUCTION_TYPE = 'instruction_type'
TOKEN_EPISODE_LENGTH_S = 'episode_length_s'
TOKEN_CAMERAS = 'cameras'
TOKEN_STATE = 'state'

# The instruction text in the phrasing of the reset, in the scene meta of a reset frame.
META_INSTRUCTION = 'task'

# [status, completed, total, score] for the task's subtask chain, beside the arm fields of ``protocol``.
SUBTASK = 'subtask'
# The scene state, as a flat dict from a suffix to its value. The env server cannot import ``eval_vocabulary``. So
# the words below restate the layout of ``eval_vocabulary.episode.GT_STATE_SIGNAL``, and a test keeps them the same.
GT_STATE = 'gt_state'
GT_STATE_OBJECTS = 'objects'
GT_STATE_POSE = 'pose'
GT_STATE_GRIPPER_CONTACT = 'gripper_contact'
GT_STATE_PROGRESS = 'progress'

# The robots an env server can put in the RoboLab scenes: the DROID rig (a Franka arm with a Robotiq 2F-85), or the
# two-arm i2rt YAM. A keys module imports nothing, so the names are strings and ``ROBOTS`` lists them.
DROID = 'droid'
YAM = 'yam'

# The cameras RoboLab renders, named as the env's ``image_obs`` group names them.
OVER_SHOULDER_LEFT_CAMERA = 'over_shoulder_left_camera'
OVER_SHOULDER_RIGHT_CAMERA = 'over_shoulder_right_camera'
WRIST_CAMERA = 'wrist_cam'
# The cameras of the YAM robot: one over the arms, one on each wrist.
TOP_CAMERA = 'top_cam'
LEFT_WRIST_CAMERA = 'left_wrist_cam'
RIGHT_WRIST_CAMERA = 'right_wrist_cam'
# The exteriors a task library places with its own ``cameras.py``; RoboLab's presets hold neither.
EXTERIOR_CAMERA = 'exterior_camera'
EXTERIOR_2_CAMERA = 'exterior_2_camera'

# The camera sets a run of the DROID rig picks from. The names follow RoboLab's own presets.
WRIST_LEFT_RIGHT = 'wrist_left_right'
WRIST_LEFT = 'wrist_left'
WRIST_RIGHT = 'wrist_right'
# Served only when a task library's ``cameras.py`` returns a set of this name.
EXTERIOR_PAIR_WRIST = 'exterior_pair_wrist'
CAMERA_SETS = {
    WRIST_LEFT_RIGHT: (OVER_SHOULDER_LEFT_CAMERA, OVER_SHOULDER_RIGHT_CAMERA, WRIST_CAMERA),
    WRIST_LEFT: (OVER_SHOULDER_LEFT_CAMERA, WRIST_CAMERA),
    WRIST_RIGHT: (OVER_SHOULDER_RIGHT_CAMERA, WRIST_CAMERA),
    EXTERIOR_PAIR_WRIST: (EXTERIOR_CAMERA, EXTERIOR_2_CAMERA, WRIST_CAMERA),
}
# The camera set of the YAM robot.
TOP_AND_WRISTS = 'top_and_wrists'
YAM_CAMERA_SETS = {TOP_AND_WRISTS: (TOP_CAMERA, LEFT_WRIST_CAMERA, RIGHT_WRIST_CAMERA)}
ROBOT_CAMERA_SETS = {DROID: CAMERA_SETS, YAM: YAM_CAMERA_SETS}
ROBOTS = tuple(ROBOT_CAMERA_SETS)
# The camera set a server of each robot renders when the launch names none.
DEFAULT_CAMERA_SETS = {DROID: WRIST_LEFT_RIGHT, YAM: TOP_AND_WRISTS}


def camera_set(robot: str, name: str) -> tuple[str, ...]:
    """The cameras of the set ``name`` of ``robot``."""
    sets = ROBOT_CAMERA_SETS[robot]
    if name not in sets:
        raise ValueError(f'the {robot} robot has the camera sets {sorted(sets)}, not {name!r}')
    return sets[name]
