"""Native observation fields and the Positronic client contract."""

JOINT_POSITION = 'observation/joint_position'
GRIPPER_POSITION = 'observation/gripper_position'
# The flange pose of an EE-50 checkpoint, [x, y, z, qw, qx, qy, qz] in the robot base frame.
EE_POSE_FIELD = 'observation/ee_pose'
WRIST = 'observation/wrist_image_left'
EXTERIOR = 'observation/exterior_image_1_left'
EXTERIOR_2 = 'observation/exterior_image_2_left'
CAMERAS = (WRIST, EXTERIOR, EXTERIOR_2)
PROMPT = 'prompt'
ACTION = 'action'
VIEW_SIZE = (640, 360)

JOINTS = 'robot_state.q'
EE_POSE = 'robot_state.ee_pose'
TARGET_EE_POSE = 'robot_command.pose'
ROBOT_COMMAND = 'robot_command'
GRIP = 'grip'
TASK = 'task'
WRIST_IMAGE = 'image.wrist'
EXTERIOR_IMAGE = 'image.exterior'
EXTERIOR_IMAGE_2 = 'image.exterior_2'
PLAN = 'plan'
CALL_TIMINGS = 'call_timings'
ELAPSED_SEC = 'elapsed_sec'
ACTION_FPS = 'action_fps'
TYPE = 'type'
CHECKPOINT_ID = 'checkpoint_id'
AUTH_TOKEN_ENV = 'AUTH_TOKEN'
MODEL_SOURCE_ENV = 'FLUX_POLICY_REPO'
MODEL_REVISION_ENV = 'FLUX_POLICY_REVISION'
MODEL_SUBFOLDER_ENV = 'FLUX_POLICY_SUBFOLDER'
# An EE-50 package holds the statistics it trained with beside its config.
EE50_STATS_FILE = 'ee50_stats.json'
EE50_MODALITY = 'action_prediction_ee50'


class RTC:
    """Remaining-plan fields and Flux3's session settings."""

    FIELD = 'rtc'
    ACTION_PREFIX = 'action_prefix'
    PREFIX_ACTIONS = 'actions'
    CHUNK_OFFSET_SEC = 'chunk_offset_sec'
    OBSERVATION_OFFSET_SEC = 'observation_offset_sec'
    CALL_AFTER_SEC = 'call_after_sec'
    MAX_DELAY_SEC = 'max_delay_sec'
