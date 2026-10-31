"""Where an episode records its statics, its meta and its signals, and the names that a labeler reads there.

The dataset writer imports these names, so it and a labeler cannot disagree.
"""

from enum import StrEnum

# The file in an episode's directory that holds its statics as one JSON object.
STATIC_FILE = 'static.json'
# The file in an episode's directory that holds its meta as one JSON object, and the episode's id in it.
META_FILE = 'meta.json'
UID = 'uid'

# The success that the env reports when the trial ends.
SUCCESS = 'eval.success'
# False when the trial ran out of time before it ended.
TERMINATED = 'eval.terminated'
# Who ended the trial, when it was not the task's own ground truth. An env's terminal leaves it absent.
ENDED_BY = 'eval.ended_by'
# A person at the station ended the trial: a button on a console, or a key.
ENDED_BY_OPERATOR = 'operator'
# The policy ended the trial: its run returned a `PolicyEnd`.
ENDED_BY_POLICY = 'policy'
# The `PolicyEnd` of a trial that the policy ended. It is the policy's claim, not a verdict.
POLICY_END = 'eval.policy_end'
# The instruction that the policy got, as it was at the end of the trial.
TASK = 'task'
# The signal of each instruction that the policy got: a sample at the first observation, and one at each change.
INSTRUCTION = 'eval.instruction'
# The id of the task that a trial runs, as the benchmark names it.
BENCHMARK_TASK = 'eval.task'
# The seed that draws a trial's scene, in the trial's params and in its episode's statics.
SEED = 'eval.seed'

# A recorded signal is the file `<signal name><SIGNAL_FILE_SUFFIX>`.
SIGNAL_FILE_SUFFIX = '.parquet'
# Each sample's time on the timeline `<name>`, in nanoseconds, is in the column `<TIMESTAMP_COLUMN_PREFIX><name>`.
TIMESTAMP_COLUMN_PREFIX = 'ts.'
# The one time column of a signal file in the format before named timelines.
LEGACY_TIME_COLUMN = 'timestamp'
# The one time column of a video frames index in the format before named timelines.
LEGACY_FRAMES_TIME_COLUMN = 'ts_ns'
# The schema metadata key that names the timeline of either legacy time column. A file without it is on `recorded`.
TIMELINE_METADATA_KEY = b'positronic.timeline'
# The column of a signal file that holds each sample's value.
VALUE_COLUMN = 'value'

# The arm's measured state. A recorded state unfolds into one signal per field, `<ROBOT_STATE><suffix>`. The field at
# `EE_POSE_SUFFIX` is the [x, y, z, qw, qx, qy, qz] of the end effector. A RoboLab episode records it at the rig's
# gripper site, in the frames of `GT_STATE_SIGNAL`'s poses.
ROBOT_STATE = 'robot_state'
EE_POSE_SUFFIX = '.ee_pose'

# RoboLab's progress at each step, as the vector [status, completed, total, score], with the score from 0 to 1.
SUBTASK_SIGNAL = 'subtask'
SUBTASK_COMPLETED_INDEX = 1
SUBTASK_TOTAL_INDEX = 2
SUBTASK_SCORE_INDEX = 3
# RoboLab's scene state at each step, as one signal per value:
# `<GT_STATE_SIGNAL>.<GT_STATE_OBJECTS>.<object>.<GT_STATE_POSE>`, the [x, y, z, qw, qx, qy, qz] of the object, with
# the position in the frame of the scene clone and the rotation in the world frame;
# `<GT_STATE_SIGNAL>.<GT_STATE_OBJECTS>.<object>.<GT_STATE_GRIPPER_CONTACT>`, True when RoboLab's gripper contact
# sensor touches the object. On the DROID rig that sensor is the left inner finger of the Robotiq 2F-85, at 0.1 N.
# The value is False on the frame of a reset;
# `<GT_STATE_SIGNAL>.<GT_STATE_PROGRESS>.<subtask>.<group>.<index>.<condition>`, True when RoboLab counts the
# condition at that index of the group's chain in that subtask as done. `<subtask>` is the 0-based index of the
# subtask. It equals the count at `SUBTASK_COMPLETED_INDEX` while the subtask is current. `<group>` is RoboLab's
# name for the chain, often an object name. A subtask's signals have samples while it is the current subtask, and
# on the step that completes it.
GT_STATE_SIGNAL = 'gt_state'
GT_STATE_OBJECTS = 'objects'
GT_STATE_POSE = 'pose'
GT_STATE_GRIPPER_CONTACT = 'gripper_contact'
GT_STATE_PROGRESS = 'progress'
# The `<condition>` that opens each pick chain of RoboLab's tasks: the arm holds the object.
GT_STATE_GRABBED = 'object_grabbed'

# ABC's task evaluator records each of its metrics as the signal `<ABC_TASK_EVAL_SIGNAL>.<metric>`, under ABC's name.
ABC_TASK_EVAL_SIGNAL = 'task_eval'
# ABC's progress through the task at each step, from 0 to 1. ABC scores progress as its maximum over the episode.
ABC_REWARD_SIGNAL = f'{ABC_TASK_EVAL_SIGNAL}.reward'


class PolicyEnd(StrEnum):
    """What a policy says about its attempt when it ends the trial. A recorded value never changes."""

    # The policy thinks that it did the task.
    COMPLETE = 'complete'
    # The policy stops before it did the task.
    GIVE_UP = 'give_up'
