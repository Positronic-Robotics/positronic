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
# True when the env ends the trial at the step where it sees success. A policy end then means that the env saw none.
ENV_ENDS_ON_SUCCESS = 'eval.env_ends_on_success'
# The instruction that the policy got.
TASK = 'task'
# The id of the task that a trial runs, as the benchmark names it.
BENCHMARK_TASK = 'eval.task'

# A recorded signal is the file `<signal name><SIGNAL_FILE_SUFFIX>`.
SIGNAL_FILE_SUFFIX = '.parquet'
# Each sample's time on the timeline `<name>`, in nanoseconds, is in the column `<TIMESTAMP_COLUMN_PREFIX><name>`.
TIMESTAMP_COLUMN_PREFIX = 'ts.'
# The one time column of a signal file in the format before named timelines, on the world clock.
LEGACY_TIME_COLUMN = 'timestamp'
# The column of a signal file that holds each sample's value.
VALUE_COLUMN = 'value'

# RoboLab's progress at each step, as the vector [status, completed, total, score], with the score from 0 to 1.
SUBTASK_SIGNAL = 'subtask'
SUBTASK_COMPLETED_INDEX = 1
SUBTASK_TOTAL_INDEX = 2
SUBTASK_SCORE_INDEX = 3


class PolicyEnd(StrEnum):
    """What a policy says about its attempt when it ends the trial. A recorded value never changes."""

    # The policy thinks that it did the task.
    COMPLETE = 'complete'
    # The policy stops before it did the task.
    GIVE_UP = 'give_up'
