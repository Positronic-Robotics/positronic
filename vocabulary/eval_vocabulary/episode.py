"""Where an episode records its statics, its meta and its signals, and the names that a labeler reads there.

The dataset writer imports these names, so it and a labeler cannot disagree.
"""

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
ENDED_BY_OPERATOR = 'operator'
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

# RoboLab's progress on the task at each step, as the vector [status, completed, total, score]: the subtasks
# completed, the subtasks of the task, and RoboLab's score of the progress, from 0 to 1.
SUBTASK_SIGNAL = 'subtask'
SUBTASK_COMPLETED_INDEX = 1
SUBTASK_TOTAL_INDEX = 2
SUBTASK_SCORE_INDEX = 3
