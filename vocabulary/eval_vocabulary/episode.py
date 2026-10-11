"""Where an episode records its statics, its meta and its signals, and the names that a labeler reads there.

The dataset writer and a labeler both import these names, so they cannot disagree about them.
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
# The seconds from the start of the trial to its end.
DURATION = 'eval.duration'
# Who ended the trial, when it was not the task's own ground truth. An env's terminal leaves it absent.
ENDED_BY = 'eval.ended_by'
ENDED_BY_OPERATOR = 'operator'
# The instruction that the policy got.
TASK = 'task'

# A recorded signal is the file `<signal name><SIGNAL_FILE_SUFFIX>`. A signal file in the format before named
# timelines keeps the time of each sample, in nanoseconds, in the column `LEGACY_TIME_COLUMN`.
SIGNAL_FILE_SUFFIX = '.parquet'
LEGACY_TIME_COLUMN = 'timestamp'
