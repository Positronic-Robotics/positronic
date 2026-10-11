"""Where an episode records its statics, its meta and its signals, and the names that a labeler reads there.

The dataset writer imports these names, so it and a labeler cannot disagree. `EMITTED_WORLD` must equal pimm's name
for the emitted world timeline.
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

# A recorded signal is the file `<signal name><SIGNAL_FILE_SUFFIX>`.
SIGNAL_FILE_SUFFIX = '.parquet'
# Each sample's time on the timeline `<name>`, in nanoseconds, is in the column `<TIMESTAMP_COLUMN_PREFIX><name>`.
TIMESTAMP_COLUMN_PREFIX = 'ts.'
# The timeline of when each sample was made, on the world clock that a trial runs on.
EMITTED_WORLD = 'emitted.world'
# The one time column of a signal file in the format before named timelines, on the world clock.
LEGACY_TIME_COLUMN = 'timestamp'
