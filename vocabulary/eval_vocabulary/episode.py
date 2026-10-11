"""Where an episode records its statics, its meta and its signals, and the names that a labeler reads there.

A labeler imports these names, and so does the dataset writer. A test pins the copy of `EMITTED_WORLD` that pimm
keeps. So the writer and a labeler cannot disagree about them.
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

# A recorded signal is the file `<signal name><SIGNAL_FILE_SUFFIX>`. It keeps the time of each sample, in nanoseconds,
# on each timeline `<name>` in the column `<TIMESTAMP_COLUMN_PREFIX><name>`. On `EMITTED_WORLD`, the time is when the
# sample was made, on the world clock that a trial runs on. A signal file in the format before named timelines keeps
# one world time per sample, in the column `LEGACY_TIME_COLUMN`.
SIGNAL_FILE_SUFFIX = '.parquet'
TIMESTAMP_COLUMN_PREFIX = 'ts.'
EMITTED_WORLD = 'emitted.world'
LEGACY_TIME_COLUMN = 'timestamp'
