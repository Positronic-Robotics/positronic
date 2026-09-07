"""The keys of a RoboLab task record and of the trial params the eval config owns."""

# The seconds RoboLab gives an episode of this task; the eval config sets the trial's deadline from it.
EPISODE_LENGTH = 'eval.episode_length'
# The phrasing of the instruction, which the eval config owns and ``_reset_token`` reads back.
INSTRUCTION_TYPE = 'eval.instruction_type'
