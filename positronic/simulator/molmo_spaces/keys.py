"""The keys of a MolmoSpaces trial's params: the episode the eval selects and the horizon the sim enforces."""

# The benchmark episode a trial runs, as ``task_params`` names it from the env's task records; ``_reset_token``
# reads it back into the token that selects the episode.
EPISODE_INDEX = 'eval.episode_index'
# The sim-enforced episode deadline in sim-seconds; the eval config sets the trial's backstop deadline from it.
TASK_HORIZON = 'eval.task_horizon'
