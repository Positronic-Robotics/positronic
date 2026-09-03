"""The keys of a MolmoSpaces trial's params: the benchmark episode the eval selects and the horizon the sim
enforces."""

# The benchmark a trial runs, by the four directory levels that place it under the asset packs, and the episode
# within it, as ``task_params`` names them from the env's task records; ``_reset_token`` reads them back into
# the token that selects the episode.
SUITE = 'eval.suite'
SCENE_DATASET = 'eval.scene_dataset'
TASK_CONFIG = 'eval.task_config'
BENCHMARK = 'eval.benchmark'
# The four, in ``mapping.BenchmarkPath`` field order.
BENCHMARK_DIMENSIONS = (SUITE, SCENE_DATASET, TASK_CONFIG, BENCHMARK)
EPISODE_INDEX = 'eval.episode_index'
# The sim-enforced episode deadline in sim-seconds; the eval config sets the trial's backstop deadline from it.
TASK_HORIZON = 'eval.task_horizon'
