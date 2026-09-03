"""The keys of a MolmoSpaces trial's params: the benchmark episode the eval selects and the horizon the sim
enforces."""

# The benchmark a trial runs, by the four directory levels that place it under the asset packs (in
# ``mapping.BenchmarkPath`` field order), and the episode within it, as ``task_params`` names them from the env's
# task records; ``_reset_token`` reads them back into the token that selects the episode.
BENCHMARK_DIMENSIONS = ('molmo.suite', 'molmo.scene_dataset', 'molmo.task_config', 'molmo.benchmark')
EPISODE_INDEX = 'molmo.episode_index'
# The sim-enforced episode deadline in sim-seconds; the eval config sets the trial's backstop deadline from it.
TASK_HORIZON = 'molmo.task_horizon'
