"""MolmoSpaces trial parameter keys."""

# Order matches mapping.BenchmarkPath.
BENCHMARK_DIMENSIONS = ('molmo.suite', 'molmo.scene_dataset', 'molmo.task_config', 'molmo.benchmark')
EPISODE_INDEX = 'molmo.episode_index'
TASK_HORIZON = 'molmo.task_horizon'  # Simulator horizon in seconds.
EXTERIOR_SIDES = 'molmo.exterior_sides'  # The shoulder in each exterior slot, ``image.exterior`` first.
