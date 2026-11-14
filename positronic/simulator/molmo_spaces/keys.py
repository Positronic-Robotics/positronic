"""MolmoSpaces trial parameter keys."""

# Order matches mapping.BenchmarkPath.
BENCHMARK_DIMENSIONS = ('molmo.suite', 'molmo.scene_dataset', 'molmo.task_config', 'molmo.benchmark')
EPISODE_INDEX = 'molmo.episode_index'
TASK_HORIZON = 'molmo.task_horizon'  # Simulator horizon in seconds.
# The horizon in seconds that replaces the horizon of the benchmark. It is absent when the benchmark sets it.
TASK_HORIZON_OVERRIDE = 'molmo.task_horizon_override'
EXTERIOR_SIDES = 'molmo.exterior_sides'  # The shoulder in each exterior slot, ``image.exterior`` first.
# The episode camera that the env serves under each image key in place of its default camera.
EPISODE_CAMERAS = 'molmo.episode_cameras'
# The file name and content digest of the hook file of a server that the eval launches, or None without a hook.
ENV_HOOK = 'molmo.env_hook'
