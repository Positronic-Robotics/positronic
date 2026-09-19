"""The keys of a LIBERO scene: the task the eval config selects and the settings it owns."""

# The scene a trial runs: the task, as ``task_params`` names it from the env's task records, plus the render
# and control settings the eval config owns; ``_reset_token`` reads them back. The server caches an env by
# ``(suite, task_id, camera_resolution, control_mode)``.
SUITE = 'libero.suite'
TASK_ID = 'libero.task_id'
CAMERA_RESOLUTION = 'libero.camera_resolution'
CONTROL_MODE = 'libero.control_mode'
SETTLE_STEPS = 'libero.settle_steps'
