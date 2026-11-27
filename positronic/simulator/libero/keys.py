"""LIBERO scene settings, privileged observation names, and the env server's config variable."""

SIM_STATE = 'sim_state'

# The environment variable LIBERO reads its config dir from.
CONFIG_PATH_ENV = 'LIBERO_CONFIG_PATH'

# The scene a trial runs: the task, as ``task_params`` names it from the env's task records, plus the render
# and control settings the eval config owns; ``_reset_token`` reads them back. The server caches an env by
# ``(suite, task_id, camera_resolution, control_mode)``.
SUITE = 'libero.suite'
TASK_ID = 'libero.task_id'
CAMERA_RESOLUTION = 'libero.camera_resolution'
CONTROL_MODE = 'libero.control_mode'
SETTLE_STEPS = 'libero.settle_steps'
