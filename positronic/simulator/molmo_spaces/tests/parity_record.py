"""The interface between ``parity_native.py`` and ``parity.py``: the options one is spawned with, and the npz
field names it writes about a rollout for the other to read back.

Imported from two interpreters, like ``mapping``: as a package module by the comparison, and flat off
``PYTHONPATH`` by the native reference inside MolmoSpaces' venv. It holds names only — no imports at all — so
both shapes resolve without a fallback.

The per-camera frame hashes are one field per camera name, under ``CAM_HASH_PREFIX``.
"""

# The native reference's CLI: ``parity.py`` builds the command, ``parity_native.py``'s parser declares it.
OPT_BENCHMARK_DIR = '--benchmark_dir'
OPT_EPISODE_INDEX = '--episode_index'
OPT_SEED = '--seed'
OPT_MAX_STEPS = '--max_steps'
OPT_OUT = '--out'

CAM_HASH_PREFIX = 'cam_hash__'
CAMERA_NAMES = 'camera_names'
HORIZON_STEPS = 'native_horizon'
HORIZON_SEC = 'horizon_sec'
TERMINATION_STEP = 'termination_step'
FINAL_SUCCESS = 'final_success'
