"""The npz field names ``parity_native.py`` writes about a rollout and ``parity.py`` reads back.

Imported from two interpreters, like ``mapping``: as a package module by the comparison, and flat off
``PYTHONPATH`` by the native reference inside MolmoSpaces' venv. It holds names only — no imports at all — so
both shapes resolve without a fallback.

The per-camera frame hashes are one field per camera name, under ``CAM_HASH_PREFIX``.
"""

CAM_HASH_PREFIX = 'cam_hash__'
CAMERA_NAMES = 'camera_names'
HORIZON_STEPS = 'native_horizon'
HORIZON_SEC = 'horizon_sec'
TERMINATION_STEP = 'termination_step'
FINAL_SUCCESS = 'final_success'
