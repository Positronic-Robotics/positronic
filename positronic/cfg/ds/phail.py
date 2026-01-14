"""Public PhAIL datasets ready for training.

All datasets are hosted at s3://positronic-public/datasets/ and have transforms
baked in during migration (no runtime transforms).

Migration Process (for future reference):
------------------------------------------
To migrate a dataset from internal s3://raw/ to public s3://positronic-public/:

1. Start remote server serving the internal dataset:
   uv run python -m positronic.dataset.remote_server.server \\
     --dataset=@positronic.cfg.ds.internal.<INTERNAL_CONFIG> --port=8080

2. Run migration to local staging directory:
   uv run python -m positronic.dataset.utilities.migrate_remote \\
     --source_url=http://localhost:8080 \\
     --dest_path=~/staging/public-datasets/<DATASET_NAME>/

3. Stop the remote server (Ctrl+C)

4. Upload to S3:
   aws s3 sync ~/staging/public-datasets/<DATASET_NAME>/ \\
     s3://positronic-public/datasets/<DATASET_NAME>/ \\
     --endpoint-url=https://storage.eu-north1.nebius.cloud

See internal configs (cfg/ds/internal.py) for source dataset definitions.
"""

from . import PUBLIC, local, local_all

# DROID teleoperation data for PhAIL tasks (towels, spoons, scissors)
# Migrated from: @positronic.cfg.ds.internal.droid_ds
# Size: 12GB, 352 episodes with task labels baked in static.json
phail = local_all.override(path='s3://positronic-public/datasets/phail/', profile=PUBLIC)

# Simulated cube stacking dataset
# Migrated from: @positronic.cfg.ds.internal.cubes_sim
# Size: 499MB, 317 episodes with transforms baked in (ee_pose, robot_joints, task)
sim_stack_cubes = local.override(path='s3://positronic-public/datasets/sim-stack-cubes/', profile=PUBLIC)

# Simulated pick-and-place dataset
# Migrated from: @positronic.cfg.ds.internal.pnp_sim
# Size: 1.3GB, 214 episodes with transforms baked in
sim_pick_place = local.override(path='s3://positronic-public/datasets/sim-pick-place/', profile=PUBLIC)
