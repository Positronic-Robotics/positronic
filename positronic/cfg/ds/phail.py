"""Public PhAIL datasets ready for training."""

import configuronic as cfn

from . import PUBLIC, local, local_all


@cfn.config()
def phail():
    """DROID teleoperation data for PhAIL tasks (towels, spoons, scissors).

    Migration process:
    1. Terminal 1: Start remote server with internal config
       uv run python -m positronic.dataset.remote_server.server \\
         --dataset=@positronic.cfg.ds.internal.droid_ds --port=8080

    2. Terminal 2: Migrate to local staging directory
       uv run python -m positronic.dataset.utilities.migrate_remote \\
         --source_url=http://localhost:8080 \\
         --dest_path=~/staging/public-datasets/phail/

    3. Terminal 1: Stop server (Ctrl+C)

    4. Upload to S3:
       aws s3 sync ~/staging/public-datasets/phail/ \\
         s3://positronic-public/datasets/phail/ \\
         --endpoint-url=https://storage.eu-north1.nebius.cloud

    Result: 12GB, 352 episodes with task labels baked in static.json
    """
    return local_all(path='s3://positronic-public/datasets/phail/', profile=PUBLIC)


@cfn.config()
def sim_stack_cubes():
    """Simulated cube stacking dataset.

    Migration process:
    1. Terminal 1: Start remote server with internal config
       uv run python -m positronic.dataset.remote_server.server \\
         --dataset=@positronic.cfg.ds.internal.cubes_sim --port=8080

    2. Terminal 2: Migrate to local staging directory
       uv run python -m positronic.dataset.utilities.migrate_remote \\
         --source_url=http://localhost:8080 \\
         --dest_path=~/staging/public-datasets/sim-stack-cubes/

    3. Terminal 1: Stop server (Ctrl+C)

    4. Upload to S3:
       aws s3 sync ~/staging/public-datasets/sim-stack-cubes/ \\
         s3://positronic-public/datasets/sim-stack-cubes/ \\
         --endpoint-url=https://storage.eu-north1.nebius.cloud

    Result: 499MB, 317 episodes with transforms baked in (ee_pose, robot_joints, task)
    """
    return local(path='s3://positronic-public/datasets/sim-stack-cubes/', profile=PUBLIC)


@cfn.config()
def sim_pick_place():
    """Simulated pick-and-place dataset.

    Migration process:
    1. Terminal 1: Start remote server with internal config
       uv run python -m positronic.dataset.remote_server.server \\
         --dataset=@positronic.cfg.ds.internal.pnp_sim --port=8080

    2. Terminal 2: Migrate to local staging directory
       uv run python -m positronic.dataset.utilities.migrate_remote \\
         --source_url=http://localhost:8080 \\
         --dest_path=~/staging/public-datasets/sim-pick-place/

    3. Terminal 1: Stop server (Ctrl+C)

    4. Upload to S3:
       aws s3 sync ~/staging/public-datasets/sim-pick-place/ \\
         s3://positronic-public/datasets/sim-pick-place/ \\
         --endpoint-url=https://storage.eu-north1.nebius.cloud

    Result: 1.3GB, 214 episodes with transforms baked in
    """
    return local(path='s3://positronic-public/datasets/sim-pick-place/', profile=PUBLIC)
