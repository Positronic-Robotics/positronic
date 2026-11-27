"""Configs of the round driver (``positronic.dagger.rounds``): the rig and its teleoperator, the tasks and the model.

Run the round driver from the root of the repository, with the ``lerobot_0_3_3`` extra, so that ``python`` in the
commands of the model has LeRobot. A run gives the license of each of its sources, and lists them in
``training_licenses``::

    uv run --locked --extra lerobot_0_3_3 python -m positronic.dagger mujoco_lift_cube --output_dir=~/dagger/1 \
      --round_0_license=internal --rig.license=internal --model.license=internal --training_licenses='["internal"]'
"""

from collections.abc import Iterator
from itertools import count

import configuronic as cfn

from positronic import keys
from positronic.cfg.simulator import lift_cube_loaders
from positronic.dagger.mujoco import MujocoRig
from positronic.dagger.rounds import ModelCommands, run
from positronic.eval import Task
from positronic.eval import keys as eval_keys
from positronic.utils import package_assets_path

# The Franka of ``MujocoSim`` with the ``lift_cube`` scene, and the wrist and the exterior cameras that
# ``lerobot_act`` learns from.
mujoco_rig = cfn.Config(
    MujocoRig,
    mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
    loaders=lift_cube_loaders,
    cameras={keys.WRIST_IMAGE: 'handcam_left_ph', keys.EXTERIOR_IMAGE: 'agentview'},
)


@cfn.config(timeout_sec=15.0, first_seed=0)
def lift_cube(timeout_sec: float, first_seed: int) -> Iterator[Task]:
    """The lift-cube task of ``MujocoSim``, with the seeds from ``first_seed`` on. The seed draws the place of the
    cube."""
    for seed in count(first_seed):
        yield Task(
            'lift the cube',
            timeout_sec,
            prepare_args={eval_keys.SCENE: {eval_keys.SEED: seed}},
            meta={eval_keys.SEED: seed},
            charge_inference_time=False,
        )


# The modules of LeRobot ACT. ``python`` is the interpreter of the environment that runs the round driver.
_LEROBOT = 'positronic.vendors.lerobot_0_3_3'
_GRIPPER_FRAME = '@positronic.cfg.codecs.gripper_frame'


@cfn.config(num_train_steps=50_000)
def lerobot_act(license: str, num_train_steps: int) -> ModelCommands:
    """LeRobot ACT of ``positronic.vendors.lerobot_0_3_3``, which learns the pose and the grip command of the arm from
    the wrist and the exterior images, the pose and the grip. ``license`` is the license of each checkpoint."""
    return ModelCommands(
        labels=(keys.TARGET_EE_POSE, keys.TARGET_GRIP),
        convert=(
            *('python', '-m', f'{_LEROBOT}.to_lerobot', 'convert'),
            *('--dataset.dataset=.local', '--dataset.dataset.path={training_data}'),
            *(f'--dataset.codec=@{_LEROBOT}.codecs.ee', f'--dataset.codec.ee_frame={_GRIPPER_FRAME}'),
            '--output_dir={converted}',
        ),
        train=(
            *('python', '-m', f'{_LEROBOT}.train', '--input_path={converted}'),
            *('--output_dir={checkpoint.parent}', '--exp_name={checkpoint.name}'),
            f'--num_train_steps={num_train_steps}',
        ),
        serve=(
            *('python', '-m', f'{_LEROBOT}.server', 'ee', '--model.checkpoints_dir={checkpoint}'),
            *('--model.license={license}', f'--pipeline.ee_frame={_GRIPPER_FRAME}'),
            *('--websocket.served_address.host={host}', '--websocket.served_address.port={port}'),
        ),
        license=license,
    )


# DAgger rounds of ``lift_cube`` on the Franka of ``MujocoSim``, with LeRobot ACT.
mujoco_lift_cube = run.override(rig=mujoco_rig, tasks=lift_cube, model=lerobot_act, rounds=3, episodes=20)
