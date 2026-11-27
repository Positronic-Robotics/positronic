"""Configs of the round driver (``positronic.dagger.rounds``): the rig and its teleoperator, the tasks and the model.

Run the round driver from the root of the repository, with the ``lerobot_0_3_3`` extra, so that ``python`` in the
commands of the model has LeRobot. A run gives the license of each of its sources, and lists them in
``training_licenses``::

    uv run --locked --extra lerobot_0_3_3 python -m positronic.dagger mujoco_lift_cube --output_dir=~/dagger/1 \
      --round_0_license=internal --rig.license=internal --model.license=internal --training_licenses='["internal"]'
"""

import math
from collections.abc import Iterator, Sequence
from functools import partial
from itertools import count

import configuronic as cfn
import numpy as np

from positronic import geom, keys
from positronic.cfg.eval.sim.robolab import yam_camera_dict, yam_embodiment
from positronic.cfg.simulator import lift_cube_loaders
from positronic.dagger.env_server import EnvServerRig, MakeTeleoperator
from positronic.dagger.mujoco import MujocoRig
from positronic.dagger.rounds import ModelCommands, Rig, run
from positronic.eval import Task
from positronic.eval import keys as eval_keys
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab.adapter import RobolabYamAdapter
from positronic.simulator.robolab.expert import BananaInBowlExpert, layout
from positronic.simulator.robolab.launcher import serve_robolab
from positronic.utils import package_assets_path

# Where a scripted expert parks the grip point of a YAM arm, from the mount. Nearer the mount or higher, the grip cannot
# point straight down: joint 4 gets to its limit, and the IK finds no solution.
YAM_PARK = (0.2, 0.0, 0.2)
# How far the deadline of a RoboLab trial is past the episode length, so that the verdict of the env ends each trial,
# as in ``positronic.cfg.eval.sim.robolab``.
_VERDICT_MARGIN_SEC = 10.0

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


@cfn.config(codec='ee', arms=keys.ONE_ARM, num_train_steps=50_000)
def lerobot_act(codec: str, arms: tuple[str | None, ...], license: str, num_train_steps: int) -> ModelCommands:
    """LeRobot ACT of ``positronic.vendors.lerobot_0_3_3`` with the codec and the server pipeline ``codec``, which
    learns the pose and the grip command of each arm of ``arms``. ``license`` is the license of each checkpoint."""
    labels = (
        *(keys.arm_key(keys.TARGET_EE_POSE, arm) for arm in arms),
        *(keys.arm_channel(keys.TARGET_GRIP, arm) for arm in arms),
    )
    return ModelCommands(
        labels=labels,
        convert=(
            *('python', '-m', f'{_LEROBOT}.to_lerobot', 'convert'),
            *('--dataset.dataset=.local', '--dataset.dataset.path={training_data}'),
            *(f'--dataset.codec=@{_LEROBOT}.codecs.{codec}', f'--dataset.codec.ee_frame={_GRIPPER_FRAME}'),
            '--output_dir={converted}',
        ),
        train=(
            *('python', '-m', f'{_LEROBOT}.train', '--input_path={converted}', f'--codec=@{_LEROBOT}.codecs.{codec}'),
            *('--output_dir={checkpoint.parent}', '--exp_name={checkpoint.name}'),
            f'--num_train_steps={num_train_steps}',
        ),
        serve=(
            *('python', '-m', f'{_LEROBOT}.server', codec, '--model.checkpoints_dir={checkpoint}'),
            *('--model.license={license}', f'--pipeline.ee_frame={_GRIPPER_FRAME}'),
            *('--websocket.served_address.host={host}', '--websocket.served_address.port={port}'),
        ),
        license=license,
    )


# DAgger rounds of ``lift_cube`` on the Franka of ``MujocoSim``, with LeRobot ACT.
mujoco_lift_cube = run.override(rig=mujoco_rig, tasks=lift_cube, model=lerobot_act, rounds=3, episodes=20)


@cfn.config(cameras=robolab_keys.TOP_AND_WRISTS)
def robolab_rig(cameras: str, teleoperator: MakeTeleoperator) -> Rig:
    """The two-arm YAM in the RoboLab env server, which renders the camera set ``cameras``. The teleoperator is the
    expert ``teleoperator``. The tasks give the same camera set."""
    camera_dict = yam_camera_dict(robolab_keys.camera_set(robolab_keys.YAM, cameras))
    return EnvServerRig(
        RobolabYamAdapter(camera_dict),
        partial(serve_robolab, robot=robolab_keys.YAM),
        embodiment=partial(yam_embodiment, camera_dict=camera_dict),
        ground_truth=robolab_keys.GT_STATE,
        teleoperator=teleoperator,
    )


@cfn.config(timeout_sec=30.0, first_seed=0, move_m=0.05, turn_deg=30.0, cameras=robolab_keys.TOP_AND_WRISTS)
def banana_in_bowl(timeout_sec: float, first_seed: int, move_m: float, turn_deg: float, cameras: str) -> Iterator[Task]:
    """RoboLab's ``BananaInBowlTask``, with the seeds from ``first_seed`` on and the camera set ``cameras`` of the rig.
    The env ends each episode at ``timeout_sec``. The layout of each seed moves the banana and the bowl up to ``move_m``
    from the scene file, and turns the banana up to ``turn_deg``."""
    for seed in count(first_seed):
        scene = {
            eval_keys.TASK: 'BananaInBowlTask',
            robolab_keys.INSTRUCTION_TYPE: 'default',
            robolab_keys.CAMERAS: cameras,
            robolab_keys.EPISODE_LENGTH_OVERRIDE: timeout_sec,
            robolab_keys.STATE: layout(np.random.default_rng(seed), move_m, math.radians(turn_deg)),
        }
        yield Task(
            'put the banana in the bowl',
            timeout_sec + _VERDICT_MARGIN_SEC,
            prepare_args={eval_keys.SCENE: scene},
            meta={eval_keys.SEED: seed},
            charge_inference_time=False,
        )


# The statics of the YAM in RoboLab, as positronic-next measured them on its RoboLab YAM. The pads of its linear
# gripper close 40.8 mm off ``GRIPPER_SITE``. Its grip points straight down at the grasp height up to 0.48 m from the
# mount. Joint 6 is at 0 at the yaw 90 degrees from the mount heading, and spans +-120 degrees. The IK misses past about
# 110 degrees, so the expert keeps to 100.
@cfn.config(
    grip_point=(0.0, 0.0408, -0.003),
    reach=0.48,
    rest_yaw_deg=90.0,
    wrist_turn_deg=100.0,
    park=YAM_PARK,
    rise_margin=0.03,
)
def banana_in_bowl_expert(
    license: str,
    grip_point: Sequence[float],
    reach: float,
    rest_yaw_deg: float,
    wrist_turn_deg: float,
    park: Sequence[float],
    rise_margin: float,
) -> MakeTeleoperator:
    """The scripted expert of ``BananaInBowlTask`` on the YAM, which states the license ``license``. Each arm has the
    statics of ``ReachStatics``: ``grip_point`` (a position from ``GRIPPER_SITE``), ``reach``, ``park``,
    ``rise_margin``, and the rest yaw and the wrist turn in degrees."""
    return partial(
        BananaInBowlExpert.for_yam,
        license,
        grip_point=geom.Transform3D(np.asarray(grip_point, dtype=np.float64)),
        reach=reach,
        rest_yaw=math.radians(rest_yaw_deg),
        wrist_turn=math.radians(wrist_turn_deg),
        park=np.asarray(park, dtype=np.float64),
        rise_margin=rise_margin,
    )


# DAgger rounds of ``BananaInBowlTask`` on the two-arm YAM in the RoboLab env server, with LeRobot ACT.
robolab_banana_in_bowl = run.override(
    rig=robolab_rig.override(teleoperator=banana_in_bowl_expert),
    tasks=banana_in_bowl,
    model=lerobot_act.override(codec='ee_bimanual', arms=keys.BIMANUAL_ARMS),
    rounds=3,
    episodes=20,
)
