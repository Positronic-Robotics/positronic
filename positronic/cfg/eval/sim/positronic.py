import configuronic as cfn

import positronic.cfg.simulator
from positronic import keys
from positronic.cfg.embodiment import mujoco_franka
from positronic.cfg.eval import build_tasks
from positronic.eval import Eval, Observation
from positronic.simulator.mujoco.sim import MujocoSim
from positronic.utils import package_assets_path


@cfn.config(
    mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
    camera_fps=15,
    camera_dict={
        keys.WRIST_IMAGE: 'handcam_left_ph',
        keys.EXTERIOR_IMAGE: 'back_view_ph',
        'image.agent_view': 'agentview',
    },
    timeout=15,
    seed=None,
    trial_count=1,
)
def _mujoco_franka_eval(mujoco_model_path, loaders, camera_fps, camera_dict, instruction, timeout, seed, trial_count):
    """A Mujoco Franka sim eval: the eval holds the sim, the embodiment is pure robot.

    Every rollout carries the instruction and the ``timeout``; the eval carries the privileged sim-state
    ground truth (built from its sim, recorded but never fed to the policy) and the sim's seeded scene
    reset. The scene shape (``loaders``) is embodiment-specific and wired here, not a per-rollout field;
    the loaders carry no seeds of their own — the seed in each rollout's scene, handed to ``sim.reset``,
    drives the whole scene draw. ``trial_count`` seeds (from ``seed``) make the sweep; this eval has no
    task axis, so each rollout is a fresh scene draw.
    """
    sim = MujocoSim(mujoco_model_path, loaders, camera_fps=camera_fps)
    embodiment = mujoco_franka(sim, camera_dict)
    return [
        Eval(
            embodiment,
            build_tasks(instruction, timeout, seed, trial_count),
            reset=lambda scene: sim.reset(scene.get('eval.seed')),
            # Full sim state is the privileged ground truth; scoring is computed downstream.
            privileged={'sim_state': Observation(sim.sim_state, None)},
        )
    ]


stack_cubes = _mujoco_franka_eval.override(instruction='Pick up the green cube and place it on the red cube.')

multi_tote = _mujoco_franka_eval.override(
    loaders=positronic.cfg.simulator.multi_tote_loaders,
    instruction='Pick up objects from the red tote and place them in the green tote.',
)
