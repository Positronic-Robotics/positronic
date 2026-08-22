import configuronic as cfn

import positronic.cfg.simulator
from positronic import keys
from positronic.cfg.embodiment import mujoco_franka
from positronic.cfg.eval import build_tasks
from positronic.eval import Eval, Observation, Task
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
    """A Mujoco Franka sim eval: the sim is the embodiment's, the eval holds what a trial runs.

    The task carries the instruction and the per-trial ``timeout``; the eval carries the privileged
    sim-state ground truth, recorded but never fed to the policy. The scene (``loaders``) is
    embodiment-specific and wired here, not a generic Task field; the loaders carry no seeds of their
    own — the per-trial seed the scene prepare is asked with drives the whole scene draw. ``trial_count``
    seeds (from ``seed``) make the trial sweep; this eval has no task axis, so each is a fresh scene draw.
    """
    sim = MujocoSim(mujoco_model_path, loaders, camera_fps=camera_fps)
    embodiment = mujoco_franka(sim, camera_dict)
    # Full sim state is the privileged ground truth; scoring is computed downstream.
    privileged = {'sim_state': Observation(sim.sim_state, None)}
    return Eval(
        embodiment,
        build_tasks(Task(instruction_source=instruction, timeout_sec=timeout), seed, trial_count),
        privileged=privileged,
    )


stack_cubes = _mujoco_franka_eval.override(instruction='Pick up the green cube and place it on the red cube.')

multi_tote = _mujoco_franka_eval.override(
    loaders=positronic.cfg.simulator.multi_tote_loaders,
    instruction='Pick up objects from the red tote and place them in the green tote.',
)
