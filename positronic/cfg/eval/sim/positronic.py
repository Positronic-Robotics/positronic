import configuronic as cfn

import positronic.cfg.simulator
from positronic.cfg.embodiment import mujoco_franka
from positronic.eval import Eval, Observation, Task
from positronic.simulator.mujoco.sim import FullSimState, MujocoSim
from positronic.utils import package_assets_path


@cfn.config(
    mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
    camera_fps=15,
    camera_dict={'image.wrist': 'handcam_left_ph', 'image.exterior': 'back_view_ph', 'image.agent_view': 'agentview'},
    # Full sim state is the privileged ground truth; scoring is computed downstream.
    observers={'sim_state': FullSimState()},
    timeout=15,
    seed=None,
)
def _mujoco_franka_eval(mujoco_model_path, loaders, camera_fps, camera_dict, observers, instruction, timeout, seed):
    """A Mujoco Franka sim eval: the eval holds the sim, the embodiment is pure robot.

    The task carries the instruction, the per-trial ``timeout``, the privileged sim-state
    ground truth (built from the eval's sim, recorded but never fed to the policy), and the
    sim's seeded scene reset. The scene (``loaders``) is embodiment-specific and wired here,
    not a generic Task field; the loaders carry no seeds of their own — the per-trial seed
    handed to ``sim.reset`` drives the whole scene draw.
    """
    sim = MujocoSim(mujoco_model_path, loaders, observers=observers, camera_fps=camera_fps)
    embodiment = mujoco_franka(sim, camera_dict)
    privileged = {name: Observation(sim.observations[name], None) for name in observers}
    task = Task(instruction=instruction, timeout=timeout, privileged=privileged, seed=seed, reset=sim.reset)
    return Eval(embodiment, task)


stack_cubes = _mujoco_franka_eval.override(instruction='Pick up the green cube and place it on the red cube.')

multi_tote = _mujoco_franka_eval.override(
    loaders=positronic.cfg.simulator.multi_tote_loaders,
    observers={},
    instruction='Pick up objects from the red tote and place them in the green tote.',
)
