import logging
from pathlib import Path

import configuronic as cfn

from positronic.cfg.eval import number_trials, spec
from positronic.drivers.roboarm.models import GRASP_SITE_LINK, bundled_franka_model
from positronic.eval import Eval, Observation, Task
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_franka_embodiment
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import DEFAULT_CAMERA_DICT, MolmoAdapter
from positronic.simulator.molmo_spaces.launcher import serve_molmo_spaces

# How far the harness deadline sits above the benchmark horizon. Being sim-time, the spare budget costs
# nothing unless the sim stops terminating, which is the only thing the deadline is there to catch.
_TIMEOUT_MARGIN_SEC = 10.0


@cfn.config(camera_dict=DEFAULT_CAMERA_DICT, episodes=None, trial_count=1, timeout=None, seed=None)
def _molmo_eval(
    benchmark_dir: str,
    episodes: int | list[int] | None,
    trial_count: int,
    timeout: float | None,
    camera_dict: dict[str, str],
    seed: int | None,
) -> Eval:
    """A MolmoSpaces eval: the embodiment proxies a remote MolmoSpaces env, the task carries the scenario.

    MolmoSpaces (https://github.com/allenai/molmospaces) is AllenAI's MuJoCo manipulation benchmark on the DROID
    rig (Franka arm + Robotiq 2F-85) across ProcTHOR scenes; a benchmark is a directory holding a ``benchmark.json``
    (a JSON list of episode specs — house, task, exact object poses, cameras, language goal), so
    ``--eval.benchmark_dir`` names that directory and ``--eval.episodes`` optionally pins a subset of episode
    indices (default: the whole benchmark). The asset packs live under ``MLSPACES_ASSETS_DIR``.

    positronic launches a single task-agnostic env server in MolmoSpaces' own interpreter; the proxy drives it
    over the socket, the env answers which episodes the sweep runs, and the episode index rides each trial's reset
    token. The instruction is never pinned: the task reads its language live from the env, which reports the
    episode's resolved goal in every reset's meta. Episodes are exact-pose deterministic, so ``trial_count``
    defaults to 1.

    ``timeout`` is not the benchmark horizon — the sim owns that (the benchmark's ``task_horizon_sec``, enforced
    env-side and delivered as a terminal ``done``). It is only a runaway-cost safety net for a sim that never
    terminates, so its default is the benchmark's own horizon plus a margin. An explicit value can only lower the
    deadline, never raise it, and one at or below the horizon truncates valid episodes — so any value that
    differs from the default is warned about.
    """
    # A non-positive count yields no trials at all, and an empty plan reads to the self-driving harness as a
    # finished run — the command would exit 0 having evaluated nothing.
    if trial_count < 1:
        raise ValueError(f'--eval.trial_count must be at least 1, got {trial_count}')
    proxy = RemoteEnvControlSystem(MolmoAdapter(camera_dict), serve_molmo_spaces(Path(benchmark_dir)))
    # MolmoSpaces drives a Franka DROID rig; recordings carry the same model (URDF + meshes + joint names +
    # control frame) for the 3D viewer and offline IK, supplied here since the molmo server can't import
    # positronic to emit it via ``robot_meta``. ``DEFAULT_FRAME`` is declared on the gripper's grasp site,
    # which is where ``env.py`` reports ``robot_state.ee_pose`` and resolves Cartesian targets, so a policy
    # frame reached from it via ``ChangeEEFrame`` and offline IK over a recording both anchor correctly.
    embodiment = remote_franka_embodiment(
        proxy, camera_dict, descriptor='remote.molmo_spaces.droid', static_meta=bundled_franka_model(GRASP_SITE_LINK)
    )
    # The env's full MuJoCo state is recorded as privileged ground truth, never fed to the policy.
    privileged = {mapping.OBS_SIM_STATE: Observation(proxy.privileged[mapping.OBS_SIM_STATE], None)}

    def tasks() -> list[Task]:
        params = proxy.tasks(spec(episodes=episodes))
        # The benchmark declares one horizon over all its episodes (the env refuses an inconsistent one), so one
        # backstop deadline covers the run.
        backstop = params[0][molmo_keys.TASK_HORIZON] + _TIMEOUT_MARGIN_SEC
        if timeout is not None and timeout != backstop:
            logging.warning(
                '--eval.timeout %ss overrides the benchmark backstop of %ss (the %ss horizon plus a margin); '
                'running with %ss. The deadline only catches a sim that stopped terminating, and a deadline at '
                'or below the horizon cuts valid episodes short and scores them as failures.',
                timeout,
                backstop,
                params[0][molmo_keys.TASK_HORIZON],
                min(timeout, backstop),
            )
        deadline = backstop if timeout is None else min(timeout, backstop)
        task = Task(instruction_source=lambda: proxy.meta[mapping.META_TASK], timeout_sec=deadline)
        # Benchmark episodes are exact-pose deterministic and carry their own seed. An unset ``seed`` leaves
        # ``eval.seed`` off the trial, so the env falls back to the episode's spec seed (reproducing the
        # benchmark); an explicit ``seed`` overrides it, sweeping ``seed .. seed + trial_count - 1``.
        return number_trials([
            (task, {**p, **({eval_keys.SEED: seed + t} if seed is not None else {})})
            for p in params
            for t in range(trial_count)
        ])

    return Eval(embodiment, tasks, privileged=privileged, done=proxy.done)


# The whole benchmark in one run (every episode in ``--eval.benchmark_dir``'s benchmark.json).
benchmark = _molmo_eval

# A single-episode smoke target: the first episode of the benchmark.
first_episode = _molmo_eval.override(episodes=0)
