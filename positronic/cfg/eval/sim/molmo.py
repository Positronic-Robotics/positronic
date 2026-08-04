import json
from pathlib import Path

import configuronic as cfn

from positronic import keys
from positronic.drivers.roboarm.models import GRASP_SITE_LINK, bundled_franka_model
from positronic.eval import EVAL_EPISODE_INDEX, EVAL_SEED, EVAL_TRIAL_COUNT, EVAL_TRIAL_INDEX, Eval, Observation, Task
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_franka_embodiment
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import MolmoAdapter
from positronic.simulator.molmo_spaces.launcher import serve_molmo_spaces


def _episode_count(benchmark_dir: Path) -> int:
    """The episode count of a MolmoSpaces benchmark dir, mirroring ``load_all_episodes``' two layouts.

    positronic cannot import ``molmo_spaces`` here (it lives in the env server's own venv), so this counts the
    benchmark files directly: a single ``benchmark.json`` (a JSON list of episode specs) when present, else the
    legacy ``house_*/episode_*.json`` layout the loader also accepts.
    """
    manifest = benchmark_dir / 'benchmark.json'
    if manifest.exists():
        return len(json.loads(manifest.read_text()))
    return sum(1 for _ in benchmark_dir.glob('house_*/episode_*.json'))


@cfn.config(
    camera_dict={keys.WRIST_IMAGE: mapping.MOLMO_WRIST_CAMERA, keys.EXTERIOR_IMAGE: mapping.MOLMO_EXTERIOR_CAMERA},
    benchmark_dir=None,
    episodes=None,
    trial_count=1,
    timeout=60.0,
    seed=None,
    task_horizon_steps=None,
)
def _molmo_eval(
    benchmark_dir: str | None,
    episodes: int | list[int] | None,
    trial_count: int,
    timeout: float,
    camera_dict: dict[str, str],
    seed: int | None,
    task_horizon_steps: int | None,
) -> Eval:
    """A MolmoSpaces eval: the embodiment proxies a remote MolmoSpaces env, the task carries the scenario.

    MolmoSpaces (https://github.com/allenai/molmospaces) is AllenAI's MuJoCo manipulation benchmark on the DROID
    rig (Franka arm + Robotiq 2F-85) across ProcTHOR scenes; a benchmark is a directory holding a ``benchmark.json``
    (a JSON list of episode specs — house, task, exact object poses, cameras, language goal), so
    ``--eval.benchmark_dir`` names that directory and ``--eval.episodes`` optionally pins a subset of episode
    indices (default: the whole benchmark). The asset packs live under ``MLSPACES_ASSETS_DIR``.

    positronic launches a single task-agnostic env server in MolmoSpaces' own interpreter; the proxy drives it
    over the socket and the episode index rides each trial's reset token, so one embodiment serves every episode.
    The instruction is never pinned: the task reads its language live from the env, which reports the episode's
    resolved goal in every reset's meta. Episodes are exact-pose deterministic, so ``trial_count`` defaults to 1.

    ``timeout`` is not the benchmark horizon — the sim owns that (the benchmark's ``task_horizon_sec``, enforced
    env-side and delivered as a terminal ``done``). It is only a runaway-cost safety net for a sim that never
    terminates, so it must stay longer than the sim's native horizon; the env reports that horizon at reset and
    the harness rejects a ``timeout`` that isn't strictly weaker, so a too-short budget fails loud instead of
    silently truncating a valid episode. Being sim-time, the spare budget costs nothing unless the sim misbehaves.

    ``task_horizon_steps`` optionally pins the episode horizon, mirroring MolmoSpaces' ``--task_horizon_steps`` —
    use it to reproduce a reference run whose horizon differs from the benchmark's declared value. Default
    (``None``) reads the benchmark's own ``task_horizon_sec`` (DROID Pick = 20 s -> 303 steps). Discrepancy worth
    knowing: MolmoSpaces' shipped benchmarks carry ``task_horizon_sec`` at the episode level, but its own
    ``determine_task_horizon`` reads only the task dict and so raises on them — a native run needs this override
    (or a ``patch_benchmarks`` pass, which defaults PickTask to 20 s and a later patch bumps it to 30 s).
    """
    if benchmark_dir is None:
        raise ValueError('MolmoSpaces eval needs --eval.benchmark_dir pointing at a dir with benchmark.json')
    base = Path(benchmark_dir)
    count = _episode_count(base)
    if episodes is None:
        indices = list(range(count))
    else:
        indices = [episodes] if isinstance(episodes, int) else list(episodes)
    if not indices:
        raise ValueError(
            f'no benchmark episodes found under {base}; expected a benchmark.json or a legacy '
            'house_*/episode_*.json layout (or pass --eval.episodes explicitly)'
        )
    # Reject explicit selectors outside the manifest before the costly server spawn: a negative index would
    # silently run a from-the-end episode mislabeled by its own index, and an over-range one fails only after setup.
    out_of_range = [i for i in indices if not 0 <= i < count]
    if out_of_range:
        raise ValueError(f'--eval.episodes {out_of_range} out of range for the {count} episodes under {base}')
    # A non-positive count yields no trials at all, and an empty plan reads to the self-driving harness as a
    # finished run — the command would exit 0 having evaluated nothing.
    if trial_count < 1:
        raise ValueError(f'--eval.trial_count must be at least 1, got {trial_count}')
    proxy = RemoteEnvControlSystem(
        MolmoAdapter(camera_dict), serve_molmo_spaces(base, task_horizon_steps=task_horizon_steps)
    )
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
    task = Task(
        instruction=lambda: proxy.meta['task'],
        timeout=timeout,
        privileged=privileged,
        reset=proxy.reset,
        done=proxy.done,
        horizon=lambda: proxy.horizon,
    )
    # Benchmark episodes are exact-pose deterministic and carry their own seed. An unset ``seed`` leaves
    # ``eval.seed`` off the trial, so the env falls back to the episode's spec seed (reproducing the benchmark);
    # an explicit ``seed`` overrides it, sweeping ``seed .. seed + trial_count - 1``. (``build_trials`` injects a
    # random seed when ``seed`` is None, which would clobber the spec seed and make the run non-reproducible.)
    trials = [
        {EVAL_EPISODE_INDEX: i, **({EVAL_SEED: seed + t} if seed is not None else {})}
        for i in indices
        for t in range(trial_count)
    ]
    for j, ctx in enumerate(trials):
        ctx.update({EVAL_TRIAL_INDEX: j, EVAL_TRIAL_COUNT: len(trials)})
    return Eval(embodiment, task, trials)


# The whole benchmark in one run (every episode in ``--eval.benchmark_dir``'s benchmark.json).
benchmark = _molmo_eval

# A single-episode smoke target: the first episode of the benchmark.
first_episode = _molmo_eval.override(episodes=0)
