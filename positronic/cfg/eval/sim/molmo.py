import json
import logging
import os
from pathlib import Path
from typing import Any

import configuronic as cfn

from positronic import keys
from positronic.cfg.eval import number_trials
from positronic.drivers.roboarm.models import GRASP_SITE_LINK, bundled_franka_model
from positronic.eval import Eval, Observation, Task
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_franka_embodiment
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import DEFAULT_CAMERA_DICT, MolmoAdapter
from positronic.simulator.molmo_spaces.launcher import serve_molmo_spaces

# How far the harness deadline sits above the benchmark horizon. Being sim-time, the spare budget costs
# nothing unless the sim stops terminating, which is the only thing the deadline is there to catch.
_TIMEOUT_MARGIN_SEC = 10.0


def _load_episodes(benchmark_dir: Path) -> list[dict[str, Any]]:
    """The episode specs of a MolmoSpaces benchmark dir, mirroring ``load_all_episodes``' two layouts.

    positronic cannot import ``molmo_spaces`` here (it lives in the env server's own venv), so this reads the
    benchmark files directly: a single ``benchmark.json`` (a JSON list of episode specs) when present, else the
    legacy ``house_*/episode_*.json`` layout the loader also accepts.
    """
    manifest = benchmark_dir / mapping.MOLMO_BENCHMARK_MANIFEST
    if manifest.exists():
        return json.loads(manifest.read_text())
    return [json.loads(p.read_text()) for p in sorted(benchmark_dir.glob('house_*/episode_*.json'))]


def _discovery_hint() -> str:
    """The benchmark dirs found under ``MLSPACES_ASSETS_DIR``, appended to a path that holds none."""
    assets = os.environ.get(mapping.ASSETS_DIR_ENV)
    if not assets:
        return f' Point {mapping.ASSETS_DIR_ENV} at the MolmoSpaces asset packs to have the available ones listed.'
    root = Path(assets) / mapping.ASSETS_BENCHMARKS_DIR
    found = sorted(str(p.parent) for p in root.rglob(mapping.MOLMO_BENCHMARK_MANIFEST))
    if not found:
        return f' No {mapping.MOLMO_BENCHMARK_MANIFEST} found under {root}.'
    return f' Available under {root}: {", ".join(found)}'


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
    over the socket and the episode index rides each trial's reset token, so one embodiment serves every episode.
    The instruction is never pinned: the task reads its language live from the env, which reports the episode's
    resolved goal in every reset's meta. Episodes are exact-pose deterministic, so ``trial_count`` defaults to 1.

    ``timeout`` is not the benchmark horizon — the sim owns that (the benchmark's ``task_horizon_sec``, enforced
    env-side and delivered as a terminal ``done``). It is only a runaway-cost safety net for a sim that never
    terminates, so its default is the benchmark's own horizon plus a margin. An explicit value can only lower the
    deadline, never raise it, and one at or below the horizon truncates valid episodes — so any value that
    differs from the default is warned about.
    """
    base = Path(benchmark_dir)
    specs = _load_episodes(base)
    count = len(specs)
    if count == 0:
        raise ValueError(
            f'no benchmark episodes under {base}; expected a {mapping.MOLMO_BENCHMARK_MANIFEST} or a legacy '
            f'house_*/episode_*.json layout.{_discovery_hint()}'
        )
    indices = list(range(count)) if episodes is None else [episodes] if isinstance(episodes, int) else list(episodes)
    if not indices:
        raise ValueError('--eval.episodes selected no episodes; omit it to run the whole benchmark')
    # Reject explicit selectors outside the manifest before the costly server spawn: a negative index would
    # silently run a from-the-end episode mislabeled by its own index, and an over-range one fails only after setup.
    out_of_range = [i for i in indices if not 0 <= i < count]
    if out_of_range:
        raise ValueError(f'--eval.episodes {out_of_range} out of range for the {count} episodes under {base}')
    # A non-positive count yields no trials at all, and an empty plan reads to the self-driving harness as a
    # finished run — the command would exit 0 having evaluated nothing.
    if trial_count < 1:
        raise ValueError(f'--eval.trial_count must be at least 1, got {trial_count}')
    # Before the costly server spawn, and the same rule the env server then applies to the same specs.
    horizon = mapping.declared_task_horizon_sec(
        spec.get(mapping.MOLMO_EPISODE_TASK, {}).get(mapping.MOLMO_TASK_HORIZON_SEC) for spec in specs
    )
    backstop = horizon + _TIMEOUT_MARGIN_SEC
    if timeout is not None and timeout != backstop:
        logging.warning(
            '--eval.timeout %ss overrides the benchmark backstop of %ss (the %ss horizon plus a margin); running '
            'with %ss. The deadline only catches a sim that stopped terminating, and a deadline at or below the '
            'horizon cuts valid episodes short and scores them as failures.',
            timeout,
            backstop,
            horizon,
            min(timeout, backstop),
        )
    timeout = backstop if timeout is None else min(timeout, backstop)
    proxy = RemoteEnvControlSystem(MolmoAdapter(camera_dict), serve_molmo_spaces(base))
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
    task = Task(instruction_source=lambda: proxy.meta[mapping.META_TASK], timeout_sec=timeout)
    # Benchmark episodes are exact-pose deterministic and carry their own seed. An unset ``seed`` leaves
    # ``eval.seed`` off the trial, so the env falls back to the episode's spec seed (reproducing the benchmark);
    # an explicit ``seed`` overrides it, sweeping ``seed .. seed + trial_count - 1``. (``build_tasks`` injects a
    # random seed when ``seed`` is None, which would clobber the spec seed and make the run non-reproducible.)
    params = [
        {keys.EVAL_EPISODE_INDEX: i, **({keys.EVAL_SEED: seed + t} if seed is not None else {})}
        for i in indices
        for t in range(trial_count)
    ]
    return Eval(embodiment, number_trials(task, params), privileged=privileged, reset=proxy.reset, done=proxy.done)


# The whole benchmark in one run (every episode in ``--eval.benchmark_dir``'s benchmark.json).
benchmark = _molmo_eval

# A single-episode smoke target: the first episode of the benchmark.
first_episode = _molmo_eval.override(episodes=0)
