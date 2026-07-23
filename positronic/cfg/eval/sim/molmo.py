import json
from pathlib import Path

import configuronic as cfn

from positronic.drivers.roboarm.models import bundled_franka_model
from positronic.eval import Eval, Task
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_franka_embodiment
from positronic.simulator.molmo_spaces.adapter import MolmoAdapter
from positronic.simulator.molmo_spaces.launcher import serve_molmo_spaces


def _episode_count(benchmark_dir: str) -> int:
    """The number of episodes in a MolmoSpaces ``benchmark.json`` (a JSON list of episode specs)."""
    return len(json.loads((Path(benchmark_dir) / 'benchmark.json').read_text()))


@cfn.config(
    camera_dict={'image.wrist': 'wrist_camera', 'image.exterior': 'exo_camera_1'},
    benchmark_dir=None,
    episodes=None,
    trial_count=1,
    timeout=60.0,
    seed=None,
)
def _molmo_eval(benchmark_dir, episodes, trial_count, timeout, camera_dict, seed):
    """A MolmoSpaces eval: the embodiment proxies a remote MolmoSpaces env, the task carries the scenario.

    MolmoSpaces (https://github.com/allenai/molmospaces) is AllenAI's MuJoCo manipulation benchmark on the DROID
    rig (Franka arm + Robotiq 2F-85) across ProcTHOR scenes; a benchmark is a ``benchmark.json`` of episode specs
    (house, task, exact object poses, cameras, language goal), so ``--eval.benchmark_dir`` names the benchmark to
    run and ``--eval.episodes`` optionally pins a subset of episode indices (default: the whole benchmark). The
    asset packs live under ``MLSPACES_ASSETS_DIR``.

    positronic launches a single task-agnostic env server in MolmoSpaces' own interpreter; the proxy drives it
    over the socket and the episode index rides each trial's reset token, so one embodiment serves every episode.
    The instruction is never pinned: the task reads its language live from the env, which reports the episode's
    resolved goal in every reset's meta. Episodes are exact-pose deterministic, so ``trial_count`` defaults to 1.
    """
    if benchmark_dir is None:
        raise ValueError('MolmoSpaces eval needs --eval.benchmark_dir pointing at a dir with benchmark.json')
    if episodes is None:
        indices = list(range(_episode_count(benchmark_dir)))
    else:
        indices = [episodes] if isinstance(episodes, int) else list(episodes)
    proxy = RemoteEnvControlSystem(MolmoAdapter(camera_dict), serve_molmo_spaces(benchmark_dir))
    # MolmoSpaces drives a Franka DROID rig; recordings carry the same model (URDF + meshes + joint names +
    # control frame) for the 3D viewer and offline IK, supplied here since the molmo server can't import
    # positronic to emit it via ``robot_meta``.
    embodiment = remote_franka_embodiment(
        proxy, camera_dict, descriptor='remote.molmo_spaces.droid', static_meta=bundled_franka_model()
    )
    task = Task(instruction=lambda: proxy.meta['task'], timeout=timeout, reset=proxy.reset, done=proxy.done)
    # Benchmark episodes are exact-pose deterministic and carry their own seed. An unset ``seed`` leaves
    # ``eval.seed`` off the trial, so the env falls back to the episode's spec seed (reproducing the benchmark);
    # an explicit ``seed`` overrides it, sweeping ``seed .. seed + trial_count - 1``. (``build_trials`` injects a
    # random seed when ``seed`` is None, which would clobber the spec seed and make the run non-reproducible.)
    trials = [
        {'eval.episode_index': i, **({'eval.seed': seed + t} if seed is not None else {})}
        for i in indices
        for t in range(trial_count)
    ]
    for j, ctx in enumerate(trials):
        ctx.update({'eval.trial_index': j, 'eval.trial_count': len(trials)})
    return Eval(embodiment, task, trials)


# The whole benchmark in one run (every episode in ``--eval.benchmark_dir``'s benchmark.json).
benchmark = _molmo_eval

# A single-episode smoke target: the first episode of the benchmark.
first_episode = _molmo_eval.override(episodes=0)
