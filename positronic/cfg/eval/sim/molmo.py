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
_TIMEOUT_MARGIN_SEC = 1.0


# !!!!!!!!!! Where does benchmark_dir come from


@cfn.config(camera_dict=DEFAULT_CAMERA_DICT, episodes=None, trial_count=1, timeout=None, seed=None)
def benchmark(
    benchmark_dir: str,
    episodes: int | list[int] | None,
    trial_count: int,
    timeout: float | None,
    camera_dict: dict[str, str],
    seed: int | None,
) -> Eval:
    """A MolmoSpaces eval: the embodiment proxies a remote MolmoSpaces env, the task carries the scenario.

    MolmoSpaces (https://github.com/allenai/molmospaces) is AllenAI's MuJoCo manipulation benchmark on the DROID
    rig (Franka arm + Robotiq 2F-85) across ProcTHOR scenes. A benchmark is a directory holding a ``benchmark.json`` -
    a JSON list of episode specs: house, task, exact object poses, cameras, language goal. Hence
    ``--eval.benchmark_dir`` names that directory and ``--eval.episodes`` optionally pins a subset of episode
    indices (default: the whole benchmark).

    positronic launches a single task-agnostic env server in MolmoSpaces' own subprocess. The proxy controls it
    over the socket, the env answers which episodes the sweep runs, and the episode index rides each trial's reset
    token. Every reset's meta reports the instruction (aka prompt).

    The benchmark's ``task_horizon_sec``, enforced on env-side and delivered as a terminal ``done`` signal.
    By default, ``timeout`` is the benchmark horizon plus a margin.
    """
    if trial_count < 1:
        raise ValueError(f'--eval.trial_count must be at least 1, got {trial_count}')
    proxy = RemoteEnvControlSystem(MolmoAdapter(camera_dict), serve_molmo_spaces(Path(benchmark_dir)))
    # MolmoSpaces drives a Franka DROID rig.
    embodiment = remote_franka_embodiment(
        proxy, camera_dict, descriptor='remote.molmo_spaces.droid', static_meta=bundled_franka_model(GRASP_SITE_LINK)
    )
    privileged = {mapping.OBS_SIM_STATE: Observation(proxy.privileged[mapping.OBS_SIM_STATE], None)}

    def tasks() -> list[Task]:
        params = proxy.tasks(spec(episodes=episodes))
        # The benchmark declares one horizon over all its episodes (the env refuses an inconsistent one), so one
        # backstop deadline covers the run.
        deadline = params[0][molmo_keys.TASK_HORIZON] + _TIMEOUT_MARGIN_SEC
        if timeout is not None:
            logging.warning('--eval.timeout %ss overrides the benchmark backstop of %ss', timeout, deadline)
            deadline = timeout
        task = Task(instruction_source=lambda: proxy.meta[mapping.META_TASK], timeout_sec=deadline)
        # ``seed`` of None means using default seed used by the benchmark.
        return number_trials([
            (task, {**p, **({eval_keys.SEED: seed + t} if seed is not None else {})})
            for p in params
            for t in range(trial_count)
        ])

    return Eval(embodiment, tasks, privileged=privileged, done=proxy.done)


# A single-episode smoke target: the first episode of the benchmark.
first_episode = benchmark.override(episodes=0)
