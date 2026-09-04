import logging

import configuronic as cfn

from positronic.cfg.eval import number_trials, spec
from positronic.drivers.roboarm.models import GRASP_SITE_LINK, bundled_franka_model
from positronic.eval import Eval, Observation, Task
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_franka_embodiment
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import CAMERAS, MolmoAdapter
from positronic.simulator.molmo_spaces.launcher import serve_molmo_spaces

# How far the harness deadline sits above the benchmark horizon. Being sim-time, the spare budget costs
# nothing unless the sim stops terminating, which is the only thing the deadline is there to catch.
_TIMEOUT_MARGIN_SEC = 1.0


@cfn.config(
    suite=None,
    scene_dataset=None,
    task_config=None,
    benchmark=None,
    episodes=None,
    trial_count=1,
    timeout=None,
    seed=None,
)
def benchmarks(
    suite: str | list[str] | None,
    scene_dataset: str | list[str] | None,
    task_config: str | list[str] | None,
    benchmark: str | list[str] | None,
    episodes: int | list[int] | None,
    trial_count: int,
    timeout: float | None,
    seed: int | None,
) -> Eval:
    """A MolmoSpaces eval: the embodiment proxies a remote MolmoSpaces env, the task carries the scenario.

    MolmoSpaces (https://github.com/allenai/molmospaces) is AllenAI's MuJoCo manipulation benchmark on the DROID
    rig (Franka arm + Robotiq 2F-85) across ProcTHOR scenes. Its asset packs (``MLSPACES_ASSETS_DIR``) hold the
    benchmarks as ``benchmarks/<suite>/<scene_dataset>/<task_config>/<benchmark>/benchmark.json``, a JSON list
    of episode specs: house, task, exact object poses, cameras, language goal. The four dimensions select the
    benchmarks a run sweeps and ``episodes`` the episodes within each; every one is a single value, a list, or
    unbound (all found). The env resolves the selection when the run starts.

    positronic launches a single benchmark-agnostic env server in MolmoSpaces' own subprocess. The proxy controls
    it over the socket, the env answers which episodes the sweep runs, and the benchmark and episode index ride
    each trial's reset token. Every reset's meta reports the instruction (aka prompt).

    Each benchmark's ``task_horizon_sec`` is enforced on env-side and delivered as a terminal ``done`` signal.
    By default, a trial's ``timeout`` is its benchmark's horizon plus a margin; an explicit one replaces it.
    """
    if trial_count < 1:
        raise ValueError(f'--eval.trial_count must be at least 1, got {trial_count}')
    proxy = RemoteEnvControlSystem(MolmoAdapter(), serve_molmo_spaces())
    # MolmoSpaces drives a Franka DROID rig.
    embodiment = remote_franka_embodiment(
        proxy, CAMERAS, descriptor='remote.molmo_spaces.droid', static_meta=bundled_franka_model(GRASP_SITE_LINK)
    )
    privileged = {mapping.OBS_SIM_STATE: Observation(proxy.privileged[mapping.OBS_SIM_STATE], None)}

    def tasks() -> list[Task]:
        selection = spec(
            suite=suite, scene_dataset=scene_dataset, task_config=task_config, benchmark=benchmark, episodes=episodes
        )
        if timeout is not None:
            logging.warning('--eval.timeout %ss replaces the benchmark horizon backstop', timeout)
        trials = []
        for params in proxy.tasks(selection):
            deadline = timeout if timeout is not None else params[molmo_keys.TASK_HORIZON] + _TIMEOUT_MARGIN_SEC
            task = Task(instruction_source=lambda: proxy.meta[mapping.META_TASK], timeout_sec=deadline)
            # ``seed`` of None means using default seed used by the benchmark.
            trials += [
                (task, {**params, **({eval_keys.SEED: seed + t} if seed is not None else {})})
                for t in range(trial_count)
            ]
        return number_trials(trials)

    return Eval(embodiment, tasks, privileged=privileged, done=proxy.done)


# The two suites MolmoSpaces documents (``molmo_spaces/evaluation/ms-bench.md`` and ``mb-bench.md``).
bench_v1 = benchmarks.override(suite='molmospaces-bench-v1')
bench_v2 = benchmarks.override(suite='molmospaces-bench-v2')

# The v1 pick benchmark, and its first episode as a smoke target.
pick_v1 = bench_v1.override(task_config='FrankaPickDroidMiniBench')
first_episode = pick_v1.override(episodes=0)
