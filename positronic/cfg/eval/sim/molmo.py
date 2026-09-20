import logging
import math

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

# Margin for the harness to observe the simulator's terminal signal.
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
    """A DROID evaluation over selected MolmoSpaces benchmark episodes.

    Each benchmark dimension accepts a name, a list or None (all found); ``episodes`` accepts indices.
    ``seed=None`` follows the benchmark's seed convention. The default timeout is the benchmark horizon
    plus a margin; an explicit timeout replaces it.
    """
    if trial_count < 1:
        raise ValueError(f'--eval.trial_count must be at least 1, got {trial_count}')
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError(f'--eval.timeout must be finite and positive, got {timeout}')
    proxy = RemoteEnvControlSystem(MolmoAdapter(), serve_molmo_spaces())
    embodiment = remote_franka_embodiment(
        proxy, CAMERAS, descriptor='remote.molmo_spaces.droid', static_meta=bundled_franka_model(GRASP_SITE_LINK)
    )
    privileged = {mapping.OBS_SIM_STATE: Observation(proxy.privileged[mapping.OBS_SIM_STATE], None)}

    def tasks() -> list[Task]:
        selection = spec(
            suite=suite,
            scene_dataset=scene_dataset,
            task_config=task_config,
            benchmark=benchmark,
            **{mapping.SELECT_EPISODES: episodes},
        )
        if timeout is not None:
            logging.warning('--eval.timeout %ss replaces the benchmark horizon backstop', timeout)
        trials = []
        for params in proxy.tasks(selection):
            deadline = timeout if timeout is not None else params[molmo_keys.TASK_HORIZON] + _TIMEOUT_MARGIN_SEC
            task = Task(instruction_source=lambda: proxy.meta[mapping.META_TASK], timeout_sec=deadline)
            trials += [
                (task, {**params, **({eval_keys.SEED: seed + t} if seed is not None else {})})
                for t in range(trial_count)
            ]
        return number_trials(trials)

    return Eval(embodiment, tasks, privileged=privileged, done=proxy.done)


bench_v1 = benchmarks.override(suite='molmospaces-bench-v1')
bench_v2 = benchmarks.override(suite='molmospaces-bench-v2')

pick_v1 = bench_v1.override(task_config='FrankaPickDroidMiniBench')
first_episode = pick_v1.override(episodes=0)
