import logging
import math
import random

import configuronic as cfn

from positronic.cfg.eval import number_trials, spec
from positronic.eval import Eval, Task
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem, remote_embodiment
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import MolmoAdapter, ShoulderSide, camera_dict
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
    with_right_shoulder=True,
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
    with_right_shoulder: bool,
) -> Eval:
    """A DROID evaluation over selected MolmoSpaces benchmark episodes.

    Each benchmark dimension accepts a name, a list or None (all found); ``episodes`` accepts indices.
    ``seed=None`` follows the benchmark's seed convention. ``timeout`` replaces the benchmark horizon in the simulator
    and in the harness; the harness deadline is the horizon plus a margin.

    The policy sees the wrist camera and two shoulder cameras, left and right, in ``image.exterior`` and
    ``image.exterior_2``. The run draws their order once from ``seed``, or at random when ``seed`` is None,
    and every trial records it under ``molmo.exterior_sides``. The right shoulder camera hides the grasp site
    in some episodes; an open selection leaves those out, and a pinned one is refused.

    ``with_right_shoulder=False`` serves the left shoulder camera alone, in ``image.exterior``, and runs every episode.
    """
    if trial_count < 1:
        raise ValueError(f'--eval.trial_count must be at least 1, got {trial_count}')
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError(f'--eval.timeout must be finite and positive, got {timeout}')
    exterior_sides = random.Random(seed).sample(list(ShoulderSide), 2) if with_right_shoulder else [ShoulderSide.LEFT]
    cameras = camera_dict(exterior_sides)
    adapter = MolmoAdapter(cameras)
    proxy = RemoteEnvControlSystem(adapter, serve_molmo_spaces())
    embodiment = remote_embodiment(
        proxy, cameras, descriptor='remote.molmo_spaces.droid', static_meta=adapter.robot_model
    )
    privileged = {mapping.OBS_SIM_STATE: proxy.privileged[mapping.OBS_SIM_STATE]}

    def tasks() -> list[Task]:
        selection = spec(
            suite=suite,
            scene_dataset=scene_dataset,
            task_config=task_config,
            benchmark=benchmark,
            **{mapping.SELECT_EPISODES: episodes, mapping.WITH_RIGHT_SHOULDER: with_right_shoulder},
        )
        if timeout is not None:
            logging.warning('--eval.timeout %ss replaces the benchmark horizon', timeout)
        trials = []
        for params in proxy.tasks(selection):
            horizon = timeout if timeout is not None else params[molmo_keys.TASK_HORIZON]
            # The env ends a trial on its first successful step, so a trial that the policy ends has failed.
            task = Task(
                instruction_source=lambda: proxy.meta[mapping.META_TASK],
                timeout_sec=horizon + _TIMEOUT_MARGIN_SEC,
                policy_end_success=False,
            )
            scene = {
                **params,
                molmo_keys.TASK_HORIZON: horizon,
                molmo_keys.EXTERIOR_SIDES: [side.value for side in exterior_sides],
            }
            trials += [
                (task, {**scene, **({eval_keys.SEED: seed + t} if seed is not None else {})})
                for t in range(trial_count)
            ]
        return number_trials(trials)

    return Eval(embodiment, tasks, privileged=privileged, done=proxy.done)


bench_v1 = benchmarks.override(suite='molmospaces-bench-v1')
bench_v2 = benchmarks.override(suite='molmospaces-bench-v2')

pick_v1 = bench_v1.override(task_config='FrankaPickDroidMiniBench')
first_episode = pick_v1.override(episodes=0)
