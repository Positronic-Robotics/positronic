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


def _parse_episode_path(path: str) -> tuple[mapping.BenchmarkPath, int]:
    bench, _, index = path.rpartition('/')
    if not bench or not index.isdigit():
        raise ValueError(f'an episode path is <benchmark path>/<episode index>, not {path!r}')
    return mapping.BenchmarkPath.parse(bench), int(index)


def _require_finite_positive(arg: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError(f'--eval.{arg} must be finite and positive, got {value}')


@cfn.config(
    suite=None,
    scene_dataset=None,
    task_config=None,
    benchmark=None,
    episodes=None,
    episode_paths=None,
    trial_count=1,
    timeout=None,
    seed=None,
    horizon_sec=None,
    with_right_shoulder=True,
)
def benchmarks(
    suite: str | list[str] | None,
    scene_dataset: str | list[str] | None,
    task_config: str | list[str] | None,
    benchmark: str | list[str] | None,
    episodes: int | list[int] | None,
    episode_paths: list[str] | None,
    trial_count: int,
    timeout: float | None,
    seed: int | None,
    horizon_sec: float | None,
    with_right_shoulder: bool,
) -> Eval:
    """A DROID evaluation over selected MolmoSpaces benchmark episodes.

    Each benchmark dimension accepts a name, a list or None (all found); ``episodes`` accepts indices.
    ``episode_paths`` pins episodes across benchmarks instead, each as ``<benchmark path>/<index>``, and runs
    them in list order; it takes no benchmark dimension and no ``episodes``.
    ``seed=None`` follows the benchmark's seed convention. The default timeout is the benchmark horizon
    plus a margin; an explicit timeout replaces it. ``horizon_sec`` replaces the horizon of every selected
    benchmark, and the default timeout follows it. It is for demonstrations by a person, not for a scored eval.

    The policy sees the wrist camera and two shoulder cameras, left and right, in ``image.exterior`` and
    ``image.exterior_2``. The run draws their order once from ``seed``, or at random when ``seed`` is None,
    and every trial records it under ``molmo.exterior_sides``. The right shoulder camera hides the grasp site
    in some episodes; an open selection leaves those out, and a pinned one is refused.

    ``with_right_shoulder=False`` serves the left shoulder camera alone, in ``image.exterior``, and runs every episode.
    """
    if trial_count < 1:
        raise ValueError(f'--eval.trial_count must be at least 1, got {trial_count}')
    _require_finite_positive('timeout', timeout)
    _require_finite_positive('horizon_sec', horizon_sec)
    if episode_paths is not None and not episode_paths:
        raise ValueError('--eval.episode_paths must name at least one episode')
    pins = None if episode_paths is None else [_parse_episode_path(path) for path in episode_paths]
    if pins is not None:
        names = (*mapping.BenchmarkPath._fields, mapping.SELECT_EPISODES)
        values = (suite, scene_dataset, task_config, benchmark, episodes)
        bound = [name for name, value in zip(names, values, strict=True) if value is not None]
        if bound:
            raise ValueError(f'--eval.episode_paths names its benchmarks, so --eval.{bound[0]} must be unset')
    exterior_sides = random.Random(seed).sample(list(ShoulderSide), 2) if with_right_shoulder else [ShoulderSide.LEFT]
    cameras = camera_dict(exterior_sides)
    adapter = MolmoAdapter(cameras)
    proxy = RemoteEnvControlSystem(adapter, serve_molmo_spaces())
    embodiment = remote_embodiment(
        proxy, cameras, descriptor='remote.molmo_spaces.droid', static_meta=adapter.robot_model
    )
    privileged = {mapping.OBS_SIM_STATE: proxy.privileged[mapping.OBS_SIM_STATE]}

    def pinned_listing(pins: list[tuple[mapping.BenchmarkPath, int]]) -> list[dict]:
        """The env's trial params of each pin, in pin order: one listing per benchmark."""
        by_bench: dict[mapping.BenchmarkPath, list[int]] = {}
        for bench, index in pins:
            by_bench.setdefault(bench, []).append(index)
        found = {}
        for bench, indices in by_bench.items():
            selection = spec(
                **bench._asdict(),
                **{
                    mapping.SELECT_EPISODES: list(dict.fromkeys(indices)),
                    mapping.SELECT_HORIZON_SEC: horizon_sec,
                    mapping.WITH_RIGHT_SHOULDER: with_right_shoulder,
                },
            )
            for params in proxy.tasks(selection):
                listed_bench = mapping.BenchmarkPath(*(params[key] for key in molmo_keys.BENCHMARK_DIMENSIONS))
                found[listed_bench, params[molmo_keys.EPISODE_INDEX]] = params
        return [found[pin] for pin in pins]

    def listed() -> list[dict]:
        if pins is not None:
            return pinned_listing(pins)
        selection = spec(
            suite=suite,
            scene_dataset=scene_dataset,
            task_config=task_config,
            benchmark=benchmark,
            **{
                mapping.SELECT_EPISODES: episodes,
                mapping.SELECT_HORIZON_SEC: horizon_sec,
                mapping.WITH_RIGHT_SHOULDER: with_right_shoulder,
            },
        )
        return proxy.tasks(selection)

    def tasks() -> list[Task]:
        if timeout is not None:
            logging.warning('--eval.timeout %ss replaces the benchmark horizon backstop', timeout)
        trials = []
        for params in listed():
            deadline = timeout if timeout is not None else params[molmo_keys.TASK_HORIZON] + _TIMEOUT_MARGIN_SEC
            task = Task(instruction_source=lambda: proxy.meta[mapping.META_TASK], timeout_sec=deadline)
            scene = {**params, molmo_keys.EXTERIOR_SIDES: [side.value for side in exterior_sides]}
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

# Demonstrations by a person on the pick mini benchmark: two minutes for each episode, not the benchmark horizon. It
# runs the episodes that the eval selects with no `episodes`, so it leaves out each episode that hides the start
# grasp site from the right shoulder camera.
pick_mini_collect = bench_v2.override(
    scene_dataset='procthor-10k',
    task_config='FrankaPickDroidMiniBench',
    benchmark='FrankaPickDroidMiniBench_json_benchmark_20251231',
    horizon_sec=120.0,
)
