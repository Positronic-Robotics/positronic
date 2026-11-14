import hashlib
import logging
import random
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import configuronic as cfn
from eval_vocabulary.episode import MOLMO_TASK_METRICS_SIGNAL

from positronic import keys
from positronic.cfg.eval import env_server_address, number_trials, require_finite_positive, spec
from positronic.eval import Eval, Task
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server.proxy import EnvTask, RemoteEnvControlSystem, remote_embodiment
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import SHOULDER_CAMERAS, MolmoAdapter, ShoulderSide, camera_dict
from positronic.simulator.molmo_spaces.launcher import serve_molmo_spaces

# Margin for the harness to observe the simulator's terminal signal.
_TIMEOUT_MARGIN_SEC = 1.0


def _parse_episode_path(path: str) -> tuple[mapping.BenchmarkPath, int]:
    bench, _, index = path.rpartition('/')
    if not bench or not index.isdigit():
        raise ValueError(f'an episode path is <benchmark path>/<episode index>, not {path!r}')
    return mapping.BenchmarkPath.parse(bench), int(index)


def _env_server(
    env_hook: str | None, env_server: str | None
) -> tuple[AbstractContextManager[tuple[str, int]], dict[str, Any]]:
    """The env server of the eval, and the hook record of its trials: empty for a server that already runs."""
    if env_server is not None:
        if env_hook is not None:
            raise ValueError(
                '--eval.env_hook applies to a server that the eval launches, so --eval.env_server must be unset'
            )
        return nullcontext(env_server_address(env_server)), {}
    if env_hook is None:
        return serve_molmo_spaces(), {molmo_keys.ENV_HOOK: None}
    hook = Path(env_hook).expanduser().resolve()
    digest = hashlib.sha256(hook.read_bytes()).hexdigest()
    return serve_molmo_spaces(hook=hook), {molmo_keys.ENV_HOOK: f'{hook.name}@sha256:{digest}'}


_EPISODE_CAMERA_KEYS = (keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2)
_SHOULDER_ALIASES = frozenset(camera for cameras in SHOULDER_CAMERAS.values() for camera in cameras)


def _cameras(exterior_sides: list[ShoulderSide], episode_cameras: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """The camera aliases of each image key: the shoulder of each exterior side, then ``episode_cameras``."""
    cameras = camera_dict(exterior_sides)
    for key, camera in episode_cameras.items():
        if key not in _EPISODE_CAMERA_KEYS:
            raise ValueError(f'--eval.episode_cameras maps {key}, not one of {", ".join(_EPISODE_CAMERA_KEYS)}')
        if camera in _SHOULDER_ALIASES:
            raise ValueError(
                f'--eval.episode_cameras maps {key} to the shoulder camera {camera}; with_right_shoulder sets those'
            )
        if len(exterior_sides) > 1 and key in (keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2):
            raise ValueError(
                f'--eval.episode_cameras replaces {key}, whose camera the seed draws; set with_right_shoulder=False'
            )
        cameras[key] = (camera,)
    return cameras


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
    with_object_image_points=True,
    episode_cameras=None,
    env_hook=None,
    env_server=None,
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
    with_object_image_points: bool,
    episode_cameras: dict[str, str] | None,
    env_hook: str | None,
    env_server: str | None,
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

    ``with_object_image_points=False`` drops the object image points sensor of MolmoSpaces, which renders a segmentation
    mask for each camera at each step. No observation reads it, but it draws from the global NumPy generator, so a
    scored eval keeps it.

    ``episode_cameras`` maps the wrist or an exterior image key to an episode camera that the env serves under it,
    in place of the default camera, for example ``{'image.exterior_2': 'exo_camera_2'}``. An exterior key needs
    ``with_right_shoulder=False``, and a shoulder camera is refused. Every trial records the map under
    ``molmo.episode_cameras``. A map that replaces ``image.exterior`` records no ``molmo.exterior_sides``.

    ``env_hook`` is a file that defines ``HOOK``, an ``env_hook.EnvHook`` that the server applies to each episode:
    for example a task class, a camera model or a grip scale of its own. A hook can change the robot only within the
    bundled FR3 joints, kinematics and 2F-85 grasp site, for example its mount or a gripper detail, and the eval
    records the bundled robot model. Every trial records the hook file's name and SHA-256 digest under
    ``molmo.env_hook``, or None without a hook. A run with a hook or an episode camera is not comparable with
    stock scores.

    ``env_server`` is the ``host:port`` of a MolmoSpaces env server that already runs. Without it, the eval launches
    a server of its own. ``env_hook`` applies only to a server that the eval launches, and the trials of a dialed
    server record no ``molmo.env_hook``. Start a server to dial with
    ``python -m positronic.simulator.molmo_spaces.launcher --port <port> [--hook <file>]``: it serves one eval run,
    with one episode at a time, and stops when that run ends.
    """
    if trial_count < 1:
        raise ValueError(f'--eval.trial_count must be at least 1, got {trial_count}')
    require_finite_positive('timeout', timeout)
    require_finite_positive('horizon_sec', horizon_sec)
    if episode_paths is not None and not episode_paths:
        raise ValueError('--eval.episode_paths must name at least one episode')
    pins = None if episode_paths is None else [_parse_episode_path(path) for path in episode_paths]
    if pins is not None:
        names = (*mapping.BenchmarkPath._fields, mapping.SELECT_EPISODES)
        values = (suite, scene_dataset, task_config, benchmark, episodes)
        bound = [name for name, value in zip(names, values, strict=True) if value is not None]
        if bound:
            raise ValueError(f'--eval.episode_paths names its benchmarks, so --eval.{bound[0]} must be unset')
    serve, hook_record = _env_server(env_hook, env_server)
    episode_cameras = episode_cameras or {}
    exterior_sides = random.Random(seed).sample(list(ShoulderSide), 2) if with_right_shoulder else [ShoulderSide.LEFT]
    cameras = _cameras(exterior_sides, episode_cameras)
    sides_record = (
        {} if keys.EXTERIOR_IMAGE in episode_cameras else {molmo_keys.EXTERIOR_SIDES: [s.value for s in exterior_sides]}
    )
    adapter = MolmoAdapter(cameras, with_right_shoulder, with_object_image_points)
    proxy = RemoteEnvControlSystem(adapter, serve)
    embodiment = remote_embodiment(
        proxy, cameras, descriptor='remote.molmo_spaces.droid', static_meta=adapter.robot_model
    )
    privileged = {
        mapping.OBS_SIM_STATE: proxy.privileged[mapping.OBS_SIM_STATE],
        MOLMO_TASK_METRICS_SIGNAL: proxy.privileged[mapping.OBS_TASK_METRICS],
    }

    def pinned_listing(pins: list[tuple[mapping.BenchmarkPath, int]]) -> list[EnvTask]:
        """The env's task of each pin, in pin order: one listing per benchmark."""
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
            for entry in proxy.tasks(selection):
                listed_bench = mapping.BenchmarkPath(*(entry.params[key] for key in molmo_keys.BENCHMARK_DIMENSIONS))
                found[listed_bench, entry.params[molmo_keys.EPISODE_INDEX]] = entry
        return [found[pin] for pin in pins]

    def listed() -> list[EnvTask]:
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
        for entry in listed():
            params = entry.params
            deadline = timeout if timeout is not None else params[molmo_keys.TASK_HORIZON] + _TIMEOUT_MARGIN_SEC
            task = Task(instruction=entry.instruction, timeout_sec=deadline)
            scene = {**params, **sides_record, molmo_keys.EPISODE_CAMERAS: episode_cameras, **hook_record}
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
# grasp site from the right shoulder camera. It drops the object image points sensor, so each step runs faster.
pick_mini_collect = bench_v2.override(
    scene_dataset='procthor-10k',
    task_config='FrankaPickDroidMiniBench',
    benchmark='FrankaPickDroidMiniBench_json_benchmark_20251231',
    horizon_sec=120.0,
    with_object_image_points=False,
)
