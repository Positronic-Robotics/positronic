"""MolmoSpaces trial construction and evaluation through the policy harness."""

import logging
from contextlib import nullcontext
from typing import Any

import numpy as np
import pos3
import pytest

from positronic import keys
from positronic.cfg.eval.sim import molmo as molmo_cfg
from positronic.cfg.eval.sim.molmo import _TIMEOUT_MARGIN_SEC, benchmarks
from positronic.cli.eval.run import main
from positronic.dataset import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.drivers.roboarm.command import JointPosition
from positronic.eval import keys as eval_keys
from positronic.policy.tests.test_harness import StubPolicy
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.env_server.server import EnvProtocol
from positronic.simulator.env_server.tests.conftest import serve_env
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import ShoulderSide, camera_dict
from positronic.simulator.molmo_spaces.tests.make_fixture import build_payload

_SUITE, _SCENE_DATASET, _TASK_CONFIG, _BENCHMARK = molmo_keys.BENCHMARK_DIMENSIONS
_PICK = {
    _SUITE: 'molmospaces-bench-v1',
    _SCENE_DATASET: 'procthor-10k',
    _TASK_CONFIG: 'FrankaPickDroidMiniBench',
    _BENCHMARK: 'pick_20251231',
}
_PLACE = {**_PICK, _TASK_CONFIG: 'FrankaPickandPlaceDroidMiniBench', _BENCHMARK: 'pnp_20260111'}
_PICK_HORIZON_SEC = 30.0
_PLACE_HORIZON_SEC = 60.0


@pytest.fixture
def asked(monkeypatch) -> list[Any]:
    """Specs sent to the proxy, which returns one episode from each of two benchmarks."""
    specs: list[Any] = []

    def tasks(self, selection: Any) -> list[dict[str, Any]]:
        specs.append(selection)
        return [
            {
                **_PICK,
                eval_keys.TASK: 'put the banana in the bowl',
                molmo_keys.EPISODE_INDEX: 0,
                molmo_keys.TASK_HORIZON: _PICK_HORIZON_SEC,
            },
            {
                **_PLACE,
                eval_keys.TASK: 'put the mug on the shelf',
                molmo_keys.EPISODE_INDEX: 0,
                molmo_keys.TASK_HORIZON: _PLACE_HORIZON_SEC,
            },
        ]

    monkeypatch.setattr(RemoteEnvControlSystem, 'tasks', tasks)
    return specs


def test_the_env_answers_which_episodes_the_sweep_runs(asked):
    ev = benchmarks.override(trial_count=2).instantiate()
    assert asked == []

    trials = list(ev.tasks())

    assert asked == [{mapping.WITH_RIGHT_SHOULDER: True}]
    scenes = [trial.prepare_args[eval_keys.SCENE] for trial in trials]
    assert [scene[_TASK_CONFIG] for scene in scenes] == [
        'FrankaPickDroidMiniBench',
        'FrankaPickDroidMiniBench',
        'FrankaPickandPlaceDroidMiniBench',
        'FrankaPickandPlaceDroidMiniBench',
    ]
    assert all(eval_keys.SEED not in scene for scene in scenes)
    assert [trial.meta[eval_keys.TRIAL_INDEX] for trial in trials] == [0, 1, 2, 3]
    assert trials[0].meta[_BENCHMARK] == 'pick_20251231'


def test_the_benchmark_dimensions_and_the_episode_selection_ride_the_spec(asked):
    ev = benchmarks.override(suite='molmospaces-bench-v2', task_config=['A', 'B'], episodes=[0, 1]).instantiate()
    ev.tasks()
    assert asked == [
        {
            'suite': 'molmospaces-bench-v2',
            'task_config': ['A', 'B'],
            mapping.SELECT_EPISODES: [0, 1],
            mapping.WITH_RIGHT_SHOULDER: True,
        }
    ]


def test_a_horizon_override_rides_the_spec_and_no_override_sends_none(asked):
    benchmarks.override(horizon_sec=120.0).instantiate().tasks()
    benchmarks.instantiate().tasks()
    assert asked == [
        {mapping.SELECT_HORIZON_SEC: 120.0, mapping.WITH_RIGHT_SHOULDER: True},
        {mapping.WITH_RIGHT_SHOULDER: True},
    ]


@pytest.mark.parametrize('horizon_sec', [0.0, -1.0, float('nan'), float('inf')])
def test_an_invalid_horizon_override_is_refused(horizon_sec):
    with pytest.raises(ValueError, match='horizon_sec must be finite and positive'):
        benchmarks.override(horizon_sec=horizon_sec).instantiate()


def test_the_collection_config_asks_the_pick_mini_benchmark_for_a_two_minute_horizon(asked):
    molmo_cfg.pick_mini_collect.instantiate().tasks()
    assert asked == [
        {
            'suite': 'molmospaces-bench-v2',
            'scene_dataset': 'procthor-10k',
            'task_config': 'FrankaPickDroidMiniBench',
            'benchmark': 'FrankaPickDroidMiniBench_json_benchmark_20251231',
            mapping.SELECT_HORIZON_SEC: 120.0,
            mapping.WITH_RIGHT_SHOULDER: True,
        }
    ]
    assert molmo_cfg.pick_mini_collect.kwargs['object_image_points'] is False


def _listing_of(selection: dict[str, Any]) -> list[dict[str, Any]]:
    """What the env lists for a selection of one benchmark and pinned episodes."""
    bench = {
        key: selection[dimension]
        for dimension, key in zip(mapping.BenchmarkPath._fields, molmo_keys.BENCHMARK_DIMENSIONS, strict=True)
    }
    return [
        {**bench, eval_keys.TASK: 'pick it', molmo_keys.EPISODE_INDEX: i, molmo_keys.TASK_HORIZON: _PICK_HORIZON_SEC}
        for i in selection[mapping.SELECT_EPISODES]
    ]


def test_episode_paths_list_each_benchmark_once_and_run_the_pins_in_their_order(monkeypatch):
    specs: list[dict[str, Any]] = []
    monkeypatch.setattr(
        RemoteEnvControlSystem, 'tasks', lambda self, selection: specs.append(selection) or _listing_of(selection)
    )
    pick = 'molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/pick_20251231'
    hard = 'molmospaces-bench-v1/holodeck-objaverse/FrankaPickHardBench/hard_20260206'
    paths = [f'{pick}/11', f'{hard}/0', f'{pick}/49', f'{hard}/295']

    trials = benchmarks.override(episode_paths=paths, trial_count=2).instantiate().tasks()

    assert [s[mapping.SELECT_EPISODES] for s in specs] == [[11, 49], [0, 295]]
    assert [mapping.BenchmarkPath(*(s[f] for f in mapping.BenchmarkPath._fields)).scene_dataset for s in specs] == [
        'procthor-10k',
        'holodeck-objaverse',
    ]
    scenes = [trial.prepare_args[eval_keys.SCENE] for trial in trials]
    assert [(scene[_SCENE_DATASET], scene[molmo_keys.EPISODE_INDEX]) for scene in scenes] == [
        ('procthor-10k', 11),
        ('procthor-10k', 11),
        ('holodeck-objaverse', 0),
        ('holodeck-objaverse', 0),
        ('procthor-10k', 49),
        ('procthor-10k', 49),
        ('holodeck-objaverse', 295),
        ('holodeck-objaverse', 295),
    ]


def test_episode_paths_carry_the_horizon_override(monkeypatch):
    specs: list[dict[str, Any]] = []
    monkeypatch.setattr(
        RemoteEnvControlSystem, 'tasks', lambda self, selection: specs.append(selection) or _listing_of(selection)
    )
    path = 'molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/pick_20251231/11'

    benchmarks.override(episode_paths=[path], horizon_sec=120.0).instantiate().tasks()
    benchmarks.override(episode_paths=[path]).instantiate().tasks()

    assert [s.get(mapping.SELECT_HORIZON_SEC) for s in specs] == [120.0, None]


@pytest.mark.parametrize('bound', [{'suite': 'molmospaces-bench-v1'}, {'episodes': [0]}, {'task_config': 'A'}])
def test_episode_paths_refuse_a_benchmark_dimension_or_episodes_beside_them(asked, bound):
    path = 'molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/pick_20251231/11'
    with pytest.raises(ValueError, match=f'--eval.{next(iter(bound))} must be unset'):
        benchmarks.override(episode_paths=[path], **bound).instantiate()
    assert asked == []


@pytest.mark.parametrize('path', ['procthor-10k/FrankaPickDroidMiniBench/pick_20251231/11', 'a/b/c/d/x', 'a/b/c/d'])
def test_a_malformed_episode_path_is_refused(asked, path):
    with pytest.raises(ValueError):
        benchmarks.override(episode_paths=[path]).instantiate()
    assert asked == []


def test_an_empty_episode_path_list_is_refused(asked):
    with pytest.raises(ValueError, match='--eval.episode_paths must name at least one episode'):
        benchmarks.override(episode_paths=[]).instantiate()
    assert asked == []


def test_an_explicit_seed_sweeps_each_episode(asked):
    ev = benchmarks.override(seed=5, trial_count=2).instantiate()
    scenes = [trial.prepare_args[eval_keys.SCENE] for trial in ev.tasks()]
    assert [scene[eval_keys.SEED] for scene in scenes] == [5, 6, 5, 6]


def _exterior_sides(ev) -> list:
    return [trial.meta[molmo_keys.EXTERIOR_SIDES] for trial in ev.tasks()]


@pytest.mark.parametrize('seed', [None, 3])
def test_every_trial_of_a_run_shares_one_exterior_order(asked, seed):
    ev = benchmarks.override(seed=seed, trial_count=2).instantiate()
    sides = _exterior_sides(ev)
    assert len(sides) == 4 and all(s == sides[0] for s in sides)
    assert sorted(sides[0]) == [side.value for side in ShoulderSide]
    assert {keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2} <= ev.embodiment.observations.keys()


def test_an_explicit_seed_fixes_the_exterior_order_and_seeds_draw_both_orders(asked):
    orders = {seed: _exterior_sides(benchmarks.override(seed=seed).instantiate())[0] for seed in range(16)}
    assert all(_exterior_sides(benchmarks.override(seed=seed).instantiate())[0] == orders[seed] for seed in orders)
    assert {tuple(order) for order in orders.values()} == {tuple(ShoulderSide), tuple(reversed(ShoulderSide))}


def test_a_run_without_the_right_shoulder_serves_the_left_shoulder_alone_and_keeps_every_episode(asked):
    ev = benchmarks.override(with_right_shoulder=False, seed=3).instantiate()
    assert _exterior_sides(ev) == [[ShoulderSide.LEFT.value]] * 2
    assert asked[-1] == {mapping.WITH_RIGHT_SHOULDER: False}
    assert keys.EXTERIOR_IMAGE in ev.embodiment.observations.keys()
    assert keys.EXTERIOR_IMAGE_2 not in ev.embodiment.observations.keys()


def test_a_non_positive_trial_count_is_refused():
    with pytest.raises(ValueError, match='trial_count'):
        benchmarks.override(trial_count=0).instantiate()


@pytest.mark.parametrize('timeout', [0.0, -1.0, float('nan'), float('inf'), -float('inf')])
def test_invalid_timeout_fails_before_discovering_tasks(asked, timeout):
    with pytest.raises(ValueError, match='timeout must be finite and positive'):
        benchmarks.override(timeout=timeout).instantiate()
    assert asked == []


def test_timeout_defaults_to_each_benchmark_horizon_plus_a_margin(asked, caplog):
    with caplog.at_level(logging.WARNING):
        trials = list(benchmarks.instantiate().tasks())
    expected = [_PICK_HORIZON_SEC + _TIMEOUT_MARGIN_SEC, _PLACE_HORIZON_SEC + _TIMEOUT_MARGIN_SEC]
    assert [trial.timeout_sec for trial in trials] == expected
    assert not caplog.records


def test_an_explicit_timeout_replaces_the_backstop_and_warns(asked, caplog):
    with caplog.at_level(logging.WARNING):
        short = benchmarks.override(timeout=20.0).instantiate().tasks()
        long = benchmarks.override(timeout=999.0).instantiate().tasks()
    assert [trial.timeout_sec for trial in short] == [20.0, 20.0]
    assert [trial.timeout_sec for trial in long] == [999.0, 999.0]
    assert len(caplog.records) == 2


class _MolmoObservationEnv(EnvProtocol):
    BENCHMARK = mapping.BenchmarkPath(
        'molmospaces-bench-v1', 'procthor-10k', 'FrankaPickDroidMiniBench', 'pick_20251231'
    )
    INSTRUCTION = 'Hold the arm still'
    CONTROL_DT = 0.1
    HORIZON_STEPS = 5

    def __init__(self, done_after: int | None):
        self._done_after = done_after
        self.selections = []
        self.tokens = []
        self.actions = []
        self.observation = {**build_payload(), mapping.OBS_SIM_STATE: np.arange(10, dtype=np.float64)}

    def tasks(self, spec):
        self.selections.append(spec)
        return [
            {
                **self.BENCHMARK._asdict(),
                mapping.TOKEN_EPISODE_INDEX: 3,
                mapping.TASK_NAME: self.INSTRUCTION,
                mapping.TASK_HORIZON_SEC: self.HORIZON_STEPS * self.CONTROL_DT,
            }
        ]

    def reset(self, token):
        self.tokens.append(token)
        self.actions.clear()
        return {
            protocol.SLOTS: [{protocol.FRAME_OBS: self.observation}],
            protocol.FRAME_META: {mapping.META_TASK: self.INSTRUCTION},
            protocol.FRAME_ROBOT_META: {},
            protocol.FRAME_CONTROL_DT: self.CONTROL_DT,
        }

    def step(self, actions):
        (action,) = actions.values()  # this env serves one slot
        self.actions.append(action)
        return {
            protocol.SLOTS: [
                {
                    protocol.FRAME_OBS: self.observation,
                    protocol.FRAME_DONE: self._done_after is not None and len(self.actions) >= self._done_after,
                    protocol.FRAME_SUCCESS: False,
                }
            ],
            protocol.FRAME_CONTROL_DT: self.CONTROL_DT,
        }

    def close(self):
        pass


@pytest.mark.timeout(60.0)
@pytest.mark.parametrize(('done_after', 'timeout', 'seed'), [(5, None, 7), (None, 0.3, None)])
def test_molmo_eval_carries_observations_commands_and_trial_results(monkeypatch, tmp_path, done_after, timeout, seed):
    env = _MolmoObservationEnv(done_after)
    joints = env.observation[protocol.OBS_JOINT_POS]
    policy = StubPolicy(command=JointPosition(joints), target_grip=0.0)
    with serve_env(env) as address, pos3.mirror():
        monkeypatch.setattr(molmo_cfg, 'serve_molmo_spaces', lambda: nullcontext(address))
        ev = benchmarks.override(**env.BENCHMARK._asdict(), episodes=3, seed=seed, timeout=timeout).instantiate()
        main(policy=policy, evals=[ev], output_dir=tmp_path)

    assert env.selections == [
        {**env.BENCHMARK._asdict(), mapping.SELECT_EPISODES: 3, mapping.WITH_RIGHT_SHOULDER: True}
    ]
    assert env.tokens == [
        {
            **env.BENCHMARK._asdict(),
            mapping.TOKEN_EPISODE_INDEX: 3,
            mapping.TOKEN_SEED: seed,
            mapping.WITH_RIGHT_SHOULDER: True,
            mapping.WITH_OBJECT_IMAGE_POINTS: True,
        }
    ]
    assert policy.observations
    last_obs = policy.observations[-1]
    assert last_obs[keys.TASK] == env.INSTRUCTION
    np.testing.assert_array_equal(last_obs[keys.JOINTS], joints)
    assert last_obs[keys.GRIP] == env.observation[protocol.OBS_GRIP]
    dataset = LocalDataset(tmp_path)
    assert len(dataset) == 1
    episode = dataset[0]
    assert isinstance(episode, Episode)
    for logical, candidates in camera_dict([
        ShoulderSide(s) for s in episode.static[molmo_keys.EXTERIOR_SIDES]
    ]).items():
        np.testing.assert_array_equal(last_obs[logical], env.observation[candidates[-1]])
    wires = [protocol.single_arm(action) for action in env.actions]
    commands = [w for w in wires if w[protocol.ROBOT_COMMAND][protocol.COMMAND_TYPE] == protocol.JOINT_POS]
    assert commands, 'the policy command never reached the environment'
    for wire in commands:
        np.testing.assert_array_equal(wire[protocol.ROBOT_COMMAND][protocol.COMMAND_JOINT_POS], joints)
        assert wire[protocol.TARGET_GRIP] == 0.0

    assert episode.static[eval_keys.TERMINATED] is (done_after is not None)
    if done_after is not None:
        assert episode.static[eval_keys.SUCCESS] is False
        assert len(env.actions) == done_after
    else:
        assert eval_keys.SUCCESS not in episode.static
    recorded = {keys.JOINTS, keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2, mapping.OBS_SIM_STATE}
    assert recorded <= episode.signals.keys()


@pytest.mark.timeout(60.0)
def test_an_eval_without_the_object_image_points_asks_each_reset_to_drop_them(monkeypatch, tmp_path):
    env = _MolmoObservationEnv(done_after=2)
    policy = StubPolicy(command=JointPosition(env.observation[protocol.OBS_JOINT_POS]), target_grip=0.0)
    with serve_env(env) as address, pos3.mirror():
        monkeypatch.setattr(molmo_cfg, 'serve_molmo_spaces', lambda: nullcontext(address))
        ev = benchmarks.override(**env.BENCHMARK._asdict(), episodes=3, object_image_points=False).instantiate()
        main(policy=policy, evals=[ev], output_dir=tmp_path)

    assert [token[mapping.WITH_OBJECT_IMAGE_POINTS] for token in env.tokens] == [False]
