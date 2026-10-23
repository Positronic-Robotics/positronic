"""The name of an eval, and the definitions of the public evals."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from eval_vocabulary.episode import (
    ABC_REWARD_SIGNAL,
    BENCHMARK_TASK,
    META_FILE,
    SEED,
    SIGNAL_FILE_SUFFIX,
    STATIC_FILE,
    SUCCESS,
    TERMINATED,
    TIMESTAMP_COLUMN_PREFIX,
    UID,
    VALUE_COLUMN,
)
from eval_vocabulary.timeline import EMITTED_WORLD
from positronic_evals.evals import (
    ABC_CONFIG,
    ABC_FIRST_SEED,
    ABC_MULTITASK_TASKS,
    ABC_SCORED_TASKS,
    ABC_TRIALS_PER_TASK,
    MOLMO_CONFIG,
    MOLMO_EPISODE_INDEX_KEY,
    PUBLIC_EVALS,
    EvalDefinition,
    EvalRef,
    public_eval,
)
from positronic_evals.labels import label
from positronic_evals.scoring import Scores, score
from pydantic import ValidationError


def test_an_eval_name_is_a_str_carrying_its_type():
    ref = EvalRef('robolab.public_subset')
    assert ref == 'robolab.public_subset'
    assert isinstance(ref, str)


@pytest.mark.parametrize('value', ['', ' ', 'two words', 'trailing '])
def test_a_value_that_could_never_name_an_eval_is_refused_here(value: str):
    with pytest.raises(ValueError):
        EvalRef(value)


BOARD = EvalRef('molmo.franka_pick_mini')
SMOKE = EvalRef('molmo.franka_pick_mini_smoke')


def test_the_smoke_eval_runs_the_first_five_benchmark_episodes_once_each():
    definition = public_eval(SMOKE)

    assert definition.config == '.sim.molmo.benchmarks'
    assert (definition.args['episodes'], definition.args['trial_count']) == ([0, 1, 2, 3, 4], 1)
    assert definition.time_limit_s == 7200


def test_the_board_eval_runs_the_first_twenty_benchmark_episodes_past_episode_6_once_each():
    definition = public_eval(BOARD)

    assert definition.config == '.sim.molmo.benchmarks'
    assert (definition.args['episodes'], definition.args['trial_count']) == ([*range(6), *range(7, 21)], 1)
    assert definition.time_limit_s == 7200


@pytest.mark.parametrize('name', [name for name in PUBLIC_EVALS if public_eval(name).config == MOLMO_CONFIG])
def test_a_molmo_definition_runs_each_episode_its_arguments_name_trial_count_times(name: EvalRef):
    definition = public_eval(name)
    episodes, trial_count = definition.args['episodes'], definition.args['trial_count']
    assert isinstance(episodes, list) and isinstance(trial_count, int)

    (task,) = definition.tasks
    assert task.trials == [{MOLMO_EPISODE_INDEX_KEY: i} for i in episodes for _ in range(trial_count)]


ABC_EVAL = EvalRef('abc.yam_bimanual')
ABC_MULTITASK = EvalRef('abc.yam_bimanual_multitask')


def test_the_full_abc_eval_runs_the_29_tasks_abc_scores_and_the_multitask_eval_five_of_them():
    assert len(set(ABC_SCORED_TASKS)) == len(ABC_SCORED_TASKS) == 29
    assert len(set(ABC_MULTITASK_TASKS)) == len(ABC_MULTITASK_TASKS) == 5
    assert set(ABC_MULTITASK_TASKS) <= set(ABC_SCORED_TASKS)
    assert [task.name for task in public_eval(ABC_EVAL).tasks] == list(ABC_SCORED_TASKS)
    assert [task.name for task in public_eval(ABC_MULTITASK).tasks] == list(ABC_MULTITASK_TASKS)


@pytest.mark.parametrize('name', [name for name in PUBLIC_EVALS if public_eval(name).config == ABC_CONFIG])
def test_an_abc_definition_runs_each_task_its_arguments_name_on_the_scenes_of_abcs_own_eval(name: EvalRef):
    definition = public_eval(name)
    tasks, seed, trial_count = (definition.args[key] for key in ('task', 'seed', 'trial_count'))
    assert (seed, trial_count) == (ABC_FIRST_SEED, ABC_TRIALS_PER_TASK) == (20260511, 20)
    assert isinstance(tasks, list) and isinstance(seed, int) and isinstance(trial_count, int)

    assert [task.name for task in definition.tasks] == tasks
    assert {task.name: task.trials for task in definition.tasks} == {
        task: [{SEED: seed + i} for i in range(trial_count)] for task in tasks
    }


@pytest.mark.parametrize('name', [name for name in PUBLIC_EVALS if public_eval(name).config == ABC_CONFIG])
def test_an_abc_definition_pins_the_size_the_cameras_render_at(name: EvalRef):
    args = public_eval(name).args
    assert (args['camera_height'], args['camera_width']) == (260, 416)


def _abc_episode(path: Path, static: dict[str, object] | str) -> Path:
    path.mkdir()
    (path / META_FILE).write_text(json.dumps({UID: path.name}))
    (path / STATIC_FILE).write_text(static if isinstance(static, str) else json.dumps(static))
    columns = {f'{TIMESTAMP_COLUMN_PREFIX}{EMITTED_WORLD}': [0, 1_000_000_000], VALUE_COLUMN: [0.0, 0.5]}
    pq.write_table(pa.table(columns), path / f'{ABC_REWARD_SIGNAL}{SIGNAL_FILE_SUFFIX}')
    return path


def _abc_scores(tmp_path: Path, *statics: dict[str, object] | str) -> Scores:
    scoring = public_eval(ABC_EVAL).scoring
    episodes = [_abc_episode(tmp_path / f'{i:012d}', static) for i, static in enumerate(statics)]
    picked = [label(episode, scoring.labelers) for episode in episodes]
    return score(scoring, [None if given is None else given[1] for given in picked])


BOTTLES_SUCCEEDED = {BENCHMARK_TASK: 'put_plastic_bottles_in_bin', SUCCESS: True, TERMINATED: True}


def test_an_abc_run_scores_a_trial_that_ran_out_of_time_as_a_failure(tmp_path: Path):
    timed_out = {BENCHMARK_TASK: 'put_plastic_bottles_in_bin', TERMINATED: False}

    assert _abc_scores(tmp_path, BOTTLES_SUCCEEDED, timed_out) == Scores(primary=0.5, episodes=2, unscored=0)


def test_an_abc_run_leaves_an_episode_with_damaged_statics_unscored(tmp_path: Path):
    assert _abc_scores(tmp_path, BOTTLES_SUCCEEDED, '{ truncated upload') == Scores(primary=1.0, episodes=2, unscored=1)


def test_an_abc_run_gives_each_task_the_same_weight(tmp_path: Path):
    mug_failed = {BENCHMARK_TASK: 'turn_mug_right_side_up', SUCCESS: False, TERMINATED: True}

    assert _abc_scores(tmp_path, BOTTLES_SUCCEEDED, BOTTLES_SUCCEEDED, mug_failed).primary == 0.5


def test_a_name_the_public_code_does_not_define_is_refused_with_the_public_names():
    with pytest.raises(LookupError) as refused:
        public_eval(EvalRef('molmo.held_out'))

    assert "'molmo.held_out' is not a public eval" in str(refused.value)
    assert all(name in str(refused.value) for name in PUBLIC_EVALS)


def test_a_public_definition_is_handed_out_as_a_copy():
    before = public_eval(SMOKE)
    handed = public_eval(SMOKE)
    episodes = handed.args['episodes']
    assert isinstance(episodes, list)
    episodes.append(999)
    handed.tasks[0].trials.clear()

    assert public_eval(SMOKE) == before


@pytest.mark.parametrize(
    'change',
    [
        {'unknown_field': 1},
        {'config': 'sim.molmo.benchmarks'},
        {'config': '@positronic.cfg.eval.sim.molmo.benchmarks'},
        {'positronic_revision': 'a9e13e8a'},
        {'time_limit_s': 0},
        {'tasks': []},
        {'scoring': {'labelers': [], 'primary': {'kind': 'mean_episode_score'}}},
    ],
)
def test_a_definition_refuses_a_field_no_run_can_use(change: dict):
    stated = public_eval(SMOKE).model_dump()

    with pytest.raises(ValidationError):
        EvalDefinition.model_validate({**stated, **change})
