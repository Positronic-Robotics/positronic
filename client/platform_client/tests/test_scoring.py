"""The scorers and the tally, against a recorded MolmoSpaces sweep.

The fixture holds the `static.json` of each episode of one 20-episode sweep of `pi05_droid_jointpos`, in
the `<block>/<episode>` layout that positronic records. Five of the twenty episodes succeeded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from platform_client.evals import MOLMO_SCORER, PUBLIC_EVALS, EvalRef, ScorerRef, public_eval
from platform_client.responses import Scores as PublishedScores
from platform_client.scoring import (
    MOLMO_TASK_KEY,
    PUBLIC_SCORERS,
    STATIC_FILE,
    SUCCESS_KEY,
    Outcome,
    Scores,
    read_static,
    recorded_success,
    score,
)

SWEEP = sorted(p for p in (Path(__file__).parent / 'fixtures' / 'molmo_sweep').glob('*/*') if p.is_dir())

# Read from the recording: ten tasks over twenty trials, split irregularly. Three tasks ran three times,
# two ran twice, and four ran once.
PER_TASK = {
    'pick up the bottle.': (2, 0),
    'pick up the cup.': (1, 1),
    'pick up the kettle.': (2, 0),
    'pick up the ladle.': (3, 1),
    'pick up the mug.': (1, 0),
    'pick up the pot.': (3, 0),
    'pick up the remote.': (1, 0),
    'pick up the spatula.': (1, 1),
    'pick up the spoon.': (3, 1),
    'pick up the tissue.': (3, 1),
}

# Written into a synthetic episode and read back out of `per_task`, so the assertion turns on the two
# being the same string.
CUP = 'pick up the cup.'
KETTLE = 'pick up the kettle.'


def score_molmo(episodes: list[Path]) -> Scores:
    return score(MOLMO_SCORER, episodes)


def _episodes(tmp_path: Path, *statics: dict[str, object] | str | None) -> list[Path]:
    """One episode directory per given `static.json`: a mapping is written as JSON, a string as it is, and
    None leaves the episode without one."""
    made = []
    for index, static in enumerate(statics):
        episode = tmp_path / f'{index:012d}'
        episode.mkdir()
        if static is not None:
            (episode / STATIC_FILE).write_text(static if isinstance(static, str) else json.dumps(static))
        made.append(episode)
    return made


def test_the_sweep_scores_five_of_twenty_in_this_record():
    per_task = {
        task: {'trials': trials, 'successes': successes, 'success_rate': successes / trials, 'mean_subtask_score': None}
        for task, (trials, successes) in PER_TASK.items()
    }
    assert score_molmo(SWEEP).model_dump(mode='json') == {
        'primary': 0.25,
        'success_rate': 0.25,
        'per_task': per_task,
        'episodes': 20,
        'unscored': 0,
    }


def test_an_episode_that_recorded_no_outcome_is_unscored_rather_than_failed(tmp_path: Path):
    episodes = _episodes(tmp_path, {MOLMO_TASK_KEY: CUP, SUCCESS_KEY: True}, {MOLMO_TASK_KEY: CUP}, None)
    scores = score_molmo(episodes)
    assert (scores.episodes, scores.unscored) == (3, 2)
    assert scores.success_rate == pytest.approx(1.0)
    assert scores.per_task[CUP].trials == 1


def test_a_sweep_that_recorded_nothing_scores_zero_rather_than_dividing_by_it(tmp_path: Path):
    scores = score_molmo(_episodes(tmp_path, None))
    assert (scores.episodes, scores.unscored, scores.primary) == (1, 1, 0.0)


def test_a_failed_episode_is_scored_as_a_trial_rather_than_as_nothing(tmp_path: Path):
    scores = score_molmo(_episodes(tmp_path, {MOLMO_TASK_KEY: KETTLE, SUCCESS_KEY: False}))
    assert (scores.episodes, scores.unscored, scores.primary) == (1, 0, 0.0)
    assert scores.per_task[KETTLE].trials == 1


@pytest.mark.parametrize('static', ['{ truncated upload', '[true, "pick up the cup."]'])
def test_statics_that_are_not_a_json_object_are_unscored_rather_than_fatal(tmp_path: Path, static: str):
    (episode,) = _episodes(tmp_path, static)
    assert read_static(episode) is None
    assert score_molmo([episode]).unscored == 1


@pytest.mark.parametrize('static', [{SUCCESS_KEY: 1, MOLMO_TASK_KEY: CUP}, {SUCCESS_KEY: True, MOLMO_TASK_KEY: 3}])
def test_a_success_or_task_of_the_wrong_type_is_no_outcome(static: dict[str, object]):
    assert recorded_success(static, MOLMO_TASK_KEY) is None


def _graded(episode: Path) -> Outcome | None:
    """A caller's scorer: a success scores 1.0, and a failure scores the `grade` its statics record."""
    static = read_static(episode)
    recorded = recorded_success(static, MOLMO_TASK_KEY) if static is not None else None
    if static is None or recorded is None:
        return None
    task, succeeded = recorded
    if succeeded:
        return Outcome(task, True, 1.0)
    grade = static.get('grade')
    return Outcome(task, False, grade) if isinstance(grade, float) else None


def test_a_caller_scores_with_its_own_scorer_beside_the_public_ones(tmp_path: Path):
    graded = ScorerRef('graded')
    episodes = _episodes(
        tmp_path,
        {MOLMO_TASK_KEY: CUP, SUCCESS_KEY: True},
        {MOLMO_TASK_KEY: CUP, SUCCESS_KEY: False, 'grade': 0.5},
        {MOLMO_TASK_KEY: KETTLE, SUCCESS_KEY: False},
    )

    scores = score(graded, episodes, {**PUBLIC_SCORERS, graded: _graded})

    assert (scores.episodes, scores.unscored) == (3, 1)
    assert (scores.primary, scores.success_rate) == (pytest.approx(0.75), pytest.approx(0.5))
    assert scores.per_task[CUP].mean_subtask_score == pytest.approx(0.75)
    assert KETTLE not in scores.per_task


def test_an_ungraded_task_reports_no_subtask_score_beside_a_graded_one(tmp_path: Path):
    def grades_only_the_cup(episode: Path) -> Outcome | None:
        outcome = PUBLIC_SCORERS[MOLMO_SCORER](episode)
        if outcome is None or outcome.task != CUP:
            return outcome
        return Outcome(outcome.task, outcome.succeeded, 0.25)

    episodes = _episodes(
        tmp_path, {MOLMO_TASK_KEY: CUP, SUCCESS_KEY: False}, {MOLMO_TASK_KEY: KETTLE, SUCCESS_KEY: True}
    )
    per_task = score(ScorerRef('mixed'), episodes, {ScorerRef('mixed'): grades_only_the_cup}).per_task

    assert per_task[CUP].mean_subtask_score == pytest.approx(0.25)
    assert per_task[KETTLE].mean_subtask_score is None


def test_a_scorer_the_table_does_not_hold_is_refused():
    with pytest.raises(LookupError, match="no scorer 'held_out'; the scorers are molmo"):
        score(ScorerRef('held_out'), SWEEP)


@pytest.mark.parametrize('name', PUBLIC_EVALS)
def test_every_public_eval_names_a_public_scorer(name: EvalRef):
    assert public_eval(name).scorer in PUBLIC_SCORERS


def test_the_published_scores_are_a_subset_of_the_record():
    record = score_molmo(SWEEP).model_dump(mode='json')
    assert set(PublishedScores.model_fields) < set(Scores.model_fields)
    published = PublishedScores.model_validate(record).model_dump(mode='json')
    assert published == {name: record[name] for name in PublishedScores.model_fields}
