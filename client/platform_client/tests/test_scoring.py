"""The episode score of a label, and the scores of a run, against a recorded MolmoSpaces sweep.

The fixture is the sweep that `test_labels.py` describes. Five of its twenty episodes succeeded.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from platform_client.evals import MOLMO_SCORING, PUBLIC_EVALS, EvalRef, public_eval
from platform_client.labels import (
    PUBLIC_LABELERS,
    Fail,
    ItemsDelivered,
    Label,
    LabelerRef,
    Outcome,
    Safety,
    Stage,
    StageReached,
    SubtaskAchieved,
    Success,
    label,
)
from platform_client.responses import Scores as PublishedScores
from platform_client.scoring import EpisodeScore, MeanEpisodeScore, Scores, Scoring, episode_score, score
from pydantic import ValidationError

SWEEP = sorted(p for p in (Path(__file__).parent / 'fixtures' / 'molmo_sweep').glob('*/*') if p.is_dir())

# Read from the recording: ten tasks over twenty trials, split irregularly. Four tasks ran three times,
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

CUP = 'pick up the cup.'
KETTLE = 'pick up the kettle.'

# A board's rule that scores the furthest stage, and scores a fail or a safety stop 0.
DROID = EpisodeScore(
    stage={Stage.contact: 0.1, Stage.control: 0.3, Stage.at_target: 0.4}, zero=frozenset({Outcome.fail, Outcome.safety})
)


def _label(*events: object, task: str = CUP, seconds_to_end: float = 240.0, **fields: object) -> Label:
    return Label.model_validate({
        'uid': 'e1',
        'task': task,
        'seconds_to_end': seconds_to_end,
        'events': events,
        **fields,
    })


def _sweep_labels() -> list[Label | None]:
    picked = [label(episode, MOLMO_SCORING.labelers) for episode in SWEEP]
    return [given[1] if given is not None else None for given in picked]


def test_the_sweep_scores_five_of_twenty_in_this_record():
    assert score(MOLMO_SCORING, _sweep_labels()).model_dump(mode='json') == {
        'primary': 0.25,
        'episodes': 20,
        'unscored': 0,
    }


def test_each_task_of_the_sweep_has_the_trials_and_successes_it_recorded():
    given = [given_label for given_label in _sweep_labels() if given_label is not None]
    trials = Counter(given_label.task for given_label in given)
    successes = Counter(given_label.task for given_label in given if given_label.outcome is Outcome.success)
    assert {task: (trials[task], successes[task]) for task in trials} == PER_TASK


@pytest.mark.parametrize('name', PUBLIC_EVALS)
def test_every_public_eval_names_only_public_labelers(name: EvalRef):
    assert set(public_eval(name).scoring.labelers) <= set(PUBLIC_LABELERS)


def test_the_published_scores_are_a_subset_of_the_record():
    record = score(MOLMO_SCORING, _sweep_labels()).model_dump(mode='json')
    assert set(PublishedScores.model_fields) < set(Scores.model_fields)
    published = PublishedScores.model_validate(record).model_dump(mode='json')
    assert published == {name: record[name] for name in PublishedScores.model_fields}


# --- one episode


def test_a_success_scores_the_success_value_and_the_default_rule_scores_nothing_else():
    assert episode_score(_label(Success(seconds=5.0)), EpisodeScore(), None) == 1.0
    assert episode_score(_label(Success(seconds=5.0)), EpisodeScore(success=0.75), None) == 0.75
    reached = _label(
        StageReached(seconds=1.0, stage=Stage.at_target), ItemsDelivered(seconds=2.0, delivered=1), total_items=2
    )
    assert episode_score(reached, EpisodeScore(), None) == 0.0


@pytest.mark.parametrize(
    ('events', 'expected'),
    [
        ((StageReached(seconds=1.0, stage=Stage.reaching),), 0.0),
        ((StageReached(seconds=1.0, stage=Stage.contact),), 0.1),
        ((StageReached(seconds=1.0, stage=Stage.at_target), StageReached(seconds=2.0, stage=Stage.control)), 0.4),
        ((StageReached(seconds=1.0, stage=Stage.at_target), Fail(seconds=2.0)), 0.0),
        ((StageReached(seconds=1.0, stage=Stage.at_target), Safety(seconds=2.0)), 0.0),
        ((StageReached(seconds=1.0, stage=Stage.contact), Success(seconds=2.0)), 1.0),
        ((), 0.0),
    ],
    ids=['reaching', 'contact', 'furthest', 'fail', 'safety', 'success', 'nothing'],
)
def test_the_droid_rule_scores_the_furthest_stage_unless_the_episode_failed(
    events: tuple[object, ...], expected: float
):
    assert episode_score(_label(*events), DROID, None) == expected


def test_a_fail_press_before_the_time_limit_scores_zero_and_one_after_it_scores_the_stage():
    pressed = _label(StageReached(seconds=20.0, stage=Stage.control), Fail(seconds=100.0))
    assert episode_score(pressed, DROID, 120.0) == 0.0
    assert episode_score(pressed, DROID, 90.0) == 0.3


def test_a_success_after_the_time_limit_does_not_count():
    assert episode_score(_label(Success(seconds=50.0)), EpisodeScore(), 40.0) == 0.0


def test_the_largest_partial_score_counts():
    graded = _label(
        StageReached(seconds=1.0, stage=Stage.contact),
        SubtaskAchieved(seconds=2.0, achieved=1, score=0.25),
        ItemsDelivered(seconds=3.0, delivered=1),
        total_items=2,
    )
    rule = EpisodeScore(stage={Stage.contact: 0.1}, use_subtask_score=True, use_item_share=True)
    assert episode_score(graded, rule, None) == 0.5
    assert episode_score(graded, EpisodeScore(stage={Stage.contact: 0.1}, use_subtask_score=True), None) == 0.25


def test_a_stage_mark_with_items_scores_its_stage():
    marked = _label(StageReached.model_validate({'seconds': 1.0, 'stage': 'control', 'items': {'count': 2}}))
    assert episode_score(marked, DROID, None) == 0.3


# --- a run


def test_an_episode_with_no_label_is_unscored_rather_than_failed():
    scores = score(MOLMO_SCORING, [_label(Success(seconds=1.0)), None, None])
    assert (scores.primary, scores.episodes, scores.unscored) == (1.0, 3, 2)


def test_a_run_that_has_no_label_scores_zero_rather_than_dividing_by_it():
    assert score(MOLMO_SCORING, [None]).model_dump(mode='json') == {'primary': 0.0, 'episodes': 1, 'unscored': 1}


def test_a_trial_that_ran_out_of_time_counts_as_a_trial_that_did_not_succeed():
    scores = score(MOLMO_SCORING, [_label(Success(seconds=1.0)), _label(seconds_to_end=60.0)])
    assert (scores.primary, scores.unscored) == (0.5, 0)


def test_per_task_weighs_each_task_the_same():
    labels = [_label(Success(seconds=1.0), task=CUP), _label(task=KETTLE), _label(task=KETTLE), _label(task=KETTLE)]
    by_episode = Scoring(labelers=MOLMO_SCORING.labelers, primary=MeanEpisodeScore())
    by_task = Scoring(labelers=MOLMO_SCORING.labelers, primary=MeanEpisodeScore(per_task=True))
    assert (score(by_episode, labels).primary, score(by_task, labels).primary) == (0.25, 0.5)


def test_the_metric_applies_its_time_limit():
    late = Scoring(labelers=MOLMO_SCORING.labelers, primary=MeanEpisodeScore(time_limit_s=10.0))
    assert score(late, [_label(Success(seconds=5.0)), _label(Success(seconds=20.0))]).primary == 0.5


# --- the parameters


def test_the_scoring_reads_back_from_its_wire_form_with_member_names():
    droid = Scoring(
        labelers=(LabelerRef('labeling-tool'), LabelerRef('rig-console')),
        primary=MeanEpisodeScore(episode_score=DROID, per_task=True),
    )
    wire = droid.model_dump(mode='json')
    assert wire['primary']['kind'] == 'mean_episode_score'
    assert wire['primary']['episode_score']['stage'] == {'contact': 0.1, 'control': 0.3, 'at_target': 0.4}
    assert sorted(wire['primary']['episode_score']['zero']) == ['fail', 'safety']
    assert Scoring.model_validate_json(droid.model_dump_json()) == droid


@pytest.mark.parametrize(
    'change',
    [
        {'labelers': []},
        {'primary': {'kind': 'median'}},
        {'primary': {'kind': 'mean_episode_score', 'episode_score': {'success': 1.5}}},
        {'primary': {'kind': 'mean_episode_score', 'episode_score': {'stage': {'teleported': 0.5}}}},
        {'primary': {'kind': 'mean_episode_score', 'time_limit_s': 0}},
    ],
    ids=['no-labeler', 'unknown-metric', 'success-above-one', 'unknown-stage', 'zero-limit'],
)
def test_the_scoring_refuses_a_value_no_run_can_use(change: dict[str, object]):
    with pytest.raises(ValidationError):
        Scoring.model_validate({**MOLMO_SCORING.model_dump(mode='json'), **change})
