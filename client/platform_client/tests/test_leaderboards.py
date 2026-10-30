"""The leaderboard contract: every model survives its JSON form, and the wire field names hold."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from platform_client import leaderboards
from platform_client.ids import LeaderboardVersionId, SubmissionId
from platform_client.leaderboards import (
    PRIMARY_METRIC,
    TIME_LIMIT_PLACEHOLDER,
    Baseline,
    CardFact,
    CardLine,
    Caveat,
    CheckpointRef,
    EpisodeColumn,
    EpisodeId,
    EpisodeMedia,
    Estimate,
    EstimateSeries,
    Highlight,
    LeaderboardChartsResponse,
    LeaderboardCounts,
    LeaderboardDescriptor,
    LeaderboardEpisode,
    LeaderboardEpisodesResponse,
    LeaderboardListResponse,
    LeaderboardModelsResponse,
    LeaderboardProgressResponse,
    LeaderboardRef,
    LeaderboardResponse,
    LeaderboardRow,
    LeaderboardSummary,
    LeaderboardTag,
    LeaderboardTask,
    LeaderboardTaskId,
    Link,
    Metric,
    MetricKey,
    ModelCard,
    ModelCheckpoint,
    ModelFamily,
    ModelFamilyRef,
    OutcomeGroup,
    OutcomeKey,
    PairVerdict,
    ProgressBand,
    ProgressPanel,
    RankRange,
    RowCharts,
    RowKey,
    RowProgress,
    RunCommands,
    RunStep,
    TagPosition,
    TeamSubmission,
    TimeLimit,
)
from platform_client.requests import (
    LeaderboardChartsQuery,
    LeaderboardListQuery,
    LeaderboardPartQuery,
    LeaderboardQuery,
)
from platform_client.responses import DataBackedReplayLink
from positronic_evals.evals import EvalRef
from pydantic import BaseModel, ValidationError

AT = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
SUB = SubmissionId(0x1F)
VERSION = LeaderboardVersionId(0x2A)
BOARD = LeaderboardRef('example-board')
TAG = LeaderboardTag('example.site')
STACK = LeaderboardTaskId('stack-the-cubes')
TOWEL = LeaderboardTaskId('fold-the-towel')
SCORE = MetricKey('score')
SPEED = MetricKey('speed')
DONE = OutcomeKey('done')
HALFWAY = OutcomeKey('halfway')
STOPPED = OutcomeKey('stopped')
REPLAY = DataBackedReplayLink(url='https://replays.example/r/token/', data_url='https://replays.example/r/token/r.json')
MEDIA = EpisodeMedia(
    video='https://media.example/s/token/0/phone.mp4',
    replay='https://media.example/s/token/0/replay.rrd',
    cameras={'image.wrist': 'https://media.example/s/token/0/image.wrist.mp4'},
)

TEAM_DESCRIPTOR = LeaderboardDescriptor(
    eval=EvalRef('fake.smoke'),
    metrics=[Metric(key=PRIMARY_METRIC, name='success_rate', decimals=2)],
    ranked_by=PRIMARY_METRIC,
)
TEAM_ROW = LeaderboardRow(
    key=RowKey('team-1f'),
    display_name='demo',
    tag='0ddba7',
    rank=RankRange(best=1, worst=1),
    metrics={PRIMARY_METRIC: Estimate(value=0.75, low=0.75, high=0.75)},
    submission=TeamSubmission(id=SUB, submitted_at=AT, replay=REPLAY),
)
TEAM_BASELINE_ROW = LeaderboardRow(
    key=RowKey('team-a0'),
    display_name='reference',
    tag='9c41de',
    model=ModelFamilyRef('reference-model'),
    rank=None,
    baseline=Baseline(publisher='Example Labs'),
    metrics={PRIMARY_METRIC: Estimate(value=0.5, low=0.5, high=0.5)},
    submission=TeamSubmission(id=SUB, submitted_at=AT),
)
TEAM_LEADERBOARD = LeaderboardResponse(
    leaderboard=BOARD,
    title='Example teams',
    method='example_best_run',
    tags=[TagPosition(tag=TAG, position=1)],
    descriptor=TEAM_DESCRIPTOR,
    computed_at=AT,
    rows=[TEAM_ROW, TEAM_BASELINE_ROW],
)

MODEL_DESCRIPTOR = LeaderboardDescriptor(
    short_name='Example',
    robot='Example arm',
    summary='Tabletop tasks on an example arm',
    setup=Link(text='Example setup', url='https://setup.example/docs/'),
    task_noun='objects',
    episode_noun='runs',
    offers_task_charts=False,
    compared_by='their scores on each task',
    metrics=[
        Metric(
            key=SCORE,
            name='Score',
            unit='% done',
            decimals=1,
            definition=f'Percent of the task done within {TIME_LIMIT_PLACEHOLDER} s.',
            row_definition='Percent of the task done, over every task.',
        ),
        Metric(
            key=SPEED, name='Speed', unit='tasks per minute', decimals=2, definition='Tasks per minute.', axis_max=5.0
        ),
    ],
    ranked_by=SCORE,
    outcomes=[
        OutcomeGroup(key=DONE, name='Done', step_score=1.0),
        OutcomeGroup(key=HALFWAY, name='Halfway', step_score=0.3),
        OutcomeGroup(key=STOPPED, name='Stopped', step_score=0.0),
    ],
    time_limit=TimeLimit(default_s=100, max_s=120, axis='Seconds'),
)
MODEL_ROW = LeaderboardRow(
    key=RowKey('model-a'),
    display_name='Model A',
    model=ModelFamilyRef('model-a'),
    checkpoints=[CheckpointRef('model-a-v2')],
    rank=RankRange(best=1, worst=2),
    metrics={SCORE: Estimate(value=41.25, low=35.0, high=47.875), SPEED: Estimate(value=1.92, low=1.61, high=2.3)},
    episodes=120,
    outcome_shares={DONE: 0.4, HALFWAY: 0.35, STOPPED: 0.25},
    caveats=[Caveat(kind='example_caveat', text='Read the numbers with this note.')],
)
MODEL_B_ROW = LeaderboardRow(
    key=RowKey('model-b'),
    display_name='Model B',
    model=ModelFamilyRef('model-b'),
    rank=RankRange(best=1, worst=2),
    metrics={SCORE: Estimate(value=38.5, low=31.0, high=45.0)},
    episodes=100,
)
MODEL_BASELINE_ROW = LeaderboardRow(
    key=RowKey('reference-person'),
    display_name='Reference person',
    model=ModelFamilyRef('reference-person'),
    rank=None,
    baseline=Baseline(),
    metrics={SCORE: Estimate(value=97.5, low=95.0, high=99.0)},
    episodes=20,
)
MODEL_LEADERBOARD = LeaderboardResponse(
    leaderboard=BOARD,
    title='Example tasks',
    method='example_method',
    tags=[TagPosition(tag=TAG, position=0)],
    descriptor=MODEL_DESCRIPTOR,
    version=VERSION,
    computed_at=AT,
    rows=[MODEL_BASELINE_ROW, MODEL_ROW, MODEL_B_ROW],
    pairs=[PairVerdict(rows=(RowKey('model-a'), RowKey('model-b')), separated=False)],
    tasks=[
        LeaderboardTask(id=STACK, label='Stack the cubes', difficulty='Easy'),
        LeaderboardTask(id=TOWEL, label='Fold the towel', counted=False),
    ],
    counts=LeaderboardCounts(tasks=2, episodes=140),
    highlights=[
        Highlight(
            episode=EpisodeId('1f/3'),
            row=RowKey('model-a'),
            task=STACK,
            caption='Done in 42 s',
            start_s=3.0,
            end_s=23.0,
            media=MEDIA,
        )
    ],
)
SERIES = EstimateSeries(
    value=[None, 10.0, 20.5], low=[None, 5.0, 15.0], high=[None, 15.0, 26.0], margin=[None, 5.0, 5.5]
)
CHARTS = LeaderboardChartsResponse(
    leaderboard=BOARD,
    version=VERSION,
    task=STACK,
    windows=[0.0, 1.0, 2.0],
    rows={
        RowKey('model-a'): RowCharts(
            episodes=12,
            series={SCORE: SERIES, SPEED: SERIES},
            density=[0.0, 1.5, 0.75],
            rank=[None, RankRange(best=1, worst=2), RankRange(best=1, worst=1)],
        ),
        RowKey('model-b'): RowCharts(
            episodes=10,
            series={SCORE: SERIES},
            density=[0.0, 1.0, 0.5],
            rank=[None, RankRange(best=1, worst=2), RankRange(best=2, worst=2)],
        ),
    },
    pairs=[PairVerdict(rows=(RowKey('model-a'), RowKey('model-b')), separated=True)],
)
EPISODE = LeaderboardEpisode(
    id=EpisodeId('1f/3'),
    row=RowKey('model-a'),
    checkpoint=CheckpointRef('model-a-v2'),
    task=STACK,
    outcome=HALFWAY,
    result='Halfway',
    steps=2,
    steps_done=1,
    furthest_stage='Second stage',
    credit=65.0,
    pace=None,
    duration_s=100.0,
    recorded_at=AT,
    media=MEDIA,
)
EPISODES = LeaderboardEpisodesResponse(
    leaderboard=BOARD,
    version=VERSION,
    result=EpisodeColumn(name='Result'),
    pace=EpisodeColumn(name='Pace', decimals=1),
    credit=EpisodeColumn(name='Credit', decimals=0),
    difficulty=EpisodeColumn(name='Difficulty', description='How hard the task is'),
    episodes=[
        EPISODE,
        LeaderboardEpisode(
            id=EpisodeId('2a/0'),
            row=RowKey('model-b'),
            task=TOWEL,
            outcome=HALFWAY,
            result='3/5',
            steps=5,
            steps_done=3,
            pace=4.5,
        ),
    ],
)
PROGRESS = LeaderboardProgressResponse(
    leaderboard=BOARD,
    version=VERSION,
    rows={
        RowKey('model-a'): RowProgress(
            horizon_s=2,
            panels=[
                ProgressPanel(
                    episodes=4,
                    bands=[
                        ProgressBand(outcome=DONE, episodes=[0, 1, 2]),
                        ProgressBand(outcome=STOPPED, episodes=[4, 3, 2]),
                    ],
                    edges=[[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [4.0, 4.0, 4.0]],
                    mean_credit=[0.0, 25.0, 50.0],
                )
            ],
        ),
        RowKey('model-b'): RowProgress(
            horizon_s=2,
            panels=[
                ProgressPanel(
                    task=TOWEL,
                    episodes=2,
                    bands=[
                        ProgressBand(steps_done=1, episodes=[0, 1, 2]),
                        ProgressBand(steps_done=0, episodes=[2, 1, 0]),
                    ],
                    edges=[[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [2.0, 2.0, 2.0]],
                    mean_credit=[0.0, 10.0, 20.0],
                    mean_steps_done=[0.0, 0.5, 1.0],
                    mean_steps_done_at_end=1.0,
                )
            ],
        ),
    },
)
RUN_COMMANDS = RunCommands(
    steps=[
        RunStep(
            label='Start the server', command='docker run example/model-a', source='https://code.example/Dockerfile'
        ),
        RunStep(label='Run the robot', command='example-robot --policy=remote', source='https://code.example/docs.md'),
    ],
    checkpoint=CheckpointRef('model-a-v2'),
)
MODEL_A_CARD = ModelCard(
    lines=[
        CardLine(field='maker', facts=[CardFact(text='Example Labs', source='https://labs.example/')]),
        CardLine(
            field='licence',
            facts=[
                CardFact(text='v1: research only', source='https://labs.example/v1/LICENSE'),
                CardFact(text='v2: open', source='https://labs.example/v2/LICENSE', href='https://x.example'),
            ],
        ),
    ],
    run_commands=RUN_COMMANDS,
)
MODELS_RESPONSE = LeaderboardModelsResponse(
    models=[
        ModelFamily(
            id=ModelFamilyRef('model-a'),
            display_name='Example Labs Model A',
            short_name='Model A',
            categories=['Generalist'],
            checkpoints=[
                ModelCheckpoint(id=CheckpointRef('model-a-v1'), display_name='Model A v1'),
                ModelCheckpoint(id=CheckpointRef('model-a-v2'), display_name='Model A v2'),
            ],
            card=MODEL_A_CARD,
            former_display_names=['Example Model A'],
        ),
        ModelFamily(
            id=ModelFamilyRef('reference-person'),
            display_name='Reference person',
            short_name='Person',
            baseline=True,
            card=ModelCard(run_guide=CardFact(text='Guide', source='https://code.example/guide.md')),
        ),
    ]
)

MODELS: list[BaseModel] = [
    LeaderboardListResponse(tag=TAG),
    LeaderboardListResponse(
        tag=TAG,
        leaderboards=[
            LeaderboardSummary(
                leaderboard=BOARD,
                title='Example tasks',
                method='example_method',
                tags=[TagPosition(tag=TAG, position=0), TagPosition(tag=LeaderboardTag('other.site'), position=3)],
                descriptor=MODEL_DESCRIPTOR,
                caveats=[Caveat(kind='frozen_source', text='A fixed set of episodes.')],
            ),
            LeaderboardSummary(
                leaderboard=LeaderboardRef('example-teams'),
                title='Example teams',
                method='m',
                tags=[TagPosition(tag=TAG, position=1)],
                descriptor=TEAM_DESCRIPTOR,
            ),
        ],
    ),
    MODEL_LEADERBOARD,
    TEAM_LEADERBOARD,
    CHARTS,
    EPISODES,
    PROGRESS,
    MODELS_RESPONSE,
    LeaderboardListQuery(tag=TAG),
    LeaderboardQuery(leaderboard=BOARD),
    LeaderboardPartQuery(leaderboard=BOARD),
    LeaderboardPartQuery(leaderboard=BOARD, version=VERSION),
    LeaderboardChartsQuery(leaderboard=BOARD, version=VERSION, task=STACK),
]


@pytest.mark.parametrize('model', MODELS, ids=lambda m: type(m).__name__)
def test_a_model_round_trips_through_its_json_form(model: BaseModel):
    assert type(model).model_validate(model.model_dump(mode='json')) == model


@pytest.mark.parametrize('model', MODELS, ids=lambda m: type(m).__name__)
def test_a_model_round_trips_through_a_real_json_string(model: BaseModel):
    assert type(model).model_validate_json(model.model_dump_json()) == model


# The wire names of every field. A rename here is a change of the contract that every reader parses.
WIRE_FIELDS: dict[type[BaseModel], set[str]] = {
    Estimate: {'value', 'low', 'high'},
    RankRange: {'best', 'worst'},
    Link: {'text', 'url'},
    Metric: {'key', 'name', 'unit', 'decimals', 'definition', 'row_definition', 'axis_max'},
    OutcomeGroup: {'key', 'name', 'step_score'},
    TimeLimit: {'default_s', 'max_s', 'axis'},
    LeaderboardDescriptor: {
        'eval',
        'short_name',
        'robot',
        'release',
        'summary',
        'setup',
        'task_noun',
        'episode_noun',
        'offers_task_charts',
        'compared_by',
        'metrics',
        'ranked_by',
        'outcomes',
        'time_limit',
        'frozen',
    },
    Caveat: {'kind', 'text'},
    TagPosition: {'tag', 'position'},
    LeaderboardSummary: {'leaderboard', 'title', 'method', 'tags', 'descriptor', 'caveats'},
    LeaderboardListResponse: {'tag', 'leaderboards'},
    Baseline: {'publisher'},
    LeaderboardRow: {
        'key',
        'display_name',
        'tag',
        'model',
        'checkpoints',
        'rank',
        'baseline',
        'metrics',
        'episodes',
        'outcome_shares',
        'submission',
        'caveats',
    },
    TeamSubmission: {'id', 'submitted_at', 'replay'},
    PairVerdict: {'rows', 'separated'},
    LeaderboardTask: {'id', 'label', 'counted', 'difficulty'},
    LeaderboardCounts: {'tasks', 'episodes'},
    EpisodeMedia: {'video', 'replay', 'cameras'},
    Highlight: {'episode', 'row', 'task', 'caption', 'start_s', 'end_s', 'media'},
    LeaderboardResponse: {
        'leaderboard',
        'title',
        'method',
        'tags',
        'descriptor',
        'caveats',
        'version',
        'computed_at',
        'rows',
        'pairs',
        'tasks',
        'counts',
        'highlights',
    },
    EstimateSeries: {'value', 'low', 'high', 'margin'},
    RowCharts: {'episodes', 'series', 'density', 'rank'},
    LeaderboardChartsResponse: {'leaderboard', 'version', 'task', 'windows', 'rows', 'pairs'},
    EpisodeColumn: {'name', 'description', 'decimals'},
    LeaderboardEpisode: {
        'id',
        'row',
        'checkpoint',
        'task',
        'outcome',
        'result',
        'steps',
        'steps_done',
        'furthest_stage',
        'credit',
        'pace',
        'seconds_to_success',
        'duration_s',
        'recorded_at',
        'media',
    },
    LeaderboardEpisodesResponse: {'leaderboard', 'version', 'result', 'pace', 'credit', 'difficulty', 'episodes'},
    ProgressBand: {'outcome', 'steps_done', 'episodes'},
    ProgressPanel: {'task', 'episodes', 'bands', 'edges', 'mean_credit', 'mean_steps_done', 'mean_steps_done_at_end'},
    RowProgress: {'horizon_s', 'panels'},
    LeaderboardProgressResponse: {'leaderboard', 'version', 'rows'},
    CardFact: {'text', 'source', 'href'},
    CardLine: {'field', 'facts'},
    RunStep: {'label', 'command', 'source'},
    RunCommands: {'steps', 'checkpoint'},
    ModelCard: {'lines', 'run_commands', 'run_guide'},
    ModelCheckpoint: {'id', 'display_name'},
    ModelFamily: {
        'id',
        'display_name',
        'short_name',
        'baseline',
        'categories',
        'checkpoints',
        'card',
        'former_display_names',
    },
    LeaderboardModelsResponse: {'models'},
    LeaderboardListQuery: {'tag'},
    LeaderboardQuery: {'leaderboard'},
    LeaderboardPartQuery: {'leaderboard', 'version'},
    LeaderboardChartsQuery: {'leaderboard', 'version', 'task'},
}


@pytest.mark.parametrize(('model', 'fields'), WIRE_FIELDS.items(), ids=lambda m: getattr(m, '__name__', ''))
def test_a_model_carries_exactly_its_wire_fields(model: type[BaseModel], fields: set[str]):
    assert set(model.model_fields) == fields


def test_every_model_of_the_module_has_its_wire_fields_pinned():
    declared = {
        member
        for member in vars(leaderboards).values()
        if isinstance(member, type)
        and issubclass(member, BaseModel)
        and member.__module__ == leaderboards.__name__
        and not member.__name__.startswith('_')
    }
    assert declared <= set(WIRE_FIELDS)


# A team row as the wire carries it.
TEAM_ROW_WIRE = {
    'key': 'team-1f',
    'tag': '0ddba7',
    'display_name': 'demo',
    'rank': {'best': 2, 'worst': 2},
    'metrics': {'primary': {'value': 0.35, 'low': 0.35, 'high': 0.35}},
    'submission': {
        'id': '1f',
        'submitted_at': '2026-03-04T05:06:07Z',
        'replay': {'url': 'https://replays.example/r/token/', 'data_url': 'https://replays.example/r/token/r.json'},
    },
}


def test_a_team_row_reads_every_field_a_site_shows():
    row = LeaderboardRow.model_validate_json(json.dumps(TEAM_ROW_WIRE))
    assert (row.display_name, row.key, row.tag) == ('demo', 'team-1f', '0ddba7')
    assert row.rank == RankRange(best=2, worst=2)
    assert row.metrics[PRIMARY_METRIC] == Estimate(value=0.35, low=0.35, high=0.35)
    assert row.submission is not None
    assert (row.submission.id, row.submission.submitted_at) == (SUB, AT)
    assert row.submission.replay is not None
    assert row.submission.replay.data_url == 'https://replays.example/r/token/r.json'
    assert row.baseline is None


def test_a_team_row_whose_replay_is_not_built_still_names_its_submission():
    wire = {**TEAM_ROW_WIRE, 'submission': {'id': '1f', 'submitted_at': '2026-03-04T05:06:07Z'}}
    row = LeaderboardRow.model_validate_json(json.dumps(wire))
    assert row.submission == TeamSubmission(id=SUB, submitted_at=AT)


@pytest.mark.parametrize('dropped', ['id', 'submitted_at'])
def test_a_submission_needs_its_id_and_its_time(dropped: str):
    submission = {'id': '1f', 'submitted_at': '2026-03-04T05:06:07Z'}
    del submission[dropped]
    with pytest.raises(ValidationError):
        LeaderboardRow.model_validate({**TEAM_ROW_WIRE, 'submission': submission})


def test_a_team_baseline_row_reads_its_publisher_and_its_model():
    wire = {**TEAM_ROW_WIRE, 'rank': None, 'baseline': {'publisher': 'Example Labs'}, 'model': 'reference-model'}
    row = LeaderboardRow.model_validate_json(json.dumps(wire))
    assert row.baseline == Baseline(publisher='Example Labs')
    assert row.model == ModelFamilyRef('reference-model')


def test_a_team_row_with_no_score_is_kept():
    row = LeaderboardRow.model_validate({**TEAM_ROW_WIRE, 'metrics': {}})
    assert PRIMARY_METRIC not in row.metrics


# A model row as the wire carries it.
MODEL_ROW_WIRE = {
    'key': 'model-a',
    'display_name': 'Model A',
    'model': 'model-a',
    'checkpoints': ['model-a-v2'],
    'rank': {'best': 1, 'worst': 3},
    'metrics': {
        'score': {'value': 41.25, 'low': 35.0, 'high': 47.875},
        'speed': {'value': 1.92, 'low': 1.61, 'high': 2.3},
    },
    'episodes': 120,
    'outcome_shares': {'done': 0.4, 'halfway': 0.35, 'stopped': 0.25},
}


def test_a_model_row_reads_its_family_its_checkpoint_and_its_ranges():
    row = LeaderboardRow.model_validate_json(json.dumps(MODEL_ROW_WIRE))
    assert row.model == ModelFamilyRef('model-a')
    assert row.checkpoints == [CheckpointRef('model-a-v2')]
    assert row.rank == RankRange(best=1, worst=3)
    assert row.metrics[SCORE].low < row.metrics[SCORE].value < row.metrics[SCORE].high
    assert all(isinstance(key, MetricKey) for key in row.metrics)
    assert all(isinstance(key, OutcomeKey) for key in row.outcome_shares)
    assert row.episodes == 120
    assert row.submission is None


def test_a_reference_model_row_needs_no_publisher():
    row = LeaderboardRow.model_validate({**MODEL_ROW_WIRE, 'rank': None, 'baseline': {}})
    assert row.baseline == Baseline()


def test_a_row_has_a_rank_exactly_when_it_is_not_a_baseline():
    for refused in ({'rank': None}, {'baseline': {'publisher': 'Example Labs'}}):
        with pytest.raises(ValidationError, match='a row has a rank when it is not a baseline'):
            LeaderboardRow.model_validate({**MODEL_ROW_WIRE, **refused})


def test_a_row_needs_its_rank_stated():
    wire = dict(MODEL_ROW_WIRE)
    del wire['rank']
    with pytest.raises(ValidationError):
        LeaderboardRow.model_validate(wire)


# One `leaderboards.list` entry with values that a later leaderboard can add.
SUMMARY_WIRE = {
    'leaderboard': 'later-board',
    'title': 'A later board',
    'method': 'a_method_this_client_has_no_name_for',
    'tags': [{'tag': 'example.site', 'position': 2}],
    'descriptor': {
        'robot': 'A robot this client has no name for',
        'metrics': [{'key': 'widgets', 'name': 'Widgets', 'unit': 'widgets per hour', 'decimals': 0}],
        'ranked_by': 'widgets',
        'outcomes': [{'key': 'a_new_group', 'name': 'A new outcome group'}],
    },
    'caveats': [{'kind': 'a_caveat_kind_this_client_has_no_name_for', 'text': 'Read with care.'}],
}


def test_a_listed_leaderboard_reads_values_this_client_has_no_name_for():
    summary = LeaderboardSummary.model_validate_json(json.dumps(SUMMARY_WIRE))
    assert summary.method == 'a_method_this_client_has_no_name_for'
    assert summary.descriptor.robot == 'A robot this client has no name for'
    assert summary.descriptor.metrics[0].unit == 'widgets per hour'
    assert summary.caveats[0].kind == 'a_caveat_kind_this_client_has_no_name_for'
    assert summary.descriptor.eval is None


def test_a_listed_leaderboard_reads_past_a_field_this_client_does_not_declare():
    wire = {**SUMMARY_WIRE, 'a_later_field': {'any': 'shape'}}
    wire['descriptor'] = {**SUMMARY_WIRE['descriptor'], 'a_later_fact': 3}
    summary = LeaderboardSummary.model_validate_json(json.dumps(wire))
    assert summary.leaderboard == LeaderboardRef('later-board')


def test_a_listed_leaderboard_gives_its_place_in_each_of_its_tags():
    summary = LeaderboardSummary.model_validate(SUMMARY_WIRE)
    assert summary.tags == [TagPosition(tag=LeaderboardTag('example.site'), position=2)]
    assert isinstance(summary.tags[0].tag, LeaderboardTag)


def test_a_simulated_leaderboard_names_the_eval_it_ranks():
    summary = LeaderboardSummary.model_validate({
        **SUMMARY_WIRE,
        'descriptor': {**SUMMARY_WIRE['descriptor'], 'eval': 'fake.smoke'},
    })
    assert summary.descriptor.eval == EvalRef('fake.smoke')


def test_a_descriptor_ranks_on_one_of_its_own_metrics():
    with pytest.raises(ValidationError, match='which is not one of the metrics'):
        LeaderboardDescriptor.model_validate({**SUMMARY_WIRE['descriptor'], 'ranked_by': 'score'})


def test_a_descriptor_refuses_two_metrics_with_one_key():
    metric = {'key': 'widgets', 'name': 'Widgets', 'decimals': 0}
    with pytest.raises(ValidationError, match='two metrics share a key'):
        LeaderboardDescriptor.model_validate({'metrics': [metric, metric], 'ranked_by': 'widgets'})


def test_a_descriptor_names_at_least_one_metric():
    with pytest.raises(ValidationError):
        LeaderboardDescriptor.model_validate({'metrics': [], 'ranked_by': 'widgets'})


def test_a_range_runs_from_low_to_high():
    assert Estimate(value=1.0, low=1.0, high=1.0).low == 1.0
    with pytest.raises(ValidationError, match='down to'):
        Estimate(value=1.0, low=2.0, high=0.5)


def test_places_run_from_the_best_to_the_worst():
    assert RankRange(best=2, worst=2).worst == 2
    with pytest.raises(ValidationError, match='down to'):
        RankRange(best=3, worst=1)
    with pytest.raises(ValidationError):
        RankRange(best=0, worst=1)


@pytest.mark.parametrize(
    'band', [{'episodes': [1]}, {'outcome': 'done', 'steps_done': 1, 'episodes': [1]}], ids=['neither key', 'both keys']
)
def test_a_progress_band_names_exactly_one_key(band: dict):
    with pytest.raises(ValidationError, match='exactly one of an outcome group and a count of steps'):
        ProgressBand.model_validate(band)


def test_a_progress_band_of_no_steps_done_is_kept():
    assert ProgressBand(steps_done=0, episodes=[2]).steps_done == 0


def test_a_model_card_gives_at_most_one_way_to_run_the_model():
    guide = CardFact(text='Guide', source='https://code.example/guide.md')
    with pytest.raises(ValidationError, match='at most one of the commands and a guide'):
        ModelCard(run_commands=RUN_COMMANDS, run_guide=guide)


def test_a_card_line_holds_at_least_one_fact():
    with pytest.raises(ValidationError):
        CardLine(field='maker', facts=[])


@pytest.mark.parametrize(
    'name_type',
    [
        LeaderboardRef,
        LeaderboardTag,
        ModelFamilyRef,
        CheckpointRef,
        MetricKey,
        OutcomeKey,
        RowKey,
        LeaderboardTaskId,
        EpisodeId,
    ],
)
@pytest.mark.parametrize('value', ['', '   ', 'two words'])
def test_a_name_that_could_never_be_one_is_refused(name_type: type[str], value: str):
    with pytest.raises(ValueError, match=f'not a {name_type.__name__}'):
        name_type(value)


def test_a_name_arrives_as_its_own_type():
    row = LeaderboardRow.model_validate(MODEL_ROW_WIRE)
    assert isinstance(row.key, RowKey)
    assert isinstance(row.model, ModelFamilyRef)
    assert isinstance(row.checkpoints[0], CheckpointRef)
    charts = LeaderboardChartsResponse.model_validate_json(CHARTS.model_dump_json())
    assert all(isinstance(key, RowKey) for key in charts.rows)
    assert isinstance(charts.task, LeaderboardTaskId)


def test_a_leaderboard_with_no_versions_reads_with_no_version():
    response = LeaderboardResponse.model_validate_json(TEAM_LEADERBOARD.model_dump_json())
    assert response.version is None
    assert response.model_dump(mode='json')['version'] is None


def test_a_version_leaves_in_its_hex_wire_form():
    assert MODEL_LEADERBOARD.model_dump(mode='json')['version'] == '2a'
    assert LeaderboardPartQuery(leaderboard=BOARD, version=VERSION).model_dump(mode='json') == {
        'leaderboard': 'example-board',
        'version': '2a',
    }


def test_a_part_names_the_version_it_comes_from():
    wire = json.loads(EPISODES.model_dump_json())
    del wire['version']
    with pytest.raises(ValidationError):
        LeaderboardEpisodesResponse.model_validate(wire)


def test_a_charts_query_refuses_an_empty_task():
    with pytest.raises(ValidationError):
        LeaderboardChartsQuery.model_validate({'leaderboard': 'example-board', 'task': ''})


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', '-inf'])
def test_an_estimate_refuses_a_value_json_cannot_carry(value: float):
    with pytest.raises(ValidationError):
        Estimate(value=value, low=0.0, high=1.0)


@pytest.mark.parametrize('value', [float('nan'), float('inf')], ids=['nan', 'inf'])
def test_a_list_of_floats_refuses_a_value_json_cannot_carry(value: float):
    with pytest.raises(ValidationError):
        EstimateSeries(value=[1.0, value], low=[1.0, 1.0], high=[1.0, 1.0], margin=[0.0, 0.0])
    with pytest.raises(ValidationError):
        RowCharts(episodes=1, density=[value], rank=[None])


def test_every_model_of_the_module_refuses_nan_and_infinity():
    declared = [
        member
        for member in vars(leaderboards).values()
        if isinstance(member, type) and issubclass(member, BaseModel) and member.__module__ == leaderboards.__name__
    ]
    assert declared
    for model in declared:
        assert model.model_config.get('allow_inf_nan') is False, model.__name__


def test_a_descriptor_refuses_two_outcome_groups_with_one_key():
    group = {'key': 'done', 'name': 'Done'}
    with pytest.raises(ValidationError, match='two outcome groups share a key'):
        LeaderboardDescriptor.model_validate({**SUMMARY_WIRE['descriptor'], 'outcomes': [group, group]})


def test_an_outcome_group_needs_its_key():
    with pytest.raises(ValidationError):
        OutcomeGroup.model_validate({'name': 'Done'})


def test_a_text_that_is_not_given_reads_as_none():
    descriptor = LeaderboardDescriptor.model_validate(SUMMARY_WIRE['descriptor'])
    metric = descriptor.metrics[0]
    assert (descriptor.short_name, descriptor.summary, metric.definition) == (None, None, None)
    assert EpisodeColumn(name='Result').description is None
    assert TimeLimit(default_s=10, max_s=10).axis is None


@pytest.mark.parametrize(
    ('build', 'low', 'high'),
    [
        (lambda share: OutcomeGroup(key=DONE, name='Done', step_score=share), 0.0, 1.0),
        (
            lambda share: LeaderboardRow(
                key=RowKey('model-a'), display_name='A', rank=None, baseline=Baseline(), outcome_shares={DONE: share}
            ),
            0.0,
            1.0,
        ),
        (
            lambda share: LeaderboardEpisode(
                id=EpisodeId('1f/0'), row=RowKey('model-a'), task=STACK, outcome=DONE, result='Done', credit=share
            ),
            0.0,
            100.0,
        ),
        (
            lambda share: ProgressPanel(
                episodes=1,
                bands=[ProgressBand(outcome=DONE, episodes=[1, 1])],
                edges=[[0.0, 0.0], [1.0, 1.0]],
                mean_credit=[50.0, share],
            ),
            0.0,
            100.0,
        ),
    ],
    ids=['step_score', 'outcome_shares', 'episode credit', 'mean_credit'],
)
def test_a_share_is_kept_inside_its_scale_and_refused_outside(build, low: float, high: float):
    build(low)
    build(high)
    for outside in (low - 0.01, high + 0.01):
        with pytest.raises(ValidationError):
            build(outside)


def test_a_descriptor_checks_each_rule_with_its_own_validator():
    validators = set(LeaderboardDescriptor.__pydantic_decorators__.model_validators)
    assert validators == {
        '_no_two_metrics_share_a_key',
        '_no_two_outcome_groups_share_a_key',
        '_the_rows_rank_on_one_of_the_metrics',
    }


def put(*path: str | int, value: object) -> Callable[[Any], None]:
    """An edit that sets the value at `path` in a wire form."""

    def edit(wire: Any) -> None:
        for step in path[:-1]:
            wire = wire[step]
        wire[path[-1]] = value

    return edit


def drop(*path: str | int) -> Callable[[Any], None]:
    """An edit that removes the key at `path` from a wire form."""

    def edit(wire: Any) -> None:
        for step in path[:-1]:
            wire = wire[step]
        del wire[path[-1]]

    return edit


def unchanged(wire: Any) -> None:
    del wire


def both(*edits: Callable[[Any], None]) -> Callable[[Any], None]:
    """An edit that applies each of `edits`, in order."""

    def edit(wire: Any) -> None:
        for one in edits:
            one(wire)

    return edit


PANEL = PROGRESS.rows[RowKey('model-a')].panels[0]
STEPS_PANEL = PROGRESS.rows[RowKey('model-b')].panels[0]
FAMILY = MODELS_RESPONSE.models[0]
LIST = LeaderboardListResponse.model_validate(MODELS[1].model_dump(mode='json'))
SUMMARY = LIST.leaderboards[0]

# Each check: the model, a valid fixture, an edit that keeps it just valid, and one that makes it just invalid.
CHECKS = {
    'default time limit within the max': (
        TimeLimit(default_s=100, max_s=120),
        put('default_s', value=120),
        put('default_s', value=121),
    ),
    'clip ends after its start': (MODEL_LEADERBOARD.highlights[0], put('end_s', value=3.01), put('end_s', value=3.0)),
    'clip end before zero': (MODEL_LEADERBOARD.highlights[0], put('end_s', value=3.01), put('end_s', value=-1.0)),
    'chart density at each window': (CHARTS, unchanged, put('rows', 'model-a', 'density', value=[0.0, 1.5])),
    'chart rank at each window': (CHARTS, unchanged, put('rows', 'model-a', 'rank', value=[None, None])),
    'chart series at each window': (
        CHARTS,
        unchanged,
        put('rows', 'model-a', 'series', 'score', value={'value': [1.0], 'low': [1.0], 'high': [1.0], 'margin': [0.0]}),
    ),
    'chart row needs its density': (CHARTS, unchanged, drop('rows', 'model-a', 'density')),
    'band count not negative': (PANEL.bands[0], put('episodes', value=[0, 0, 0]), put('episodes', value=[0, -1, 0])),
    'one place in each tag': (
        SUMMARY,
        put('tags', 1, 'position', value=0),
        put('tags', 1, 'tag', value='example.site'),
    ),
    'no two listed leaderboards share a slug': (
        LIST,
        unchanged,
        put('leaderboards', 1, 'leaderboard', value='example-board'),
    ),
    'each listed leaderboard has a place in the tag': (LIST, unchanged, put('leaderboards', 1, 'tags', value=[])),
    'listed leaderboards in the order of their places': (
        LIST,
        put('leaderboards', 1, 'tags', 0, 'position', value=1),
        put('leaderboards', 1, 'tags', 0, 'position', value=0),
    ),
    'each checkpoint of a row once': (
        MODEL_ROW,
        put('checkpoints', value=['model-a-v1', 'model-a-v2']),
        put('checkpoints', value=['model-a-v2', 'model-a-v2']),
    ),
    'outcome shares add up to at most one': (
        MODEL_ROW,
        put('outcome_shares', 'stopped', value=0.25),
        put('outcome_shares', 'stopped', value=0.26),
    ),
    'a pair is two rows': (
        PairVerdict(rows=(RowKey('a'), RowKey('b')), separated=True),
        unchanged,
        put('rows', value=['a', 'a']),
    ),
    'no two rows share a key': (MODEL_LEADERBOARD, unchanged, put('rows', 2, 'key', value='model-a')),
    'no two tasks share an id': (MODEL_LEADERBOARD, unchanged, put('tasks', 1, 'id', value='stack-the-cubes')),
    'a row names a metric of the descriptor': (
        MODEL_LEADERBOARD,
        unchanged,
        put('rows', 1, 'metrics', 'other', value={'value': 1.0, 'low': 1.0, 'high': 1.0}),
    ),
    'a row names an outcome group of the descriptor': (
        MODEL_LEADERBOARD,
        unchanged,
        put('rows', 1, 'outcome_shares', value={'other': 0.5}),
    ),
    'a pair names rows of the leaderboard': (
        MODEL_LEADERBOARD,
        unchanged,
        put('pairs', 0, 'rows', value=['model-a', 'model-z']),
    ),
    'a highlight names a row of the leaderboard': (
        MODEL_LEADERBOARD,
        unchanged,
        put('highlights', 0, 'row', value='model-z'),
    ),
    'a highlight names a task of the leaderboard': (
        MODEL_LEADERBOARD,
        unchanged,
        put('highlights', 0, 'task', value='other-task'),
    ),
    'no place past the ranked rows': (
        MODEL_LEADERBOARD,
        put('rows', 1, 'rank', value={'best': 2, 'worst': 2}),
        put('rows', 1, 'rank', value={'best': 1, 'worst': 3}),
    ),
    'counts hold no more tasks than the list': (
        MODEL_LEADERBOARD,
        put('counts', 'tasks', value=2),
        put('counts', 'tasks', value=3),
    ),
    'series of one length': (SERIES, unchanged, put('low', value=[None, 5.0])),
    'series limit with a whole estimate or none': (SERIES, unchanged, put('low', 1, value=None)),
    'series range in order': (SERIES, put('low', 2, value=26.0), put('low', 2, value=26.5)),
    'series margin not negative': (SERIES, put('margin', 1, value=0.0), put('margin', 1, value=-0.1)),
    'windows rise': (CHARTS, put('windows', value=[0.0, 1.0, 1.5]), put('windows', value=[0.0, 1.0, 1.0])),
    'windows not negative': (CHARTS, put('windows', 0, value=0.0), put('windows', 0, value=-0.5)),
    'a chart pair names rows of the chart': (CHARTS, unchanged, put('pairs', 0, 'rows', value=['model-a', 'model-z'])),
    'an episode task has a step': (EPISODE, put('steps', value=1), put('steps', value=0)),
    'an episode pace not negative': (EPISODE, put('pace', value=0.0), put('pace', value=-0.1)),
    'a success within its episode': (
        EPISODE,
        put('seconds_to_success', value=100.0),
        put('seconds_to_success', value=100.5),
    ),
    'no two episodes share an id': (EPISODES, unchanged, put('episodes', 1, 'id', value='1f/3')),
    'a panel has an episode': (PANEL, unchanged, put('episodes', value=0)),
    'a panel has a band': (PANEL, unchanged, put('bands', value=[])),
    'a panel has a second': (PANEL, unchanged, put('mean_credit', value=[])),
    'a panel bands one kind of key': (
        PANEL,
        unchanged,
        put('bands', 1, value={'steps_done': 0, 'episodes': [4, 3, 2]}),
    ),
    'a panel names each band once': (PANEL, unchanged, put('bands', 1, 'outcome', value='done')),
    'one edge more than the bands': (PANEL, unchanged, put('edges', value=[[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]])),
    'each panel series at each second': (PANEL, unchanged, put('mean_credit', value=[0.0, 25.0])),
    'the bands hold each episode once': (
        PANEL,
        both(put('bands', 0, 'episodes', value=[1, 1, 2]), put('bands', 1, 'episodes', value=[3, 3, 2])),
        put('bands', 0, 'episodes', value=[0, 1, 3]),
    ),
    'the top edge holds every episode': (PANEL, unchanged, put('edges', 2, value=[4.0, 4.0, 3.0])),
    'the edges stack': (PANEL, put('edges', 1, 2, value=4.0), put('edges', 1, 2, value=4.5)),
    'an edge not negative': (PANEL, unchanged, put('edges', 0, 0, value=-0.1)),
    'a mean of steps not negative': (
        STEPS_PANEL,
        put('mean_steps_done', 0, value=0.0),
        put('mean_steps_done', 0, value=-0.1),
    ),
    'no two panels share a task': (
        PROGRESS.rows[RowKey('model-b')],
        unchanged,
        put('panels', value=[STEPS_PANEL.model_dump(mode='json')] * 2),
    ),
    'no panel past the horizon': (
        PROGRESS.rows[RowKey('model-b')],
        put('horizon_s', value=2),
        put('horizon_s', value=1),
    ),
    'each card line once': (MODEL_A_CARD, unchanged, put('lines', 1, 'field', value='maker')),
    'no two checkpoints of a family share an id': (FAMILY, unchanged, put('checkpoints', 0, 'id', value='model-a-v2')),
    'the commands run a checkpoint of the family': (
        FAMILY,
        put('card', 'run_commands', 'checkpoint', value='model-a-v1'),
        put('card', 'run_commands', 'checkpoint', value='model-a-v9'),
    ),
    'no two families share an id': (MODELS_RESPONSE, unchanged, put('models', 1, 'id', value='model-a')),
}


@pytest.mark.parametrize(('model', 'valid', 'invalid'), CHECKS.values(), ids=CHECKS.keys())
def test_a_value_just_inside_a_check_passes_and_one_just_outside_it_is_refused(
    model: BaseModel, valid: Callable[[Any], None], invalid: Callable[[Any], None]
):
    kept, refused = (json.loads(model.model_dump_json()) for _ in range(2))
    valid(kept)
    invalid(refused)
    type(model).model_validate(kept)
    with pytest.raises(ValidationError):
        type(model).model_validate(refused)
