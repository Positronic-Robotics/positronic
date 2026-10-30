"""The leaderboard contract: every model survives its JSON form, and the wire field names hold."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from platform_client import leaderboards
from platform_client.ids import LeaderboardVersionId, SubmissionId
from platform_client.leaderboards import (
    PRIMARY_METRIC,
    Baseline,
    CardFact,
    CardLine,
    Caveat,
    CheckpointRef,
    EntrantCharts,
    EntrantId,
    EntrantProgress,
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
    Link,
    Metric,
    ModelCard,
    ModelCheckpoint,
    ModelFamily,
    ModelFamilyRef,
    OutcomeGroup,
    PairVerdict,
    ProgressBand,
    ProgressPanel,
    RankRange,
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
    entrant=EntrantId('0ddba7'),
    display_name='demo',
    rank=RankRange(best=1, worst=1),
    metrics={PRIMARY_METRIC: Estimate(value=0.75, low=0.75, high=0.75)},
    submission=TeamSubmission(id=SUB, submitted_at=AT, replay=REPLAY),
)
TEAM_BASELINE_ROW = LeaderboardRow(
    entrant=EntrantId('9c41de'),
    display_name='reference',
    model=ModelFamilyRef('reference-model'),
    rank=None,
    baseline=Baseline(publisher='Example Labs'),
    metrics={PRIMARY_METRIC: Estimate(value=0.5, low=0.5, high=0.5)},
    submission=TeamSubmission(id=SUB, submitted_at=AT),
)
TEAM_LEADERBOARD = LeaderboardResponse(
    leaderboard=BOARD,
    title='Example qualifier',
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
            key='score',
            name='Score',
            unit='% done',
            decimals=1,
            definition='Percent of the task done within {seconds} s.',
            row_definition='Percent of the task done, over every task.',
        ),
        Metric(
            key='speed', name='Speed', unit='tasks per minute', decimals=2, definition='Tasks per minute.', axis_max=5.0
        ),
    ],
    ranked_by='score',
    outcomes=[
        OutcomeGroup(name='Done', credit=1.0),
        OutcomeGroup(name='Halfway', credit=0.3),
        OutcomeGroup(name='Stopped', credit=0.0),
    ],
    time_limit=TimeLimit(default_s=100, max_s=120, axis='Seconds'),
)
MODEL_ROW = LeaderboardRow(
    entrant=EntrantId('model-a'),
    display_name='Model A',
    model=ModelFamilyRef('model-a'),
    checkpoints=[CheckpointRef('model-a-v2')],
    rank=RankRange(best=1, worst=2),
    metrics={'score': Estimate(value=41.25, low=35.0, high=47.875), 'speed': Estimate(value=1.92, low=1.61, high=2.3)},
    episodes=120,
    outcome_shares={'Done': 0.4, 'Halfway': 0.35, 'Stopped': 0.25},
    caveats=[Caveat(kind='example_caveat', text='Read the numbers with this note.')],
)
MODEL_BASELINE_ROW = LeaderboardRow(
    entrant=EntrantId('reference-person'),
    display_name='Reference person',
    model=ModelFamilyRef('reference-person'),
    rank=None,
    baseline=Baseline(),
    metrics={'score': Estimate(value=97.5, low=95.0, high=99.0)},
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
    rows=[MODEL_BASELINE_ROW, MODEL_ROW],
    pairs=[PairVerdict(entrants=(EntrantId('model-a'), EntrantId('model-b')), separated=False)],
    tasks=[
        LeaderboardTask(id='stack-the-cubes', label='Stack the cubes', difficulty='Easy'),
        LeaderboardTask(id='fold-the-towel', label='Fold the towel', counted=False),
    ],
    counts=LeaderboardCounts(tasks=2, episodes=140),
    highlights=[
        Highlight(
            episode=EpisodeId('1f/3'),
            entrant=EntrantId('model-a'),
            task='stack-the-cubes',
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
    task='stack-the-cubes',
    windows=[0.0, 1.0, 2.0],
    entrants={
        EntrantId('model-a'): EntrantCharts(
            episodes=12,
            series={'score': SERIES, 'speed': SERIES},
            density=[0.0, 1.5, 0.75],
            rank=[None, RankRange(best=1, worst=2), RankRange(best=1, worst=1)],
        )
    },
    pairs=[PairVerdict(entrants=(EntrantId('model-a'), EntrantId('model-b')), separated=True)],
)
EPISODE = LeaderboardEpisode(
    id=EpisodeId('1f/3'),
    entrant=EntrantId('model-a'),
    checkpoint=CheckpointRef('model-a-v2'),
    task='stack-the-cubes',
    outcome='Halfway',
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
            entrant=EntrantId('model-b'),
            task='Example object',
            outcome='Partway',
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
    entrants={
        EntrantId('model-a'): EntrantProgress(
            horizon_s=2,
            panels=[
                ProgressPanel(
                    episodes=4,
                    bands=[
                        ProgressBand(outcome='Done', episodes=[0, 1, 2]),
                        ProgressBand(outcome='Stopped', episodes=[4, 3, 2]),
                    ],
                    edges=[[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [4.0, 4.0, 4.0]],
                    mean_credit=[0.0, 25.0, 50.0],
                )
            ],
        ),
        EntrantId('model-b'): EntrantProgress(
            horizon_s=2,
            panels=[
                ProgressPanel(
                    task='Example object',
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
            card=ModelCard(
                lines=[
                    CardLine(field='maker', facts=[CardFact(text='Example Labs', source='https://labs.example/')]),
                    CardLine(
                        field='licence',
                        facts=[
                            CardFact(text='v1: research only', source='https://labs.example/v1/LICENSE'),
                            CardFact(
                                text='v2: open', source='https://labs.example/v2/LICENSE', href='https://x.example'
                            ),
                        ],
                    ),
                ],
                run_commands=RUN_COMMANDS,
            ),
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
            LeaderboardSummary(leaderboard=BOARD, title='Example qualifier', method='m', descriptor=TEAM_DESCRIPTOR),
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
    LeaderboardChartsQuery(leaderboard=BOARD, version=VERSION, task='stack-the-cubes'),
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
    OutcomeGroup: {'name', 'credit'},
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
        'entrant',
        'display_name',
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
    PairVerdict: {'entrants', 'separated'},
    LeaderboardTask: {'id', 'label', 'counted', 'difficulty'},
    LeaderboardCounts: {'tasks', 'episodes'},
    EpisodeMedia: {'video', 'replay', 'cameras'},
    Highlight: {'episode', 'entrant', 'task', 'caption', 'start_s', 'end_s', 'media'},
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
    EntrantCharts: {'episodes', 'series', 'density', 'rank'},
    LeaderboardChartsResponse: {'leaderboard', 'version', 'task', 'windows', 'entrants', 'pairs'},
    EpisodeColumn: {'name', 'description', 'decimals'},
    LeaderboardEpisode: {
        'id',
        'entrant',
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
    EntrantProgress: {'horizon_s', 'panels'},
    LeaderboardProgressResponse: {'leaderboard', 'version', 'entrants'},
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


# A team row as the wire carries it. A site reads the name, the entrant, the rank, the score, the submission, the time,
# the replay and the publisher of a baseline.
TEAM_ROW_WIRE = {
    'entrant': '0ddba7',
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
    assert (row.display_name, row.entrant) == ('demo', '0ddba7')
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


# A model row as the wire carries it: a family, the checkpoint the leaderboard measured, a range on each metric and a
# range of places.
MODEL_ROW_WIRE = {
    'entrant': 'model-a',
    'display_name': 'Model A',
    'model': 'model-a',
    'checkpoints': ['model-a-v2'],
    'rank': {'best': 1, 'worst': 3},
    'metrics': {
        'score': {'value': 41.25, 'low': 35.0, 'high': 47.875},
        'speed': {'value': 1.92, 'low': 1.61, 'high': 2.3},
    },
    'episodes': 120,
    'outcome_shares': {'Done': 0.4, 'Halfway': 0.35, 'Stopped': 0.25},
}


def test_a_model_row_reads_its_family_its_checkpoint_and_its_ranges():
    row = LeaderboardRow.model_validate_json(json.dumps(MODEL_ROW_WIRE))
    assert row.model == ModelFamilyRef('model-a')
    assert row.checkpoints == [CheckpointRef('model-a-v2')]
    assert row.rank == RankRange(best=1, worst=3)
    assert row.metrics['score'].low < row.metrics['score'].value < row.metrics['score'].high
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
        'outcomes': [{'name': 'A new outcome group'}],
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
    'band', [{'episodes': [1]}, {'outcome': 'Done', 'steps_done': 1, 'episodes': [1]}], ids=['neither key', 'both keys']
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
    'name_type', [LeaderboardRef, LeaderboardTag, ModelFamilyRef, CheckpointRef, EntrantId, EpisodeId]
)
@pytest.mark.parametrize('value', ['', '   ', 'two words'])
def test_a_name_that_could_never_be_one_is_refused(name_type: type[str], value: str):
    with pytest.raises(ValueError, match=f'not a {name_type.__name__}'):
        name_type(value)


def test_a_name_arrives_as_its_own_type():
    row = LeaderboardRow.model_validate(MODEL_ROW_WIRE)
    assert isinstance(row.entrant, EntrantId)
    assert isinstance(row.model, ModelFamilyRef)
    assert isinstance(row.checkpoints[0], CheckpointRef)
    charts = LeaderboardChartsResponse.model_validate_json(CHARTS.model_dump_json())
    assert all(isinstance(entrant, EntrantId) for entrant in charts.entrants)


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
