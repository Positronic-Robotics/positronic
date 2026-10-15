from pathlib import Path

import pytest
from eval_vocabulary.outcome import OUTCOME, SUCCESSFUL_ITEMS, TOTAL_ITEMS, Outcome

from pimm.time import RECEIVED_WORLD
from positronic import keys
from positronic.cfg import analysis
from positronic.cfg.analysis import ckpt
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD
from positronic.eval import keys as eval_keys
from positronic.server.positronic_server import app_state_restored, configure_tables


@pytest.mark.parametrize(
    ('flat', 'groups'),
    [
        (analysis.episodes_table, {'checkpoints': analysis.checkpoint_table}),
        (analysis.stacking_episodes_table, {'checkpoints': analysis.stacking_checkpoint_table}),
        (analysis.phail_episodes_table, {'leaderboard': analysis.phail_leaderboard}),
    ],
)
def test_every_server_group_key_and_filter_is_a_flat_table_column(flat, groups):
    """A group table's View link filters the flat table on its group keys and filters, so each is a flat column.

    A hidden column counts, so `model` on stacking and `equipment` on phail filter without a visible column.
    """
    group_tables = {name: cfg.instantiate() for name, cfg in groups.items()}
    with app_state_restored():
        configure_tables(
            root='',
            cache_dir=Path(),
            ep_table_cfg=flat(),
            group_tables=group_tables,
            home_page=None,
            max_resolution=64,
            max_hz=0,
            layout=None,
        )


def test_ckpt_act_resolves_comment_example_path():
    ep = EpisodeContainer({
        'inference.policy.type': 'act',
        'inference.policy.checkpoint_path': 'full_ft_q/act/031225/checkpoints/300000/pretrained_model/',
    })
    assert ckpt(ep) == '300000'


def test_ckpt_remote_resolves_checkpoint_id():
    ep = EpisodeContainer({'inference.policy.type': 'remote', 'inference.policy.server.checkpoint_id': '50000'})
    assert ckpt(ep) == '50000'


def test_ckpt_remote_resolves_checkpoint_path():
    ep = EpisodeContainer({
        'inference.policy.type': 'remote',
        'inference.policy.server.checkpoint_path': '/checkpoints/experiment/checkpoint-30000',
    })
    assert ckpt(ep) == '30000'


def test_ckpt_remote_resolves_lerobot_pretrained_model_path():
    ep = EpisodeContainer({
        'inference.policy.type': 'remote',
        'inference.policy.server.checkpoint_path': 'checkpoints/050000/pretrained_model',
    })
    assert ckpt(ep) == '050000'


@pytest.mark.parametrize('held_ns', [200_000_000, 300_000_000])
@pytest.mark.parametrize('timeline', [HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME])
def test_success_time_uses_the_recording_clock(held_ns, timeline):
    signal = DummySignal(
        [[0, 0], [100_000_000, 1], [100_000_000 + held_ns, 2]], [0.0, 1.0, 1.0], timelines=(timeline, 'tick')
    )
    episode = EpisodeContainer({'stacking_success': signal})
    expected = (100_000_000 + held_ns) / 1e9 if held_ns >= 250_000_000 else None
    assert analysis.success_time(episode) == expected


@pytest.mark.parametrize(
    'timelines',
    [
        (HARNESS_WORLD,),
        (RECEIVED_WORLD,),
        (RECORDED_TIME,),
        (RECEIVED_WORLD, RECORDED_TIME),
        (HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME),
    ],
)
def test_analysis_derives_duration_and_rates_on_the_recording_clock(timelines):
    signal = DummySignal(
        [[0] * len(timelines), [10_000_000_000 * (i + 1) for i in range(len(timelines))]], [0, 1], timelines=timelines
    )
    episode = EpisodeContainer({
        'state': signal,
        SUCCESSFUL_ITEMS: 2,
        TOTAL_ITEMS: 2,
        OUTCOME: Outcome.SUCCESS,
        'checkpoint': 'test',
        'model': 'test',
    })
    assert analysis.uph(episode) == 720
    assert analysis.unified_uph(episode) == 720
    assert analysis.phail_uph(episode) == 720
    assert analysis._baseline_uph(episode, items=2) == 720
    assert analysis._effective_duration(eval_keys.DURATION, episode) == 10
    table = analysis.checkpoint_table()
    assert table.group_fn([episode])['UPH'] == 720


@pytest.mark.parametrize('timeline', [HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME])
def test_item_counting_samples_the_recording_clock(timeline):
    times = [0, 100_000_000, 200_000_000, 300_000_000, 400_000_000]
    episode = EpisodeContainer({
        keys.TASK: 'test task',
        keys.TARGET_GRIP: DummySignal(times, [0, 1, 1, 0, 0], timelines=(timeline,)),
        keys.EE_POSE: DummySignal(
            times, [[0, 0, 0], [0, 0, 0], [0.2, 0, 0.1], [0.3, 0, 0.1], [0.3, 0, 0.1]], timelines=(timeline,)
        ),
    })
    assert analysis.calculate_units(episode) == 1


@pytest.mark.parametrize('outcome', [Outcome.DISCARDED, Outcome.UNSCORED, None])
def test_an_episode_nobody_scored_is_neither_a_success_nor_a_failure(outcome):
    signal = DummySignal([0, 10_000_000_000], [0, 1], timelines=(HARNESS_WORLD,))
    statics = {} if outcome is None else {OUTCOME: outcome}
    unscored = EpisodeContainer({'state': signal, 'checkpoint': 'test', 'model': 'test', **statics})
    failed = EpisodeContainer({'state': signal, 'checkpoint': 'test', 'model': 'test', OUTCOME: Outcome.FAIL})

    assert analysis.unified_success_bool(unscored) is None
    assert analysis.unified_success_bool(failed) is False
    assert analysis.checkpoint_table().group_fn([unscored, failed])['failures'] == 1
    assert analysis.phail_status(unscored) == analysis.UNSCORED
    assert analysis.phail_status(failed) == 'Fail'


def _attended(outcome: Outcome | None, items: int | None, **statics: object) -> EpisodeContainer:
    """A 10 s episode that delivered ``items`` of 2, with ``outcome`` in ``eval.outcome`` and ``statics`` beside it.

    ``items=None`` records no item counts.
    """
    episode = {
        'state': DummySignal([0, 10_000_000_000], [0, 1], timelines=(HARNESS_WORLD,)),
        'checkpoint': 'test',
        'model': 'test',
        'variant': '',
        eval_keys.OBJECT: 'cup',
        eval_keys.DURATION: 10.0,
        **statics,
    }
    if items is not None:
        episode |= {SUCCESSFUL_ITEMS: items, TOTAL_ITEMS: 2}
    if outcome is not None:
        episode[OUTCOME] = outcome
    container = EpisodeContainer(episode)
    return EpisodeContainer({**episode, 'status': analysis.phail_status(container)})


def test_every_aggregate_leaves_out_an_episode_nobody_scored():
    scored = [_attended(Outcome.SUCCESS, 2), _attended(Outcome.FAIL, 0)]
    unscored = [_attended(Outcome.DISCARDED, 2), _attended(Outcome.UNSCORED, 2), _attended(None, 2)]

    assert analysis.scored_episodes(scored + unscored) == scored
    checkpoint = analysis.checkpoint_table().group_fn(scored + unscored)
    assert checkpoint == analysis.checkpoint_table().group_fn(scored)
    assert (checkpoint['count'], checkpoint['failures'], checkpoint['success_rate']) == (2, 1, 50)
    assert checkpoint['UPH'] == pytest.approx(2 / (20 / 3600))
    assert checkpoint['MTBF'] == 20
    leaderboard = analysis.phail_leaderboard().group_fn(scored + unscored)
    assert leaderboard == analysis.phail_leaderboard().group_fn(scored)
    assert (leaderboard['count'], leaderboard['completion'], leaderboard['MTBF']) == (2, 50, 20 / 60)


# The statics of an episode that the policy ended before its time limit, with no verdict.
GAVE_UP = {
    eval_keys.ENDED_BY: eval_keys.ENDED_BY_POLICY,
    eval_keys.POLICY_END: eval_keys.PolicyEnd.GIVE_UP,
    eval_keys.TERMINATED: True,
}
COMPLETED = {**GAVE_UP, eval_keys.POLICY_END: eval_keys.PolicyEnd.COMPLETE}


def test_a_give_up_counts_as_a_failure_in_every_aggregate_and_adds_no_items():
    succeeded, gave_up = _attended(Outcome.SUCCESS, 2), _attended(None, None, **GAVE_UP)

    assert analysis.unified_success_bool(gave_up) is False
    assert analysis.phail_status(gave_up) == 'Fail'
    checkpoint = analysis.checkpoint_table().group_fn([succeeded, gave_up])
    assert (checkpoint['count'], checkpoint['failures'], checkpoint['success_rate']) == (2, 1, 50)
    assert checkpoint['items_done'] == 100
    leaderboard = analysis.phail_leaderboard().group_fn([succeeded, gave_up])
    assert (leaderboard['count'], leaderboard['completion'], leaderboard['MTBF']) == (2, 100, 20 / 60)


def test_a_complete_in_an_env_that_ends_on_success_counts_as_a_failure():
    completed = _attended(None, None, **COMPLETED, **{eval_keys.ENV_ENDS_ON_SUCCESS: True})

    assert analysis.unified_success_bool(completed) is False
    assert analysis.phail_status(completed) == 'Fail'
    assert analysis.checkpoint_table().group_fn([completed])['failures'] == 1


def test_a_complete_with_no_other_fact_is_unscored_and_left_out_of_every_aggregate():
    succeeded, completed = _attended(Outcome.SUCCESS, 2), _attended(None, None, **COMPLETED)

    assert analysis.unified_success_bool(completed) is None
    assert analysis.phail_status(completed) == analysis.UNSCORED
    assert analysis.scored_episodes([succeeded, completed]) == [succeeded]
    assert analysis.checkpoint_table().group_fn([succeeded, completed])['count'] == 1


@pytest.mark.parametrize('success', [True, False])
def test_an_env_report_scores_an_episode_with_no_verdict(success):
    reported = _attended(None, None, **{eval_keys.SUCCESS: success, eval_keys.TERMINATED: True})

    assert analysis.unified_success_bool(reported) is success
    assert analysis.scored_episodes([reported]) == [reported]
    assert analysis.unified_units_display(reported) == str(int(success))
    assert analysis.unified_success_rate(reported) == 100 * int(success)


def test_a_sim_episode_with_an_env_report_and_no_signal_counts_in_the_checkpoint_table():
    """A MolmoSpaces episode records `eval.success`, no `eval.outcome`, no item counts and no `stacking_success`."""
    reported = {eval_keys.SUCCESS: True, eval_keys.TERMINATED: True}
    succeeded, failed = (
        _attended(None, None, **reported),
        _attended(None, None, **{**reported, eval_keys.SUCCESS: False}),
    )

    row = analysis.checkpoint_table().group_fn([succeeded, failed])

    assert (row['count'], row['failures'], row['success_rate'], row['items_done']) == (2, 1, 50, None)
    assert row['UPH'] == pytest.approx(1 / (20 / 3600))


def test_an_episode_with_no_verdict_and_no_fact_raises_nothing_and_shows_no_rate():
    bare = _attended(None, None)

    assert analysis.phail_leaderboard().group_fn([bare])['count'] == 0
    assert (analysis.unified_units_display(bare), analysis.unified_success_rate(bare)) == ('-', None)


@pytest.mark.parametrize('table', [analysis.checkpoint_table, analysis.stacking_checkpoint_table])
def test_a_row_with_no_scored_episode_shows_no_rate_and_no_uph_rather_than_a_zero(table):
    """A zero would read as a checkpoint that failed every run, which is the opposite of unscored."""
    signal = DummySignal([0, 10_000_000_000], [0.0, 0.0], timelines=(HARNESS_WORLD,))
    bare = EpisodeContainer({
        'stacking_success': signal,
        'checkpoint': 'test',
        'model': 'test',
        'success': None,
        'success_time': None,
        'units': 0,
        'max_stacking_success': 0.0,
    })
    group = table.instantiate()

    row = group.group_fn([bare])

    assert (row['success_rate'], row['UPH']) == (None, None)
    assert group.format_table['success_rate'].default == '-'
    assert group.format_table['UPH'].default == '-'


def test_a_stacking_row_counts_every_run_and_shows_the_signal_without_a_verdict():
    signal = DummySignal([0, 10_000_000_000], [0.5, 0.8], timelines=(HARNESS_WORLD,))
    sim = {'stacking_success': signal, 'checkpoint': 'test', 'model': 'test', 'max_stacking_success': 0.8}
    unscored = EpisodeContainer({**sim, 'success': None, 'success_time': None, 'units': None})
    failed = EpisodeContainer({**sim, OUTCOME: Outcome.FAIL, 'success': False, 'success_time': None, 'units': 0})

    row = analysis.stacking_checkpoint_table.instantiate().group_fn([unscored, failed])

    assert (row['count'], row['scored'], row['failures'], row['success_rate']) == (2, 1, 1, 0)
    assert row['avg_max_success'] == pytest.approx(0.8)


def test_a_discarded_episode_with_item_counts_shows_no_units_no_rate_and_no_uph():
    discarded = _attended(Outcome.DISCARDED, 2)

    assert analysis.unified_units_display(discarded) == '-'
    assert analysis.unified_success_rate(discarded) is None
    assert analysis.unified_uph(discarded) is None


def test_a_stacking_signal_alone_scores_nothing_and_still_shows():
    signal = DummySignal([0, 10_000_000_000], [1.0, 1.0], timelines=(HARNESS_WORLD,))
    sim = EpisodeContainer({'stacking_success': signal, 'checkpoint': 'test', 'model': 'test'})

    assert analysis.unified_success_bool(sim) is None
    assert analysis.scored_episodes([sim]) == []
    assert analysis.checkpoint_table().group_fn([sim])['count'] == 0
    assert (analysis.unified_units_display(sim), analysis.unified_uph(sim)) == ('-', None)
    assert (analysis.success_units(sim), analysis.uph_sim(sim)) == (None, None)
    assert analysis.max_stacking_success(sim) == 1.0
    assert analysis.success_time(sim) == 10.0


@pytest.mark.parametrize(('outcome', 'signal_value'), [(Outcome.FAIL, 1.0), (Outcome.SUCCESS, 0.0)])
def test_a_scored_outcome_decides_a_sim_episode_against_its_signal(outcome, signal_value):
    signal = DummySignal([0, 10_000_000_000], [signal_value, signal_value], timelines=(HARNESS_WORLD,))
    sim = EpisodeContainer({'stacking_success': signal, 'checkpoint': 'test', 'model': 'test', OUTCOME: outcome})
    succeeded = outcome == Outcome.SUCCESS

    assert analysis.unified_success_bool(sim) is succeeded
    assert analysis.success_units(sim) == int(succeeded)
    assert analysis.unified_units_display(sim) == str(int(succeeded))
    assert analysis.unified_success_rate(sim) == 100 * int(succeeded)
    assert analysis.uph_sim(sim) is None
    assert analysis.checkpoint_table().group_fn([sim])['failures'] == int(not succeeded)
