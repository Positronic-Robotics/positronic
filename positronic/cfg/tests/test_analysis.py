from pathlib import Path

import pytest
from eval_vocabulary.outcome import OUTCOME, Outcome

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
def test_success_duration_uses_the_recording_clock(held_ns, timeline):
    signal = DummySignal(
        [[0, 0], [100_000_000, 1], [100_000_000 + held_ns, 2]], [0.0, 1.0, 1.0], timelines=(timeline, 'tick')
    )
    episode = EpisodeContainer({'stacking_success': signal})
    assert analysis.success(episode) == (held_ns >= 250_000_000)
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
        'eval.successful_items': 2,
        'eval.total_items': 2,
        'eval.outcome': 'Success',
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


def _attended(outcome: Outcome | None, items: int) -> EpisodeContainer:
    """A 10 s attended episode that delivered ``items`` of 2, with ``outcome`` in ``eval.outcome``."""
    episode = {
        'state': DummySignal([0, 10_000_000_000], [0, 1], timelines=(HARNESS_WORLD,)),
        'checkpoint': 'test',
        'model': 'test',
        'variant': '',
        'eval.object': 'cup',
        eval_keys.DURATION: 10.0,
        'eval.successful_items': items,
        'eval.total_items': 2,
    }
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


@pytest.mark.parametrize('outcome', [Outcome.DISCARDED, Outcome.UNSCORED])
def test_an_unscored_outcome_leaves_out_a_sim_episode(outcome):
    signal = DummySignal([0, 10_000_000_000], [1.0, 1.0], timelines=(HARNESS_WORLD,))
    sim = {'stacking_success': signal, 'checkpoint': 'test', 'model': 'test'}
    scored = EpisodeContainer(sim)
    discarded = EpisodeContainer({**sim, OUTCOME: outcome})

    assert analysis.unified_success_bool(discarded) is None
    assert analysis.scored_episodes([scored, discarded]) == [scored]
    assert analysis.checkpoint_table().group_fn([scored, discarded])['count'] == 1
