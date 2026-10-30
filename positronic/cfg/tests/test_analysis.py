from pathlib import Path

import pytest

from pimm.time import RECEIVED_WORLD
from positronic import keys
from positronic.cfg import analysis
from positronic.cfg.analysis import ckpt
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
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


@pytest.mark.parametrize(
    ('model_id', 'name'),
    [
        ('azure:gpt-4.1', 'gpt4.1'),
        ('openai-responses:gpt-4.1-2025-04-14', 'gpt4.1'),
        ('openai-responses:gpt-6-astra', 'gpt6-astra'),
        ('anthropic:claude-opus-5-5', 'opus5.5'),
        ('anthropic:claude-sonnet-5-5-20260929', 'sonnet5.5'),
        ('anthropic:claude-fable-5-1', 'fable5.1'),
        ('gpt-4.1', 'gpt4.1'),
    ],
)
def test_an_llm_episode_shows_the_short_name_of_its_configured_model(model_id, name):
    ep = EpisodeContainer({
        f'{policy_keys.POLICY_META}.{policy_keys.TYPE}': policy_keys.LLM_TYPE,
        f'{policy_keys.POLICY_META}.{policy_keys.MODEL}': model_id,
    })
    assert analysis.model(ep) == name


def test_an_unknown_llm_model_shows_its_id_without_provider_and_date():
    assert analysis.short_model_name('openai-chat:new-model-2027-01-01') == 'new-model'


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
