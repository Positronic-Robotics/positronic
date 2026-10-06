from pathlib import Path

import pytest

from positronic.cfg import analysis
from positronic.cfg.analysis import ckpt
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.tests.utils import DummySignal
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


RECORDED_PREFIXES = pytest.mark.parametrize('prefix', policy_keys.POLICY_META_PREFIXES)


@RECORDED_PREFIXES
def test_ckpt_act_resolves_comment_example_path(prefix):
    ep = EpisodeContainer({
        f'{prefix}.type': 'act',
        f'{prefix}.checkpoint_path': 'full_ft_q/act/031225/checkpoints/300000/pretrained_model/',
    })
    assert ckpt(ep) == '300000'


@RECORDED_PREFIXES
def test_ckpt_remote_resolves_checkpoint_id(prefix):
    ep = EpisodeContainer({f'{prefix}.type': 'remote', f'{prefix}.server.checkpoint_id': '50000'})
    assert ckpt(ep) == '50000'


@RECORDED_PREFIXES
def test_ckpt_remote_resolves_checkpoint_path(prefix):
    ep = EpisodeContainer({
        f'{prefix}.type': 'remote',
        f'{prefix}.server.checkpoint_path': '/checkpoints/experiment/checkpoint-30000',
    })
    assert ckpt(ep) == '30000'


@RECORDED_PREFIXES
def test_ckpt_remote_resolves_lerobot_pretrained_model_path(prefix):
    ep = EpisodeContainer({
        f'{prefix}.type': 'remote',
        f'{prefix}.server.checkpoint_path': 'checkpoints/050000/pretrained_model',
    })
    assert ckpt(ep) == '050000'


@pytest.mark.parametrize('held_ns', [200_000_000, 300_000_000])
def test_success_duration_uses_recorded_coordinate(held_ns):
    signal = DummySignal(
        [[0, 0], [100_000_000, 1], [100_000_000 + held_ns, 2]], [0.0, 1.0, 1.0], timelines=(RECORDED_TIME, 'tick')
    )
    episode = EpisodeContainer({'stacking_success': signal})
    assert analysis.success(episode) == (held_ns >= 250_000_000)
    expected = (100_000_000 + held_ns) / 1e9 if held_ns >= 250_000_000 else None
    assert analysis.success_time(episode) == expected


@RECORDED_PREFIXES
def test_model_and_phail_model_read_the_server_type(prefix):
    ep = EpisodeContainer({f'{prefix}.type': 'remote', f'{prefix}.server.type': 'groot'})
    assert analysis.model(ep) == 'groot'
    assert analysis.phail_model(ep) == analysis.PHAIL_MODEL_DISPLAY['groot']
