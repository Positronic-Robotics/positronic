from pathlib import Path

import pytest

from positronic.cfg import analysis
from positronic.cfg.analysis import ckpt
from positronic.dataset.episode import EpisodeContainer
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
