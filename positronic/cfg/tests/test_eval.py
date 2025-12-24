from positronic.cfg.eval import ckpt
from positronic.dataset.episode import EpisodeContainer


def test_ckpt_act_resolves_comment_example_path():
    ep = EpisodeContainer({
        'inference.policy.type': 'act',
        'inference.policy.checkpoint_path': 'full_ft_q/act/031225/checkpoints/300000/pretrained_model/',
    })
    assert ckpt(ep) == 'full_ft_q\\031225\\300000'


def test_ckpt_openpi_resolves_comment_example_path_from_checkpoint_path():
    ep = EpisodeContainer({
        'inference.policy.type': 'openpi',
        'inference.policy.checkpoint_path': 'full_ft/openpi/pi05_positronic_lowmem/061025/119999',
    })
    assert ckpt(ep) == 'full_ft\\061025\\119999'


def test_ckpt_openpi_resolves_comment_example_path_from_server_directory_when_checkpoint_path_missing():
    ep = EpisodeContainer({
        'inference.policy.type': 'openpi',
        'inference.policy.server.directory': 'full_ft/openpi/pi05_positronic_lowmem/061025/119999',
    })
    assert ckpt(ep) == 'full_ft\\061025\\119999'
