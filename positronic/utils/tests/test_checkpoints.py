import pytest

from positronic.utils import checkpoints as checkpoint_utils


def test_list_checkpoints_lists_numeric_checkpoints_sorted(monkeypatch):
    def fake_ls(_path: str, *, recursive: bool = False):
        assert recursive is False
        return [
            's3://bucket/exp/checkpoints/2/',
            's3://bucket/exp/checkpoints/10/',
            's3://bucket/exp/checkpoints/not-a-checkpoint/',
            's3://bucket/exp/checkpoints/1/',
        ]

    monkeypatch.setattr(checkpoint_utils.pos3, 'ls', fake_ls)
    checkpoints = checkpoint_utils.list_checkpoints('s3://bucket/exp/checkpoints/', prefix='')
    assert checkpoints == ['1', '2', '10']


def test_list_checkpoints_names_the_parent_when_given_one_checkpoint(monkeypatch):
    def fake_ls(_path: str, *, recursive: bool = False):
        return [
            's3://bucket/exp/checkpoints/199999/_CHECKPOINT_METADATA',
            's3://bucket/exp/checkpoints/199999/assets/',
            's3://bucket/exp/checkpoints/199999/params/',
            's3://bucket/exp/checkpoints/199999/train_state/',
        ]

    monkeypatch.setattr(checkpoint_utils.pos3, 'ls', fake_ls)
    with pytest.raises(ValueError, match='parent experiment directory'):
        checkpoint_utils.list_checkpoints('s3://bucket/exp/checkpoints/199999/')


def test_list_checkpoints_reports_a_directory_without_the_marker_generically(monkeypatch):
    """``assets`` sits beside numbered checkpoints, so it never marks a directory as one checkpoint."""

    def fake_ls(_path: str, *, recursive: bool = False):
        return ['s3://bucket/exp/checkpoints/assets/', 's3://bucket/exp/checkpoints/params/']

    monkeypatch.setattr(checkpoint_utils.pos3, 'ls', fake_ls)
    with pytest.raises(ValueError, match=r'No checkpoint found in .*\. Available files:'):
        checkpoint_utils.list_checkpoints('s3://bucket/exp/checkpoints/')
