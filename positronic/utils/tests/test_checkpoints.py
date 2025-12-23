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
