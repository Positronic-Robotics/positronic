import pytest

pytest.importorskip('huggingface_hub')

from positronic.vendors.dreamzero.server import (  # noqa: E402
    DreamZeroSource,
    _checkpoint_id,
    _experiment_name,
)

RUN_DIR = 's3://checkpoints/phail/dreamzero/w22f1_100k_200626/'
CHECKPOINT = RUN_DIR + 'checkpoint-100000'


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        (CHECKPOINT, '100000'),
        ('s3://bucket/exp/checkpoint-005000', '5000'),
        ('GEAR-Dreams/DreamZero-DROID', 'DreamZero-DROID'),
    ],
)
def test_checkpoint_id(path, expected):
    assert _checkpoint_id(path) == expected


@pytest.mark.parametrize(
    ('path', 'expected'), [(CHECKPOINT, 'w22f1_100k_200626'), ('GEAR-Dreams/DreamZero-DROID', 'DreamZero-DROID')]
)
def test_experiment_name(path, expected):
    assert _experiment_name(path) == expected


def test_run_directory_is_served_at_its_latest_step(monkeypatch):
    monkeypatch.setattr(
        'positronic.vendors.dreamzero.server.get_latest_checkpoint', lambda _path, _prefix: 'checkpoint-100000'
    )
    source = DreamZeroSource(model_path=RUN_DIR, backbone='wan2.2')

    assert source.get_models() == ['100000']
    assert source.meta('100000') == {
        'type': 'dreamzero',
        'backbone': 'wan2.2',
        'num_gpus': 1,
        'experiment_name': 'w22f1_100k_200626',
    }
