from unittest.mock import Mock

import pytest

pytest.importorskip('lerobot', minversion='0.4')

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig

from positronic.offboard import keys as offboard_keys
from positronic.policy.observation import TASK_FIELD
from positronic.vendors.lerobot import server
from positronic.vendors.lerobot.policy import LerobotModel, warm_observation


@pytest.fixture
def unloaded(monkeypatch):
    """Checkpoints ``41`` and ``42``, and a model built without downloading or warming either."""
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path, prefix='': ['41', '42'])
    monkeypatch.setattr(server.pos3, 'download', lambda path: path)
    monkeypatch.setattr(server, 'LerobotModel', lambda _path, _device, extra_meta: Mock(meta=lambda: extra_meta))
    monkeypatch.setattr(server, 'warm_observation', Mock())
    monkeypatch.setattr(server, 'warmup', Mock())


@pytest.mark.parametrize('configured, expected', [(None, '42'), ('41', '41')])
def test_the_configured_checkpoint_is_served_else_the_latest(unloaded, configured, expected):
    model = server.lerobot_model(checkpoints_dir='s3://bucket/exp', checkpoint=configured, device='cpu')
    assert model.meta()[offboard_keys.CHECKPOINT_ID] == expected


def test_a_configured_checkpoint_the_directory_lacks_is_refused(unloaded):
    with pytest.raises(ValueError, match='not found'):
        server.lerobot_model(checkpoints_dir='s3://bucket/exp', checkpoint='43', device='cpu')


def test_warmup_observation_matches_the_features_the_policy_declares():
    state, camera = 'observation.state', 'observation.images.left'
    config = ACTConfig(
        input_features={
            state: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 320)),
        },
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    obs = warm_observation(config)

    assert obs[state].shape == (8,)
    # Declared channels-first, handed over channels-last the way a session takes it.
    assert obs[camera].shape == (224, 320, 3)
    assert obs[TASK_FIELD] == ''


def test_session_owner_isolated_from_probes_and_other_episodes():
    model = object.__new__(LerobotModel)
    model._policy = Mock()
    model._preprocessor = Mock()
    model._postprocessor = Mock()
    model._session_id = 'episode'
    with pytest.raises(RuntimeError, match='another session'):
        model({}, session_id='other')
    model.end_session('probe')
    model._policy.reset.assert_not_called()
    model.end_session('episode')
    model._policy.reset.assert_called_once()
    assert model._session_id is None
