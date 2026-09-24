from unittest.mock import Mock

import pytest

pytest.importorskip('lerobot', minversion='0.4')

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig

from positronic.policy.observation import TASK_FIELD
from positronic.vendors.lerobot.policy import LerobotModel, warm_observation
from positronic.vendors.lerobot.server import LerobotSource


@pytest.mark.parametrize('configured, expected', [(None, '42'), ('41', '41')])
def test_the_configured_checkpoint_is_served_else_the_latest(monkeypatch, configured, expected):
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path, prefix='': ['41', '42'])
    assert LerobotSource('s3://bucket/exp', checkpoint=configured, device='cpu').checkpoint_id() == expected


def test_a_configured_checkpoint_the_directory_lacks_is_refused(monkeypatch):
    monkeypatch.setattr('positronic.utils.checkpoints.list_checkpoints', lambda _path, prefix='': ['41', '42'])
    with pytest.raises(ValueError, match='not found'):
        LerobotSource('s3://bucket/exp', checkpoint='43', device='cpu').checkpoint_id()


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
