import pytest

from positronic.cfg.policy import production, remote


def test_a_credential_reaches_only_the_endpoint_it_was_set_on():
    cfg = production.override(
        endpoints={'open': remote.override(url='ws://desktop:8000'), 'gated': remote.override(url='wss://gpu:443')},
        **{'endpoints.gated.headers': {'Authorization': 'Bearer t'}},
    )
    open_ep, gated_ep = cfg.instantiate()._policies
    assert open_ep._endpoint._client.headers is None
    assert gated_ep._endpoint._client.headers == {'Authorization': 'Bearer t'}


def test_weights_follow_the_endpoint_names():
    cfg = production.override(
        endpoints={'a': remote.override(url='ws://a:8000'), 'b': remote.override(url='ws://b:8000')}, weights={'b': 3.0}
    )
    assert cfg.instantiate()._weights == [1.0, 3.0]


def test_no_endpoints_is_an_error():
    with pytest.raises(ValueError, match='At least one endpoint'):
        production.instantiate()


def test_weights_naming_an_unknown_endpoint_is_an_error():
    cfg = production.override(endpoints={'a': remote.override(url='ws://a:8000')}, weights={'b': 3.0})
    with pytest.raises(ValueError, match='unknown endpoints'):
        cfg.instantiate()
