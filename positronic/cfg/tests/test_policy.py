import pytest

from positronic.cfg.policy import phail_multiple, production, remote
from positronic.offboard.server import AUTH_HEADER, AUTH_TOKEN_ENV, bearer


def test_a_credential_reaches_only_the_endpoint_it_was_set_on(monkeypatch):
    monkeypatch.setenv(AUTH_TOKEN_ENV, 'tok')
    cfg = phail_multiple.override(**{'endpoints.groot': '.authed_remote', 'endpoints.groot.url': 'https://gpu.example'})
    built = {p._endpoint._client.session_url: p._endpoint._client.headers for p in cfg.instantiate()._policies}
    assert built.pop('wss://gpu.example/api/v1/session') == {AUTH_HEADER: bearer('tok')}
    assert set(built.values()) == {None}


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
