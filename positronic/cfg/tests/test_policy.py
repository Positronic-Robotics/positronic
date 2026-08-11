import pytest

from positronic.cfg.policy import bearer_headers
from positronic.offboard.server import AUTH_HEADER, AUTH_TOKEN_ENV


def test_bearer_headers_carries_the_environment_token(monkeypatch):
    monkeypatch.setenv(AUTH_TOKEN_ENV, 'a-token')
    # Spelled out rather than built with ``bearer()``: this is the wire format a served endpoint parses,
    # and sharing the builder with the server would leave the two free to drift together.
    assert bearer_headers.instantiate() == {AUTH_HEADER: 'Bearer a-token'}


@pytest.mark.parametrize('token', [None, ''])
def test_bearer_headers_refuses_to_send_no_token(monkeypatch, token):
    if token is None:
        monkeypatch.delenv(AUTH_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(AUTH_TOKEN_ENV, token)
    with pytest.raises(ValueError, match=AUTH_TOKEN_ENV):
        bearer_headers.instantiate()
