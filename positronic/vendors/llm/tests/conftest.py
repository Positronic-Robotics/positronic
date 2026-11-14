import secrets

import pytest

pytest.importorskip('pydantic_ai')

from positronic.vendors.llm import client  # noqa: E402


@pytest.fixture(autouse=True)
def anthropic_keys_with_credit(monkeypatch):
    """Each test starts with no pool key marked empty."""
    monkeypatch.setattr(client, '_EMPTY_ANTHROPIC_KEYS', set())


@pytest.fixture
def anthropic_pool(monkeypatch) -> list[str]:
    """Two pool keys; each draw takes the first key that the pool still offers."""
    pytest.importorskip('anthropic')
    pool = ['test-secret-first-org', 'test-secret-second-org']
    monkeypatch.setattr(secrets, 'choice', lambda keys: keys[0])
    monkeypatch.setenv(client.ANTHROPIC_API_KEYS_ENV, ','.join(pool))
    monkeypatch.delenv('ANTHROPIC_CUSTOM_HEADERS', raising=False)
    return pool
