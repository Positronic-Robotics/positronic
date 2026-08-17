"""A stub platform for the commands that talk to one.

Shared by `positronic account` and `positronic eval`, which drive the same client: the
transport is stubbed and everything above it — request shaping, auth headers, response parsing — is
the real thing.
"""

import json

import configuronic as cfn
import httpx
import pytest
from platform_client.client import PlatformClient
from platform_client.ids import ApiKey

from positronic.cli.account import gateway as gateway_module

BASE = 'http://gateway.test'
KEY = 'pk_live_secret'
ID = '5f3a91c2b7d40e18'
AT = '2026-03-04T05:06:07Z'


class StubPlatform:
    """Records what a command sent, and the URL and key it was sent with, and answers a canned payload."""

    def __init__(self):
        self.status = 200
        self.payload: object = {}
        self.seen: httpx.Request | None = None
        self.base_url: str | None = None
        self.api_key: ApiKey | None = None

    def answer(self, payload: object, *, status: int = 200) -> None:
        self.payload, self.status = payload, status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen = request
        return httpx.Response(self.status, json=self.payload)

    @property
    def request(self) -> httpx.Request:
        assert self.seen is not None, 'no request reached the platform'
        return self.seen

    @property
    def body(self) -> dict:
        return json.loads(self.request.content)


@pytest.fixture
def platform(monkeypatch):
    """A configured environment whose client rides a stub transport."""
    stub = StubPlatform()

    def build(base_url: str | None = None, *, api_key: ApiKey | None = None) -> PlatformClient:
        stub.base_url, stub.api_key = base_url, api_key
        transport = httpx.MockTransport(stub)
        return PlatformClient(client=httpx.Client(base_url=base_url or BASE, transport=transport), api_key=api_key)

    monkeypatch.setattr(gateway_module, 'PlatformClient', build)
    monkeypatch.setenv(gateway_module.API_URL_ENV, BASE)
    monkeypatch.setenv(gateway_module.API_KEY_ENV, KEY)
    monkeypatch.setenv(gateway_module.CREDENTIAL_ENV, 'token')
    return stub


@pytest.fixture
def run_command():
    """Run one command as the CLI does: parsed arguments overridden onto its config, then instantiated."""

    def run(command: cfn.Config, **kwargs):
        return command.override(**kwargs).instantiate()

    return run
