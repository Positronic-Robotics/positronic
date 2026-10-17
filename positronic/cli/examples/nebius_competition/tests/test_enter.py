"""`enter.py` over a stub gateway: what it shows, the entry and the acceptance it sends, and the key it needs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
import pytest
from platform_client.client import AUTH_HEADER
from platform_client.enums import ErrorCode
from platform_client.errors import ApiErrorBody, ErrorEnvelope
from platform_client.ids import ApiKey

from positronic.cli.examples.nebius_competition import enter
from positronic.cli.examples.nebius_competition.enter import (
    COMPETITION_ACCEPT_TERMS,
    COMPETITION_ENTER,
    COMPETITION_TERMS,
    CompetitionClient,
    CompetitionEntry,
    CompetitionTerms,
)

BASE = 'http://gateway.test'
TERMS = CompetitionTerms(version='2026-10-14', url='https://terms.example/terms')
KEY = ApiKey('pk_live_team')


@dataclass
class StubGateway:
    """The two competition routes, answering scripted bodies and recording what reached them."""

    entry: CompetitionEntry = field(
        default_factory=lambda: CompetitionEntry.model_validate({'entered_at': '2026-10-14T09:30:00Z'})
    )
    refusal: httpx.Response | None = None
    requests: list[httpx.Request] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == COMPETITION_TERMS:
            return httpx.Response(200, json=TERMS.model_dump(mode='json'))
        if request.url.path == COMPETITION_ENTER:
            return self.refusal or httpx.Response(200, json=self.entry.model_dump(mode='json'))
        if request.url.path == COMPETITION_ACCEPT_TERMS:
            accepted = self.entry.model_copy()
            accepted.terms_version = TERMS.version
            return httpx.Response(200, json=accepted.model_dump(mode='json'))
        raise AssertionError(f'the command reached a route it has no business at: {request.url.path}')

    def client(self, api_key: ApiKey | None = KEY) -> CompetitionClient:
        transport = httpx.Client(base_url=BASE, transport=httpx.MockTransport(self.handle))
        return CompetitionClient(client=transport, api_key=api_key)

    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]


def test_without_the_flag_the_command_shows_what_becomes_public_and_enters_nothing(capsys):
    gateway = StubGateway()

    with gateway.client() as client:
        enter.run(client, confirmed=False, accepted_version=None)

    shown = capsys.readouterr().out
    assert TERMS.version in shown and TERMS.url in shown and 'Your data' in shown and 'videos of your runs' in shown
    assert 'first submission' in shown and '--yes' in shown and gateway.paths() == [COMPETITION_TERMS]


def test_the_flag_enters_with_no_terms(capsys):
    gateway = StubGateway()

    with gateway.client() as client:
        enter.run(client, confirmed=True, accepted_version=None)

    assert gateway.paths() == [COMPETITION_TERMS, COMPETITION_ENTER]
    # rules-allow: hardcoded-keys — the test pins the request's wire spelling.
    assert json.loads(gateway.requests[1].content) == {}
    assert gateway.requests[1].headers[AUTH_HEADER] == f'Bearer {KEY}'
    assert AUTH_HEADER not in gateway.requests[0].headers
    assert capsys.readouterr().out.endswith('entered the competition on 2026-10-14\n')


def test_accept_terms_enters_and_sends_the_version_the_team_approved(capsys):
    gateway = StubGateway()

    with gateway.client() as client:
        enter.run(client, confirmed=False, accepted_version=TERMS.version)

    assert gateway.paths() == [COMPETITION_TERMS, COMPETITION_ENTER, COMPETITION_ACCEPT_TERMS]
    sent = enter.AcceptTermsRequest.model_validate_json(gateway.requests[2].content)
    assert sent.terms_version == TERMS.version
    assert gateway.requests[2].headers[AUTH_HEADER] == f'Bearer {KEY}'
    assert f'accepted the terms version {TERMS.version}' in capsys.readouterr().out


def test_accept_terms_without_a_key_stops_before_the_entry():
    gateway = StubGateway()

    with gateway.client(api_key=None) as client, pytest.raises(SystemExit):
        enter.run(client, confirmed=False, accepted_version=TERMS.version)

    assert gateway.paths() == [COMPETITION_TERMS]


def test_accept_terms_sends_the_approved_version_and_not_a_newer_current_one():
    gateway = StubGateway()

    with gateway.client() as client:
        enter.run(client, confirmed=False, accepted_version='2026-01-01')

    sent = enter.AcceptTermsRequest.model_validate_json(gateway.requests[2].content)
    assert sent.terms_version == '2026-01-01' != TERMS.version


def test_the_flag_without_a_key_stops_before_the_entry():
    gateway = StubGateway()

    with gateway.client(api_key=None) as client, pytest.raises(SystemExit) as raised:
        enter.run(client, confirmed=True, accepted_version=None)

    assert 'platform-register' in str(raised.value) and gateway.paths() == [COMPETITION_TERMS]


def test_a_refusal_reaches_the_team(monkeypatch, capsys):
    message = 'this gateway is not running'
    envelope = ErrorEnvelope(error=ApiErrorBody(code=ErrorCode.bad_request, message=message))
    gateway = StubGateway(refusal=httpx.Response(400, json=envelope.model_dump(mode='json')))
    monkeypatch.setattr(enter, 'CompetitionClient', lambda platform_url: gateway.client())
    monkeypatch.setattr('sys.argv', ['competition-enter', '--yes'])

    with pytest.raises(SystemExit) as raised:
        enter.main()

    assert raised.value.code == 1 and message in capsys.readouterr().err


def test_an_organizer_entry_says_so(capsys):
    organizer = CompetitionEntry.model_validate({'entered_at': '2026-10-14T09:30:00Z', 'organizer': True})
    gateway = StubGateway(entry=organizer)

    with gateway.client() as client:
        enter.run(client, confirmed=True, accepted_version=None)

    assert 'as an organizer' in capsys.readouterr().out


def test_the_models_keep_the_gateway_s_field_names():
    """The gateway declares its own copy of these models, so each field name is a contract."""
    # rules-allow: hardcoded-keys — these sets pin the wire names of the gateway's separate copy.
    assert set(CompetitionTerms.model_fields) == {'version', 'url'}
    assert set(enter.EnterRequest.model_fields) == set()
    assert set(enter.AcceptTermsRequest.model_fields) == {'terms_version'}
    assert set(CompetitionEntry.model_fields) == {'entered_at', 'terms_version', 'organizer'}
