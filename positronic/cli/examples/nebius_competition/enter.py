"""Enter the Physical AI Competition: show its terms, and accept them for your team.

Usage
  competition-enter                 # show the terms link, their version, and what becomes public
  competition-enter --accept-terms  # the same, then enter your team under those terms

The key comes from POSITRONIC_PLATFORM_API_KEY, which `platform-register` prints. The gateway refuses
terms that are not current, and records the version it serves.
"""

from __future__ import annotations

import argparse
import sys

import httpx
from platform_client.client import API_KEY_ENV, API_URL_ENV, Auth, PlatformClient
from platform_client.errors import PlatformError
from platform_client.routes import API_PREFIX
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

COMPETITION_TERMS = f'{API_PREFIX}/competition.terms'
COMPETITION_ENTER = f'{API_PREFIX}/competition.enter'


class CompetitionTerms(BaseModel):
    """`competition.terms`: the terms a team accepts when it enters."""

    version: str = Field(min_length=1)
    url: str


class EnterRequest(BaseModel):
    """`competition.enter`. `terms_version` is the version this command showed the team."""

    model_config = ConfigDict(extra='forbid')

    terms_version: str = Field(min_length=1)


class CompetitionEntry(BaseModel):
    """The team's entry. `terms_version` is the version the gateway served, and an organizer entry has none."""

    entered_at: AwareDatetime
    terms_version: str | None = None


class CompetitionClient(PlatformClient):
    """The platform client, with the competition's two routes."""

    def terms(self) -> CompetitionTerms:
        return self._get(COMPETITION_TERMS, CompetitionTerms, auth=Auth.NONE)

    def enter(self, request: EnterRequest) -> CompetitionEntry:
        return self._post(COMPETITION_ENTER, request, CompetitionEntry)


def terms_notice(terms: CompetitionTerms) -> str:
    """The terms a team accepts, and what the competition publishes about the team."""
    return (
        f'Terms of the Physical AI Competition, version {terms.version}: {terms.url}\n'
        'The competition publishes your team name and the date you entered when you enter.\n'
        'It publishes your scores and the videos of your runs when they are scored.\n'
        'All of this stays public after the competition.'
    )


def entry_line(entry: CompetitionEntry) -> str:
    day = f'{entry.entered_at:%Y-%m-%d}'
    if entry.terms_version is None:
        return f'entered the competition on {day} as an organizer'
    return f'entered the competition on {day} under the terms version {entry.terms_version}'


def run(client: CompetitionClient, *, accepts_terms: bool) -> None:
    terms = client.terms()
    print(terms_notice(terms))
    if not accepts_terms:
        print('To accept these terms for your team and enter, run again with --accept-terms.')
        return
    if client.api_key is None:
        raise SystemExit(f'set {API_KEY_ENV} to the key `platform-register` prints')
    print(entry_line(client.enter(EnterRequest(terms_version=terms.version))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--accept-terms', action='store_true', help='accept the terms shown for your team, and enter the competition'
    )
    parser.add_argument('--platform-url', default=None, help=f'the platform to enter on, else {API_URL_ENV}')
    args = parser.parse_args()
    try:
        with CompetitionClient(args.platform_url) as client:
            run(client, accepts_terms=args.accept_terms)
    except (PlatformError, httpx.HTTPError, ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
