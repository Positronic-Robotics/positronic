"""Enter your team in the Physical AI Competition.

Usage
  competition-enter                           # show what becomes public about your team
  competition-enter --yes                     # the same, then enter your team
  competition-enter --accept-terms=<version>  # the same, then enter your team and accept that version

The key comes from POSITRONIC_PLATFORM_API_KEY, else from the record `positronic account register` saves.
The entry asks for no terms. The platform refuses the first submission of a team until the team accepts them,
and its refusal names the version to accept.
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx
from platform_client.client import API_KEY_ENV, API_URL_ENV, PlatformClient
from platform_client.config import REGISTER_COMMAND, api_key_from, platform_url_from, record_if_needed
from platform_client.errors import PlatformError
from platform_client.routes import API_PREFIX
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

COMPETITION_ENTER = f'{API_PREFIX}/competition.enter'
COMPETITION_ACCEPT_TERMS = f'{API_PREFIX}/competition.accept_terms'


class EnterRequest(BaseModel):
    """`competition.enter`. The entry asks for no terms."""

    model_config = ConfigDict(extra='forbid')


class AcceptTermsRequest(BaseModel):
    """`competition.accept_terms`. `terms_version` is the version the team read."""

    model_config = ConfigDict(extra='forbid')

    terms_version: str = Field(min_length=1)


class CompetitionEntry(BaseModel):
    """The team's entry. `terms_version` is the version the team accepted last, and None before its first
    submission. `organizer` marks an entry that the organizers made, which accepts no terms."""

    entered_at: AwareDatetime
    terms_version: str | None = None
    organizer: bool = False


class CompetitionClient(PlatformClient):
    """The platform client, with the competition's two routes."""

    def enter(self, request: EnterRequest) -> CompetitionEntry:
        return self._post(COMPETITION_ENTER, request, CompetitionEntry)

    def accept_terms(self, request: AcceptTermsRequest) -> CompetitionEntry:
        return self._post(COMPETITION_ACCEPT_TERMS, request, CompetitionEntry)


ENTRY_NOTICE = (
    'The competition publishes your team name and the date you entered when you enter.\n'
    'It publishes your scores and the videos of your runs when they are scored.\n'
    'All of this stays public after the competition.'
)


def entry_line(entry: CompetitionEntry) -> str:
    day = f'{entry.entered_at:%Y-%m-%d}'
    if entry.organizer:
        return f'entered the competition on {day} as an organizer'
    if entry.terms_version is None:
        return f'entered the competition on {day}'
    return f'entered the competition on {day}, and accepted the terms version {entry.terms_version}'


def run(client: CompetitionClient, *, confirmed: bool, accepted_version: str | None) -> None:
    """`accepted_version` is the terms version that the team approved, or None. The gateway refuses a version
    that is not current, so a version published after the approval is never accepted in its place."""
    print(ENTRY_NOTICE)
    if not confirmed and accepted_version is None:
        print('To join, run: competition-enter --yes')
        return
    if client.api_key is None:
        raise SystemExit(f'no API key: set {API_KEY_ENV}, or run `{REGISTER_COMMAND}`')
    entry = client.enter(EnterRequest())
    if accepted_version is not None:
        entry = client.accept_terms(AcceptTermsRequest(terms_version=accepted_version))
    print(entry_line(entry))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--yes', action='store_true', help='enter your team in the competition')
    parser.add_argument(
        '--accept-terms',
        metavar='VERSION',
        default=None,
        help='enter your team, and accept this version of the terms for your team',
    )
    parser.add_argument('--platform-url', default=None, help=f'the platform to enter on, else {API_URL_ENV}')
    args = parser.parse_args()
    enters = args.yes or args.accept_terms is not None
    record = record_if_needed(os.environ, args.platform_url, key_required=enters)
    key = api_key_from(os.environ, record) if enters else None
    try:
        with CompetitionClient(platform_url_from(os.environ, args.platform_url, record), api_key=key) as client:
            run(client, confirmed=args.yes, accepted_version=args.accept_terms)
    except (PlatformError, httpx.HTTPError, ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
