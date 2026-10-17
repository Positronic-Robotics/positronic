"""Enter your team in the Physical AI Competition.

Usage
  competition-enter                           # show what becomes public, and where to read how your data is used
  competition-enter --yes                     # the same, then enter your team
  competition-enter --accept-terms=<version>  # the same, then enter your team and accept that version

The key comes from POSITRONIC_PLATFORM_API_KEY, which `platform-register` prints. The entry asks for no
terms. The platform refuses the first submission of a team until the team accepts them.
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
COMPETITION_ACCEPT_TERMS = f'{API_PREFIX}/competition.accept_terms'


class CompetitionTerms(BaseModel):
    """`competition.terms`: the current terms, which a team accepts with its first submission."""

    version: str = Field(min_length=1)
    url: str


class EnterRequest(BaseModel):
    """`competition.enter`. The entry asks for no terms."""

    model_config = ConfigDict(extra='forbid')


class AcceptTermsRequest(BaseModel):
    """`competition.accept_terms`. `terms_version` is the version this command showed the team."""

    model_config = ConfigDict(extra='forbid')

    terms_version: str = Field(min_length=1)


class CompetitionEntry(BaseModel):
    """The team's entry. `terms_version` is the version the team accepted last, and None before its first
    submission. `organizer` marks an entry that the organizers made, which accepts no terms."""

    entered_at: AwareDatetime
    terms_version: str | None = None
    organizer: bool = False


class CompetitionClient(PlatformClient):
    """The platform client, with the competition's three routes."""

    def terms(self) -> CompetitionTerms:
        return self._get(COMPETITION_TERMS, CompetitionTerms, auth=Auth.NONE)

    def enter(self, request: EnterRequest) -> CompetitionEntry:
        return self._post(COMPETITION_ENTER, request, CompetitionEntry)

    def accept_terms(self, request: AcceptTermsRequest) -> CompetitionEntry:
        return self._post(COMPETITION_ACCEPT_TERMS, request, CompetitionEntry)


def entry_notice(terms: CompetitionTerms) -> str:
    """The terms, what the competition publishes about the team, and where to read how it uses the team's data."""
    return (
        f'Terms of the Physical AI Competition, version {terms.version}: {terms.url}\n'
        f'Your team accepts them with its first submission: run again with --accept-terms={terms.version}.\n'
        'The competition publishes your team name and the date you entered when you enter.\n'
        'It publishes your scores and the videos of your runs when they are scored.\n'
        'All of this stays public after the competition.\n'
        'The section "Your data" of the terms tells how the organizers use your data.'
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
    terms = client.terms()
    print(entry_notice(terms))
    if not confirmed and accepted_version is None:
        print('To enter your team, run again with --yes.')
        return
    if client.api_key is None:
        raise SystemExit(f'set {API_KEY_ENV} to the key `platform-register` prints')
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
    try:
        with CompetitionClient(args.platform_url) as client:
            run(client, confirmed=args.yes, accepted_version=args.accept_terms)
    except (PlatformError, httpx.HTTPError, ValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
