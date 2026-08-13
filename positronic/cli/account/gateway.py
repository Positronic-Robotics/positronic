"""The plumbing every platform command shares: a configured client, and refusals a user can read.

The URL's precedence — argument, then environment, then the default platform — belongs to the
client, so a script and a command cannot resolve it differently. A key or a credential is never an
argument: a command line is readable by every process on the box and lands in shell history.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from platform_client.client import API_KEY_ENV, API_URL_ENV, CREDENTIAL_ENV, PlatformClient
from platform_client.errors import PlatformError
from platform_client.ids import ApiKey, SubmissionId

__all__ = [
    'API_KEY_ENV',
    'API_URL_ENV',
    'CREDENTIAL_ENV',
    'credential',
    'gateway',
    'parse_submission_id',
    'refusing_bad_input',
]


@contextmanager
def refusing_bad_input() -> Iterator[None]:
    """Report a value the wire types refuse as a CLI refusal, not a traceback.

    Every one of them — a platform URL, an image reference, an eval name, a request model — raises
    `ValueError` naming the value it would not take, which is already the sentence a user needs.
    """
    try:
        yield
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


@contextmanager
def gateway(platform_url: str | None = None, *, key_required: bool = True) -> Iterator[PlatformClient]:
    """A client on the configured platform, reporting a refusal by it as a CLI failure."""
    key = os.environ.get(API_KEY_ENV)
    if key_required and not key:
        raise SystemExit(f'no API key: set {API_KEY_ENV} to the one `positronic account register` printed')
    # A misconfigured platform — an empty `--platform-url`, or one the client cannot reach.
    with refusing_bad_input():
        client_ = PlatformClient(platform_url, api_key=ApiKey(key) if key else None)
    with client_ as client:
        try:
            yield client
        except PlatformError as exc:
            lines = [f'{exc.code.name}: {exc.message}']
            # The platform owns the set of evals, so a name it does not know is answered with the
            # names it does — print them rather than making the user guess a second time.
            if exc.evals is not None:
                lines.append(f'evals on offer: {", ".join(exc.evals)}')
            raise SystemExit('\n'.join(lines)) from exc


def credential() -> str:
    """The identity to register with, from the environment."""
    value = os.environ.get(CREDENTIAL_ENV)
    if not value:
        raise SystemExit(f'no credential: set {CREDENTIAL_ENV} to the identity to register with')
    return value


def parse_submission_id(token: object) -> SubmissionId:
    """One submission id off the command line."""
    # CLI values are literal-evaluated, so an all-digit id arrives as an int, and reading that as
    # decimal would name a different submission. Such an id needs inner quotes to stay text.
    if not isinstance(token, str):
        raise SystemExit(f'submission id is hexadecimal; quote one that reads as a number: \'"{token}"\'')
    try:
        return SubmissionId.parse(token)
    except ValueError as exc:
        raise SystemExit(f'not a submission id: {exc}') from exc
