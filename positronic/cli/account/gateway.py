"""The plumbing every platform command shares: a configured client, and refusals a user can read.

The URL's precedence — argument, then environment, then the saved record, then the default platform
— belongs to `platform_client.config`, so a script and a command cannot resolve it differently. A
key or a credential is never an argument: a command line is readable by every process on the box and
lands in shell history.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

from platform_client.client import API_KEY_ENV, API_URL_ENV, CREDENTIAL_ENV, PlatformClient
from platform_client.config import (
    REGISTER_COMMAND,
    api_key_from,
    key_is_given,
    platform_url_from,
    record_if_needed,
    same_platform,
)
from platform_client.errors import PlatformError
from platform_client.ids import Id64
from pydantic import ValidationError

ID = TypeVar('ID', bound=Id64)

__all__ = [
    'API_KEY_ENV',
    'API_URL_ENV',
    'CREDENTIAL_ENV',
    'credential',
    'gateway',
    'one_line',
    'parse_id',
    'refusing_bad_input',
]


def one_line(exc: ValidationError) -> str:
    """Every error of a validation as one line: the field, then what refused it. The value stays out."""
    return '; '.join(f'{".".join(str(part) for part in error["loc"])}: {error["msg"]}' for error in exc.errors())


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
    """A client on the configured platform, reporting a refusal by it as a CLI failure.

    The key comes from the environment, then from the record `register` saved. A saved key is valid
    only on the platform it was minted on, so a command that names another platform is refused.
    """
    record = record_if_needed(os.environ, None, platform_url)
    key = api_key_from(os.environ, None, record)
    if key_required and key is None:
        raise SystemExit(f'no API key: set {API_KEY_ENV}, or run `{REGISTER_COMMAND}`')
    # A misconfigured platform — an empty `--platform-url`, or one the client cannot reach.
    with refusing_bad_input():
        client_ = PlatformClient(platform_url_from(os.environ, platform_url, record), api_key=key)
    if (
        record is not None
        and not key_is_given(os.environ, None)
        and not same_platform(client_.base_url, record.platform_url)
    ):
        raise SystemExit(
            f'the saved key belongs to {record.platform_url}, and this command names {client_.base_url}: '
            f'set {API_KEY_ENV} to a key for that platform, or register there with '
            f'`{REGISTER_COMMAND} --platform-url={client_.base_url}`'
        )
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
        except ValidationError as exc:
            # A 2xx whose body is not the route's response model. The body stays out of the message.
            raise SystemExit(f'the platform answered with a response the client cannot read: {one_line(exc)}') from exc


def credential() -> str:
    """The identity to register with, from the environment."""
    value = os.environ.get(CREDENTIAL_ENV)
    if not value:
        raise SystemExit(f'no credential: set {CREDENTIAL_ENV} to the identity to register with')
    return value


def parse_id(token: object, kind: type[ID]) -> ID:
    """One platform id off the command line, parsed as `kind`."""
    # CLI values are literal-evaluated, so an all-digit id arrives as an int, and reading that as
    # decimal would name a different record. Such an id needs inner quotes to stay text.
    if not isinstance(token, str):
        raise SystemExit(f'an id is hexadecimal; quote one that reads as a number: \'"{token}"\'')
    try:
        return kind.parse(token)
    except ValueError as exc:
        raise SystemExit(f'not an id: {exc}') from exc
