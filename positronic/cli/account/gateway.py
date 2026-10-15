"""The plumbing every platform command shares: a configured client, and refusals a user can read.

The URL's precedence — argument, then environment, then the saved record, then the default platform
— belongs to `platform_client.config`, so a script and a command cannot resolve it differently. A
key or a credential is never an argument: a command line is readable by every process on the box and
lands in shell history.
"""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from enum import IntEnum
from typing import TypeVar

import httpx
from platform_client.client import API_KEY_ENV, API_URL_ENV, CREDENTIAL_ENV, PlatformClient
from platform_client.config import REGISTER_COMMAND, api_key_from, platform_url_from, record_if_needed
from platform_client.enums import ErrorCode
from platform_client.errors import PlatformError
from platform_client.ids import Id64
from pydantic import ValidationError

ID = TypeVar('ID', bound=Id64)

__all__ = [
    'API_KEY_ENV',
    'API_URL_ENV',
    'CREDENTIAL_ENV',
    'CommandFailed',
    'Fault',
    'credential',
    'fault_of',
    'gateway',
    'one_line',
    'parse_id',
    'refusing_bad_input',
]


class Fault(IntEnum):
    """Whose fault a failed command is. The value is the exit status of `positronic`."""

    CALLER = 1
    USAGE = 2
    PLATFORM = 3


class CommandFailed(SystemExit):
    """A failure a user can read, and its fault. Outside `positronic` it exits with status 1."""

    def __init__(self, message: str, fault: Fault) -> None:
        super().__init__(message)
        self.fault = fault


def fault_of(exc: PlatformError) -> Fault:
    """The platform's fault for a 5xx or a code that names its own failure, else the caller's."""
    if exc.http_status >= 500 or exc.code in (ErrorCode.internal_error, ErrorCode.upstream_unavailable):
        return Fault.PLATFORM
    return Fault.CALLER


def one_line(exc: ValidationError) -> str:
    """Every validation error on one line with its field and message."""
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
        raise CommandFailed(str(exc), Fault.USAGE) from exc


@contextmanager
def gateway(
    platform_url: str | None = None, *, key_required: bool = True, refusal_as_json: bool = False
) -> Iterator[PlatformClient]:
    """A client on the configured platform, reporting a refusal by it as a CLI failure.

    One client, one platform: the key comes from the environment, then from the record `register`
    saved. `register` sends no key, so a registration naming its platform reads no record.
    `refusal_as_json` prints a refusal's error envelope on stdout in place of the text on stderr.
    """
    record = record_if_needed(os.environ, platform_url, key_required=key_required)
    key = api_key_from(os.environ, record) if key_required else None
    if key_required and key is None:
        raise CommandFailed(f'no API key: set {API_KEY_ENV}, or run `{REGISTER_COMMAND}`', Fault.CALLER)
    # A misconfigured platform — an empty `--platform-url`, or one the client cannot reach.
    with refusing_bad_input():
        client_ = PlatformClient(platform_url_from(os.environ, platform_url, record), api_key=key)
    with client_ as client:
        try:
            yield client
        except PlatformError as exc:
            if refusal_as_json:
                print(exc.envelope.model_dump_json(indent=2))
                raise SystemExit(fault_of(exc)) from exc
            lines = [f'{exc.code.name}: {exc.message}']
            lines += [f'  {key}: {json.dumps(value)}' for key, value in exc.details.items()]
            raise CommandFailed('\n'.join(lines), fault_of(exc)) from exc
        except ValidationError as exc:
            # A 2xx whose body is not the route's response model. The body stays out of the message.
            codes = '; '.join(error['type'] for error in exc.errors())
            raise CommandFailed(
                f'the platform answered with a response the client cannot read: {codes}', Fault.PLATFORM
            ) from exc
        except httpx.TransportError as exc:
            raise CommandFailed(f'{client.base_url} did not answer: {exc!r}', Fault.PLATFORM) from exc


def credential() -> str:
    """The identity to register with, from the environment."""
    value = os.environ.get(CREDENTIAL_ENV)
    if not value:
        raise CommandFailed(
            f"no credential: set {CREDENTIAL_ENV} to a GitHub token the platform's OAuth app minted, "
            'or run `platform-register`, which mints one itself',
            Fault.CALLER,
        )
    return value


def parse_id(token: object, kind: type[ID]) -> ID:
    """One platform id off the command line, parsed as `kind`."""
    # CLI values are literal-evaluated, so an all-digit id arrives as an int, and reading that as
    # decimal would name a different record. Such an id needs inner quotes to stay text.
    if not isinstance(token, str):
        raise CommandFailed(f'an id is hexadecimal; quote one that reads as a number: \'"{token}"\'', Fault.USAGE)
    try:
        return kind.parse(token)
    except ValueError as exc:
        raise CommandFailed(f'not an id: {exc}', Fault.USAGE) from exc
