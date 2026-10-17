"""`positronic account register` — the account this platform knows you by."""

import os
from collections.abc import Iterator
from contextlib import contextmanager

import configuronic as cfn
import httpx
from platform_client.client import PlatformClient
from platform_client.config import CONFIG_FILENAME, Config, config_dir, write_config
from platform_client.device_flow import (
    ClientIds,
    DeviceFlowError,
    NoSignIn,
    ProviderFailed,
    may_have_minted_a_key,
    register_with_device_flow,
    require_allowed_platform_url,
    sign_in_flow,
)
from platform_client.identity_providers import REQUEST_TIMEOUT
from platform_client.requests import RegisterRequest
from platform_client.responses import RegisterResponse

from positronic.cli.account.gateway import CREDENTIAL_ENV, CommandFailed, Fault, gateway, refusing_bad_input


def _register_by_sign_in(
    client: PlatformClient, *, alias: str | None, rotate: bool, plaintext_http: bool
) -> RegisterResponse:
    """Sign in through the device flow the platform names, and register with the token it yields."""
    with refusing_bad_input():
        client_ids = ClientIds.resolve(os.environ)
        require_allowed_platform_url(client.base_url, plaintext_http=plaintext_http, flag='--plaintext_http')
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as http:
            flow = sign_in_flow(client, http, client_ids)
            return register_with_device_flow(client, flow, alias=alias, rotate=rotate)
    except DeviceFlowError as exc:
        # A provider that fails and a platform that takes no sign-in are no fault of the caller.
        fault = Fault.PLATFORM if isinstance(exc, ProviderFailed | NoSignIn) else Fault.CALLER
        raise CommandFailed(str(exc), fault) from exc


@contextmanager
def _naming_rotate_when_the_answer_is_lost() -> Iterator[None]:
    """A registration whose answer is lost may have minted the key, so its failure names `--rotate`."""
    try:
        yield
    except httpx.HTTPError as exc:
        if not may_have_minted_a_key(exc):
            raise
        raise CommandFailed(
            f'{exc.request.url} did not answer the registration: {exc!r}. A key may have been minted; '
            'run `positronic account register --rotate` to mint another.',
            Fault.PLATFORM,
        ) from exc


@cfn.config()
def register(
    alias: str | None = None, rotate: bool = False, platform_url: str | None = None, plaintext_http: bool = False
):
    """Register with the platform, or rotate an existing registration's API key.

    Signs in through the device flow the platform names, or takes a credential from the environment, never
    from an argument. Saves the key it mints in the config record every other command reads.
    `plaintext_http` lets the sign-in token reach an http platform that is not loopback.
    """
    # configuronic literal-evaluates a CLI value, so `--alias=5` arrives as an int. The sign-in runs before
    # the request model sees these values, so these refusals come first.
    if alias is not None and not isinstance(alias, str):
        raise CommandFailed(f'alias is text; quote one that reads as a number: \'"{alias}"\'', Fault.USAGE)
    # `--plaintext_http=false` arrives as a string, which would open the gate it means to keep shut.
    for name, value in (('rotate', rotate), ('plaintext_http', plaintext_http)):
        if not isinstance(value, bool):
            raise CommandFailed(f'{name} is True or False, not {value!r}', Fault.USAGE)
    credential = os.environ.get(CREDENTIAL_ENV)
    # Unset runs the sign-in. Empty is a script whose secret did not arrive, and a sign-in would wait for a person.
    if credential == '':
        raise CommandFailed(f'{CREDENTIAL_ENV} is empty: set a token, or unset it to sign in', Fault.USAGE)
    # A key is minted once and cannot be fetched again, so the destination is named before the
    # platform mints one: a config directory that cannot be resolved would otherwise cost the key.
    directory = config_dir(os.environ)
    with gateway(platform_url, key_required=False) as client, _naming_rotate_when_the_answer_is_lost():
        if credential is not None:
            response = client.register(RegisterRequest(credential=credential, alias=alias, rotate=rotate))
        else:
            response = _register_by_sign_in(client, alias=alias, rotate=rotate, plaintext_http=plaintext_http)
        base_url = client.base_url
    print(f'user {response.user_id} ({response.key_status.name})')
    if response.api_key is None:
        print('no key issued: one is minted on a first registration, or by --rotate')
        return
    try:
        write_config(directory, Config(platform_url=base_url, api_key=response.api_key))
    except OSError as exc:
        # The key cannot be fetched again, so the message says how to mint another and does not print this one.
        raise CommandFailed(
            f'the platform issued a key for user {response.user_id}, and writing '
            f'{directory / CONFIG_FILENAME} failed: {exc.strerror or exc}. The key is not shown; '
            'run `positronic account register --rotate` to mint another.',
            Fault.CALLER,
        ) from exc
    print(f'key saved in {directory / CONFIG_FILENAME}')
