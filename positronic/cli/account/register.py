"""`positronic account register` — the account this platform knows you by."""

import os

import configuronic as cfn
from platform_client.config import CONFIG_FILENAME, Config, config_dir, write_config
from platform_client.requests import RegisterRequest

from positronic.cli.account.gateway import credential, gateway, refusing_bad_input


@cfn.config()
def register(alias: str | None = None, rotate: bool = False, platform_url: str | None = None):
    """Register with the platform, or rotate an existing registration's API key.

    Reads the credential from the environment, never an argument, and saves the key it mints in the
    config record every other command reads.
    """
    with refusing_bad_input():
        request = RegisterRequest(credential=credential(), alias=alias, rotate=rotate)
    with gateway(platform_url, key_required=False) as client:
        response = client.register(request)
        base_url = client.base_url
    print(f'user {response.user_id} ({response.key_status.name})')
    if response.api_key is None:
        print('no key issued: one is minted on a first registration, or by --rotate')
        return
    directory = config_dir(os.environ)
    try:
        write_config(directory, Config(platform_url=base_url, api_key=response.api_key))
    except OSError as exc:
        # The key is out and cannot be read back, so the message says what to do and never shows it.
        raise SystemExit(
            f'the platform issued a key for user {response.user_id}, and writing '
            f'{directory / CONFIG_FILENAME} failed: {exc.strerror or exc}. The key is not shown; '
            'run `positronic account register --rotate` to mint another.'
        ) from exc
    print(f'key saved in {directory / CONFIG_FILENAME}')
