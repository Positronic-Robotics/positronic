"""`positronic account register` — the account this platform knows you by."""

import shlex

import configuronic as cfn
from platform_client.requests import RegisterRequest

from positronic.cli.account.gateway import API_KEY_ENV, credential, gateway, refusing_bad_input


@cfn.config()
def register(alias: str | None = None, rotate: bool = False, platform_url: str | None = None):
    """Register with the platform, or rotate an existing registration's API key.

    Reads the credential from the environment, never an argument.
    """
    with refusing_bad_input():
        request = RegisterRequest(credential=credential(), alias=alias, rotate=rotate)
    with gateway(platform_url, key_required=False) as client:
        response = client.register(request)
    print(f'user {response.user_id} ({response.key_status.name})')
    if response.api_key is None:
        print('no key issued: one is minted on a first registration, or by --rotate')
    else:
        # The key is opaque, so it may hold whitespace or shell metacharacters; an unquoted export
        # line would either mangle it or run the rest of it.
        print(f'export {API_KEY_ENV}={shlex.quote(response.api_key)}')
