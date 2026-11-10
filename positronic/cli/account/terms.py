"""`positronic account terms` and `positronic account accept-terms` — the platform terms of the account.

The platform can refuse an account that has not accepted its current terms. A person reads the terms on their page
and accepts them here with the version that they read.
"""

import configuronic as cfn
from platform_client.client import PlatformClient
from platform_client.config import ACCEPT_TERMS_COMMAND
from platform_client.terms import ProgramEntry, Terms

from positronic.cli.account.gateway import CommandFailed, Fault, gateway


def accept_command(version: str) -> str:
    """The command that accepts `version`. The inner quotes keep a version such as `1.3` text on the command line."""
    return f'{ACCEPT_TERMS_COMMAND} --version=\'"{version}"\''


def terms_lines(terms: Terms) -> list[str]:
    return [
        f'The platform terms, version {terms.version}: {terms.url}',
        f'Read them. To accept them, run: {accept_command(terms.version)}',
    ]


def version_text(version: object) -> str:
    """The version off the command line. The CLI reads `1.3` as a number, so a version must arrive as text."""
    if not isinstance(version, str) or not version:
        raise CommandFailed(f'the version is text; quote it: --version=\'"{version}"\'', Fault.USAGE)
    return version


def accept(client: PlatformClient, version: str) -> ProgramEntry:
    """Record the acceptance of `version` and print it. The platform refuses a version that is not current."""
    entry = client.accept_platform_terms(version)
    print(f'You accepted the platform terms, version {entry.terms_version}.')
    return entry


@cfn.config()
def terms(platform_url: str | None = None):
    """Print the current platform terms, and whether this account accepted them."""
    with gateway(platform_url) as client:
        current = client.platform_terms()
        if current is None:
            print('The platform has no terms.')
            return
        entry = client.platform_entry()
    if entry is not None and entry.terms_version == current.version:
        print(f'The platform terms, version {current.version}: {current.url}')
        print(f'You accepted version {current.version}.')
        return
    for line in terms_lines(current):
        print(line)
    if entry is not None and entry.terms_version is not None:
        print(f'You accepted version {entry.terms_version}, which is not current.')


@cfn.config()
def accept_terms(version: object = None, platform_url: str | None = None):
    """Accept the current platform terms. `version` is the version that you read, which `terms` prints."""
    if version is None:
        raise CommandFailed('name the version that you read: run `positronic account terms`', Fault.USAGE)
    shown = version_text(version)
    with gateway(platform_url) as client:
        accept(client, shown)
