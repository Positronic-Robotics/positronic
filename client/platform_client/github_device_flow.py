"""The scope of the GitHub sign-in, at the path the platform web backend imports it from.

TODO: delete this module once the platform web backend no longer imports `IDENTITY_SCOPE` from here.
"""

from platform_client.identity_providers import IDENTITY_SCOPE

__all__ = ['IDENTITY_SCOPE']
