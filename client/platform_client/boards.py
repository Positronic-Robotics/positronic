"""The slug a ranking board is read by. `rankings.list` names every slug the caller can read.

The platform owns the set, so a board this client has never heard of still reaches the server.
"""

from __future__ import annotations

from positronic_evals.slug import Name


class BoardRef(Name, described_as='a board slug'):
    """The slug of a board the platform publishes, as a validated `str`."""

    __slots__ = ()
