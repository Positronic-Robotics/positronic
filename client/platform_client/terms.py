"""The platform terms: the version a person accepts, and the acceptance the platform records.

The platform serves the current version at `terms.get`, and records an acceptance at `platform.accept_terms`. The
version that a request names is a check: the platform refuses one that is not current, so a version published after
the person read the terms is never accepted in their place.
"""

from __future__ import annotations

from enum import StrEnum

from platform_client.model_config import INPUT_MODEL_CONFIG
from pydantic import AwareDatetime, BaseModel, Field


class Program(StrEnum):
    """The programs whose terms the platform serves. `platform` is the platform itself."""

    platform = 'platform'
    nebius_competition = 'nebius_competition'


class Terms(BaseModel):
    """One version of the terms, and the published page that holds it."""

    version: str = Field(min_length=1)
    url: str


class TermsResponse(BaseModel):
    """`terms.get`. `terms` is None for a program that has no terms."""

    program: Program
    terms: Terms | None


class ProgramEntry(BaseModel):
    """An account's entry in a program. `terms_version` is the version the account accepted last, and None before
    its first acceptance."""

    program: Program
    entered_at: AwareDatetime
    terms_version: str | None = None


class ProgramEntryResponse(BaseModel):
    """`programs.entry`. `entry` is None for an account that has not entered the program."""

    program: Program
    entry: ProgramEntry | None


class ProgramQuery(BaseModel):
    """The query of `terms.get` and `programs.entry`."""

    model_config = INPUT_MODEL_CONFIG

    program: Program


class AcceptTermsRequest(BaseModel):
    """`platform.accept_terms`. `terms_version` is the version the person read."""

    model_config = INPUT_MODEL_CONFIG

    terms_version: str = Field(min_length=1)


def accepted(terms: Terms, entry: ProgramEntry | None) -> bool:
    """Whether `entry` holds the acceptance of `terms`."""
    return entry is not None and entry.terms_version == terms.version
