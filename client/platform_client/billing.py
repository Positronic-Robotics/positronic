from __future__ import annotations

import hashlib
from typing import Annotated, Self

from platform_client.enums import BillingMode, BillingRole, BillingState
from platform_client.ids import INT64_MAX as INT64_MAX
from platform_client.ids import OrgSlug, PackageId, PurchaseId, UserId
from platform_client.slug import Slugged
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, computed_field, model_validator

CREDIT_SCALE = 60_000_000_000
NANOSECONDS_PER_MINUTE = 60_000_000_000
CreditUnits = Annotated[int, Field(strict=True, ge=0, le=INT64_MAX)]


class Tariff(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    version: str
    episode_units: CreditUnits
    minute_units: CreditUnits

    @staticmethod
    def content_hash(episode_units: int, minute_units: int) -> str:
        return hashlib.sha256(f'v1:{CREDIT_SCALE}:{episode_units}:{minute_units}'.encode()).hexdigest()

    @classmethod
    def for_rates(cls, episode_units: int, minute_units: int) -> Tariff:
        return cls(
            version=cls.content_hash(episode_units, minute_units),
            episode_units=episode_units,
            minute_units=minute_units,
        )

    @model_validator(mode='after')
    def _version_binds_the_rates(self) -> Self:
        if self.version != self.content_hash(self.episode_units, self.minute_units):
            raise ValueError('tariff version does not match its rates')
        return self


class QuoteLine(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    task_pos: int = Field(strict=True, ge=0, le=INT64_MAX)
    endpoint: str = Field(min_length=1)
    count: int = Field(strict=True, ge=1, le=INT64_MAX)
    cap_ns: int = Field(strict=True, ge=1, le=INT64_MAX)
    max_units: CreditUnits


class CreditQuote(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    terms: Tariff
    lines: tuple[QuoteLine, ...] = Field(min_length=1)
    total_units: CreditUnits

    @model_validator(mode='after')
    def _lines_match_the_terms(self) -> Self:
        bindings = [(line.task_pos, line.endpoint) for line in self.lines]
        if len(set(bindings)) != len(bindings):
            raise ValueError('quote repeats a task endpoint')
        for line in self.lines:
            maximum = line.count * (
                self.terms.episode_units + line.cap_ns * self.terms.minute_units // NANOSECONDS_PER_MINUTE
            )
            if line.max_units != maximum:
                raise ValueError('quote line does not match its tariff')
        if self.total_units != sum(line.max_units for line in self.lines):
            raise ValueError('quote total does not match its lines')
        return self


class RequestBilling(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    mode: Slugged[BillingMode]
    quote: CreditQuote | None = None
    state: Slugged[BillingState]

    @model_validator(mode='after')
    def _mode_has_the_matching_quote(self) -> Self:
        if (self.mode is BillingMode.prepaid) != (self.quote is not None):
            raise ValueError('a prepaid request requires a quote, and a legacy request cannot carry one')
        if self.mode is BillingMode.legacy and self.state is not BillingState.settled:
            raise ValueError('a legacy request holds no credits')
        return self


class CreditBalance(BaseModel):
    posted_units: CreditUnits
    reserved_units: CreditUnits

    @computed_field
    @property
    def available_units(self) -> int:
        return self.posted_units - self.reserved_units

    @model_validator(mode='after')
    def _holds_fit_the_balance(self) -> Self:
        if self.available_units < 0:
            raise ValueError('reserved credits exceed posted credits')
        return self


class CreditPackage(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    id: PackageId = Field(min_length=1)
    credit_units: CreditUnits = Field(gt=0)
    amount_minor: int = Field(strict=True, gt=0, le=INT64_MAX)
    currency: str = Field(pattern=r'^[a-z]{3}$')


class BillingAccount(BaseModel):
    org: OrgSlug
    mode: Slugged[BillingMode]
    billing_role: Slugged[BillingRole]
    balance: CreditBalance
    tariff: Tariff
    packages: tuple[CreditPackage, ...]


class PurchaseView(BaseModel):
    id: PurchaseId = Field(min_length=1)
    package: CreditPackage
    initiated_by: UserId
    created_at: AwareDatetime
    checkout_url: str | None = Field(default=None, min_length=1)
    granted_at: AwareDatetime | None = None
    review_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode='after')
    def _payable_link_matches_lifecycle(self) -> Self:
        if self.checkout_url is not None and (self.granted_at is not None or self.review_reason is not None):
            raise ValueError('a reviewed or credited purchase cannot carry a payable link')
        return self


class PurchaseListResponse(BaseModel):
    purchases: list[PurchaseView]
