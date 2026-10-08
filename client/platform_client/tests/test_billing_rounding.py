import hashlib

import pytest
from platform_client.billing import CREDIT_SCALE, INT64_MAX, CreditQuote, QuoteLine, Tariff
from platform_client.enums import BillingMode
from platform_client.slug import members_by_slug
from pydantic import ValidationError


@pytest.mark.parametrize(
    ('duration_ns', 'seconds'), [(0, 0), (1, 30), (29_999_999_999, 30), (30_000_000_000, 30), (30_000_000_001, 60)]
)
def test_new_tariffs_round_each_episode_up_to_thirty_seconds(duration_ns, seconds):
    terms = Tariff.for_rates(7, CREDIT_SCALE)
    assert terms.charge_units(duration_ns) == 7 + seconds * 1_000_000_000


def test_configured_rounding_binds_the_hash_and_the_reserved_maximum():
    terms = Tariff.for_rates(7, CREDIT_SCALE, duration_rounding_sec=10)
    assert terms.version != Tariff.for_rates(7, CREDIT_SCALE).version
    line = QuoteLine(task_pos=0, endpoint='a', count=2, cap_ns=31_000_000_000, max_units=80_000_000_014)
    quote = CreditQuote(terms=terms, lines=(line,), total_units=line.max_units)
    assert CreditQuote.model_validate_json(quote.model_dump_json()) == quote
    with pytest.raises(ValidationError, match='tariff'):
        CreditQuote(terms=Tariff.for_rates(7, CREDIT_SCALE), lines=(line,), total_units=line.max_units)


def test_a_stored_tariff_without_rounding_retains_its_hash_and_exact_duration_charge():
    version = hashlib.sha256(f'v1:{CREDIT_SCALE}:7:{CREDIT_SCALE}'.encode()).hexdigest()
    terms = Tariff.model_validate({'version': version, 'episode_units': 7, 'minute_units': CREDIT_SCALE})
    assert terms.duration_rounding_sec == 0
    assert terms.charge_units(1) == 8
    assert Tariff.model_validate_json(terms.model_dump_json()) == terms


@pytest.mark.parametrize('step', [-1, True, 1.5, '30', INT64_MAX // 1_000_000_000 + 1])
def test_rounding_refuses_inexact_or_unbounded_intervals(step):
    with pytest.raises(ValidationError):
        Tariff.for_rates(0, 0, duration_rounding_sec=step)


@pytest.mark.parametrize('duration', [-1, True, 1.5, '30', INT64_MAX + 1])
def test_duration_refuses_inexact_or_unbounded_nanoseconds(duration):
    with pytest.raises(ValueError, match='duration'):
        Tariff.for_rates(0, 0).charge_units(duration)


def test_rounded_charge_refuses_integer_storage_overflow():
    with pytest.raises(ValueError, match='storage limit'):
        Tariff.for_rates(INT64_MAX, CREDIT_SCALE).charge_units(1)


def test_billing_modes_keep_the_stored_numbers_and_publish_unambiguous_names():
    assert BillingMode(1) is BillingMode.packaged
    assert BillingMode(2) is BillingMode.pay_as_you_go
    assert members_by_slug(BillingMode) == {
        'packaged': BillingMode.packaged,
        'pay_as_you_go': BillingMode.pay_as_you_go,
    }
