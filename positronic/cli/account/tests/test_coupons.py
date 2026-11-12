"""`positronic account redeem`."""

import pytest
from platform_client import routes
from platform_client.billing import CREDIT_SCALE, CouponRedemption
from platform_client.client import API_KEY_ENV, AUTH_HEADER
from platform_client.enums import ErrorCode
from platform_client.ids import CouponCode, OrgSlug
from platform_client.requests import CouponRedeemRequest

from positronic.cli.account.coupons import credits_text, redeem
from positronic.cli.account.gateway import CommandFailed, Fault
from positronic.cli.conftest import KEY, refusal

ORG = OrgSlug('user-a0')
THIRTY_CREDITS = CouponRedemption(org=ORG, units=30 * CREDIT_SCALE)


def test_redeem_sends_the_code_with_the_key_and_prints_the_credits_and_the_org(platform, run_command, capsys):
    platform.answer_by_route({routes.BILLING_COUPONS_REDEEM: (THIRTY_CREDITS, 200)})

    run_command(redeem, code='PCR-7K2M-9QXA')

    body = CouponRedeemRequest.model_validate(platform.body_to(routes.BILLING_COUPONS_REDEEM))
    assert body == CouponRedeemRequest(code=CouponCode('PCR-7K2M-9QXA'))
    assert platform.request_to(routes.BILLING_COUPONS_REDEEM).headers[AUTH_HEADER] == f'Bearer {KEY}'
    assert capsys.readouterr().out.strip() == (
        f'Added 30.00 credits ({30 * CREDIT_SCALE} units) to the organization user-a0.'
    )


def test_redeem_prints_the_answer_as_json_with_the_flag(platform, run_command, capsys):
    platform.answer_by_route({routes.BILLING_COUPONS_REDEEM: (THIRTY_CREDITS, 200)})

    run_command(redeem, code='PCR-7K2M-9QXA', json=True)

    assert CouponRedemption.model_validate_json(capsys.readouterr().out) == THIRTY_CREDITS


@pytest.mark.parametrize('code', [None, '', 20261110, 1.5])
def test_redeem_refuses_a_code_that_is_not_text_before_any_call(platform, run_command, code):
    # The CLI reads `--code=20261110` as a number.
    with pytest.raises(CommandFailed) as raised:
        run_command(redeem, code=code)

    assert raised.value.fault is Fault.USAGE
    assert platform.seen is None


def test_redeem_refuses_a_code_that_is_not_text_on_a_box_with_no_key(platform, run_command, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV)

    with pytest.raises(CommandFailed) as raised:
        run_command(redeem, code=20261110)

    assert raised.value.fault is Fault.USAGE
    assert 'quote it' in str(raised.value)


@pytest.mark.parametrize(
    ('code', 'message'),
    [
        (ErrorCode.not_found, 'no coupon has this code'),
        (ErrorCode.bad_request, 'this coupon was redeemed on 2026-11-09'),
        (
            ErrorCode.forbidden,
            'this needs the platform terms, version 1.3. When the person accepts them, run '
            '`positronic account accept-terms --version=\'"1.3"\'`',
        ),
    ],
)
def test_redeem_prints_the_refusal_of_the_platform_as_the_callers_fault(platform, run_command, code, message):
    status = {ErrorCode.not_found: 404, ErrorCode.bad_request: 400, ErrorCode.forbidden: 403}[code]
    platform.answer_by_route({routes.BILLING_COUPONS_REDEEM: (refusal(code, message), status)})

    with pytest.raises(CommandFailed) as raised:
        run_command(redeem, code='PCR-7K2M-9QXA')

    assert raised.value.fault is Fault.CALLER
    assert message in str(raised.value)


def test_redeem_leaves_the_rule_of_a_long_code_to_the_platform(platform, run_command):
    message = 'no coupon has this code'
    platform.answer_by_route({routes.BILLING_COUPONS_REDEEM: (refusal(ErrorCode.not_found, message), 404)})

    with pytest.raises(CommandFailed) as raised:
        run_command(redeem, code='X' * 300)

    assert raised.value.fault is Fault.CALLER
    assert message in str(raised.value)
    assert CouponRedeemRequest.model_validate(platform.body_to(routes.BILLING_COUPONS_REDEEM)).code == 'X' * 300


@pytest.mark.parametrize(
    ('units', 'shown'),
    [(CREDIT_SCALE, '1.00'), (CREDIT_SCALE * 100 // 3, '33.33'), (CREDIT_SCALE // 100 - 1, '0.00'), (1, '0.00')],
)
def test_credits_text_rounds_down_to_two_decimals(units, shown):
    assert credits_text(units) == shown
