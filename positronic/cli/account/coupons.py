"""`positronic account redeem` — add the credits of a coupon to the account.

A coupon comes by mail with its code. The platform adds its credits to the caller's personal organization, one time.
"""

import configuronic as cfn
from platform_client.billing import CREDIT_SCALE
from platform_client.ids import CouponCode

from positronic.cli.account.gateway import CommandFailed, Fault, gateway


def code_text(code: object) -> CouponCode:
    """The code off the command line. The CLI reads a code of digits as a number, so a code must arrive as text."""
    if not isinstance(code, str) or not code:
        raise CommandFailed(f'the code is text; quote it: --code=\'"{code}"\'', Fault.USAGE)
    return CouponCode(code)


def credits_text(units: int) -> str:
    """`units` as credits with two decimals, rounded down, so the line never shows more than the balance holds."""
    hundredths = units * 100 // CREDIT_SCALE
    return f'{hundredths // 100}.{hundredths % 100:02d}'


@cfn.config()
def redeem(code: object = None, platform_url: str | None = None, json: bool = False):
    """Add the credits of a coupon to your personal organization. `code` is the code in the coupon mail."""
    if code is None:
        raise CommandFailed('name the code of the coupon: --code=<code>', Fault.USAGE)
    with gateway(platform_url, refusal_as_json=json) as client:
        redemption = client.redeem_coupon(code_text(code))
    if json:
        print(redemption.model_dump_json(indent=2))
        return
    added = f'{credits_text(redemption.units)} credits ({redemption.units} units)'
    print(f'Added {added} to the organization {redemption.org}.')
