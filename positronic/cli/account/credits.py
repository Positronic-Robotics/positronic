import sys

import configuronic as cfn
from platform_client.client import PlatformClient
from platform_client.ids import OrgSlug, PackageId, PurchaseId, TransactionKey
from platform_client.requests import (
    BillingOrgQuery,
    BillingPurchaseCreateRequest,
    BillingPurchaseGetQuery,
    BillingPurchaseListQuery,
)

from positronic.cli.account.gateway import gateway, refusing_bad_input


def _text(token: object, field: str) -> str:
    if not isinstance(token, str):
        raise SystemExit(f'{field} must be text; quote the original argument with inner double quotes')
    if not token:
        raise SystemExit(f'{field} must not be empty')
    return token


def _named_org(org: object) -> OrgSlug | None:
    if org is None:
        return None
    with refusing_bad_input():
        return BillingOrgQuery(org=OrgSlug(_text(org, 'org'))).org


def _org(client: PlatformClient, named: OrgSlug | None) -> OrgSlug:
    """`named` or the caller's personal org."""
    if named is not None:
        org, source = named, 'from --org'
    elif (personal := client.me().personal_org) is not None:
        org, source = personal, 'personal org'
    else:
        raise SystemExit('name an org with --org: the platform names no personal org')
    # On stderr, because stdout carries only the JSON answer.
    print(f'org: {org} ({source})', file=sys.stderr)
    return org


@cfn.config()
def account(org: object = None, platform_url: str | None = None):
    """Print exact credit units, configured tariff rates and purchase packages of `org` or the caller's personal org."""
    named = _named_org(org)
    with gateway(platform_url) as client:
        result = client.billing_account(_org(client, named))
    print(result.model_dump_json(indent=2))


@cfn.config()
def buy(package_id: object, transaction_key: object, org: object = None, platform_url: str | None = None):
    """Create a purchase for `org` or the caller's personal org. A used retry key reads the same purchase."""
    package, key = _text(package_id, 'package_id'), _text(transaction_key, 'transaction_key')
    named = _named_org(org)
    with gateway(platform_url) as client:
        owner = _org(client, named)
        with refusing_bad_input():
            request = BillingPurchaseCreateRequest(
                org=owner, package_id=PackageId(package), transaction_key=TransactionKey(key)
            )
        result = client.create_purchase(request)
    print(result.model_dump_json(indent=2))


@cfn.config()
def purchase(id: object, platform_url: str | None = None):
    """Print one purchase and any Checkout URL still available to its initiating member."""
    with refusing_bad_input():
        query = BillingPurchaseGetQuery(id=PurchaseId(_text(id, 'id')))
    with gateway(platform_url) as client:
        result = client.get_purchase(query.id)
    print(result.model_dump_json(indent=2))


@cfn.config()
def purchases(org: object = None, after: object | None = None, limit: int = 50, platform_url: str | None = None):
    """Print one purchase history page. Pass its next cursor as after to continue."""
    named = _named_org(org)
    cursor = PurchaseId(_text(after, 'after')) if after is not None else None
    with gateway(platform_url) as client:
        with refusing_bad_input():
            query = BillingPurchaseListQuery(org=_org(client, named), after=cursor, limit=limit)
        result = client.list_purchases(query.org, after=query.after, limit=query.limit)
    print(result.model_dump_json(indent=2))
