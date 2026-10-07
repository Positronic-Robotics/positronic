import configuronic as cfn
from platform_client.billing import PurchasePageLimit
from platform_client.ids import OrgSlug, PackageId, PurchaseId, TransactionKey
from platform_client.requests import (
    DEFAULT_PURCHASE_PAGE_SIZE,
    BillingOrgQuery,
    BillingPurchaseCreateRequest,
    BillingPurchaseGetQuery,
    BillingPurchaseListQuery,
)
from pydantic import ConfigDict, TypeAdapter

from positronic.cli.account.gateway import gateway, refusing_bad_input


def _text(token: object, field: str) -> str:
    if not isinstance(token, str):
        raise SystemExit(f'{field} must be text; quote the original argument with inner double quotes')
    return token


@cfn.config()
def account(org: object, platform_url: str | None = None):
    """Print exact credit units, configured tariff rates, and purchase packages."""
    with refusing_bad_input():
        query = BillingOrgQuery(org=OrgSlug(_text(org, 'org')))
    with gateway(platform_url) as client:
        result = client.billing_account(query.org)
    print(result.model_dump_json(indent=2))


@cfn.config()
def buy(org: object, package_id: object, transaction_key: object, platform_url: str | None = None):
    """Create a purchase, or read the same purchase by its original retry key."""
    with refusing_bad_input():
        request = BillingPurchaseCreateRequest(
            org=OrgSlug(_text(org, 'org')),
            package_id=PackageId(_text(package_id, 'package_id')),
            transaction_key=TransactionKey(_text(transaction_key, 'transaction_key')),
        )
    with gateway(platform_url) as client:
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
def purchases(
    org: object,
    after: object | None = None,
    limit: object = DEFAULT_PURCHASE_PAGE_SIZE,
    platform_url: str | None = None,
):
    """Print one purchase history page. Pass its next cursor as after to continue."""
    with refusing_bad_input():
        query = BillingPurchaseListQuery(
            org=OrgSlug(_text(org, 'org')),
            after=PurchaseId(_text(after, 'after')) if after is not None else None,
            limit=TypeAdapter(PurchasePageLimit, config=ConfigDict(title='limit')).validate_python(limit),
        )
    with gateway(platform_url) as client:
        result = client.list_purchases(query.org, after=query.after, limit=query.limit)
    print(result.model_dump_json(indent=2))
