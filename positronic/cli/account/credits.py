"""Read credit balances and create or inspect a frozen credit purchase."""

import configuronic as cfn
from platform_client.ids import OrgSlug, TransactionKey
from platform_client.requests import BillingAccountQuery, BillingPurchaseCreateRequest, BillingPurchaseGetQuery

from positronic.cli.account.gateway import gateway, refusing_bad_input


@cfn.config()
def account(org: str, platform_url: str | None = None):
    """Print exact credit units, configured tariff rates, and purchase packages."""
    with refusing_bad_input():
        query = BillingAccountQuery(org=OrgSlug(org))
    with gateway(platform_url) as client:
        result = client.billing_account(query.org)
    print(result.model_dump_json(indent=2))


@cfn.config()
def buy(org: str, package_id: str, transaction_key: str, platform_url: str | None = None):
    """Create a purchase, or read the same purchase by its original retry key."""
    with refusing_bad_input():
        request = BillingPurchaseCreateRequest(
            org=OrgSlug(org), package_id=package_id, transaction_key=TransactionKey(transaction_key)
        )
    with gateway(platform_url) as client:
        result = client.create_purchase(request)
    print(result.model_dump_json(indent=2))


@cfn.config()
def purchase(id: str, platform_url: str | None = None):
    """Print one purchase and any Checkout URL still available to its initiating member."""
    with refusing_bad_input():
        query = BillingPurchaseGetQuery(id=id)
    with gateway(platform_url) as client:
        result = client.get_purchase(query.id)
    print(result.model_dump_json(indent=2))


@cfn.config()
def purchases(org: str, platform_url: str | None = None):
    """Print the purchase history of one organization this account belongs to."""
    with refusing_bad_input():
        query = BillingAccountQuery(org=OrgSlug(org))
    with gateway(platform_url) as client:
        result = client.list_purchases(query.org)
    print(result.model_dump_json(indent=2))
