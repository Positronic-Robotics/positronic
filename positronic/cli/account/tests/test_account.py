"""`positronic account` commands over a stub platform transport."""

import json
import os

import pytest
from fire import parser
from platform_client import config as config_module
from platform_client import routes
from platform_client.billing import (
    CREDIT_SCALE,
    BillingAccount,
    CreditBalance,
    CreditPackage,
    PurchaseListResponse,
    PurchaseView,
    Tariff,
)
from platform_client.client import PlatformClient
from platform_client.config import CONFIG_FILENAME, Config, config_dir, read_config, write_config
from platform_client.enums import BillingMode, BillingRole, ErrorCode, KeyStatus
from platform_client.ids import ApiKey, OrgSlug, PackageId, PurchaseId
from platform_client.responses import MeResponse, RegisterResponse
from pydantic import ValidationError

from positronic.cli.account import commands
from positronic.cli.account import gateway as gateway_module
from positronic.cli.account.credits import account, buy, get_purchase, list_purchases
from positronic.cli.account.gateway import CommandFailed, Fault
from positronic.cli.account.register import register
from positronic.cli.conftest import USER, WHEN, refusal

CREATED = RegisterResponse(
    user_id=USER, artifact_location='s3://b/users/a0/', api_key=ApiKey('pk_new'), key_status=KeyStatus.created
)
EXISTING = RegisterResponse(user_id=USER, artifact_location='s3://b/users/a0/', key_status=KeyStatus.existing)


def test_register_saves_the_minted_key_in_the_record(platform, run_command, capsys):
    platform.answer(CREATED)

    run_command(register, alias='demo')

    assert platform.request.url.path == routes.USERS_REGISTER
    assert 'authorization' not in platform.request.headers
    assert platform.body == {'credential': 'token', 'alias': 'demo', 'rotate': False}
    saved = read_config(config_dir(os.environ))
    assert saved is not None and saved.api_key == 'pk_new'
    out = capsys.readouterr().out
    assert 'user a0 (created)' in out
    # The record holds the key, so the command names the file and prints none of it.
    assert 'pk_new' not in out


def test_the_key_is_recorded_against_the_platform_it_was_minted_on(platform, run_command):
    platform.answer(CREATED)

    run_command(register, platform_url='http://other.test')

    saved = read_config(config_dir(os.environ))
    assert saved is not None and saved.platform_url.startswith('http://other.test')


def test_a_registration_goes_to_the_platform_it_names_whatever_the_record_holds(platform, run_command, monkeypatch):
    # A registration sends no key, so the record has nothing to say about where it goes.
    write_config(config_dir(os.environ), Config(platform_url='http://gateway.test', api_key=ApiKey('pk_old')))
    monkeypatch.delenv(gateway_module.API_KEY_ENV)
    monkeypatch.delenv(gateway_module.API_URL_ENV)
    platform.answer(EXISTING)

    run_command(register, platform_url='http://other.test')

    assert platform.base_url == 'http://other.test'
    assert platform.request.url.path == routes.USERS_REGISTER
    assert 'authorization' not in platform.request.headers


def test_a_config_directory_that_cannot_be_named_stops_before_a_key_is_minted(platform, run_command, monkeypatch):
    # A key is minted once. Resolving the destination after the mint would spend it on a record the
    # command cannot write, and a retry without --rotate answers `existing` with no key.
    monkeypatch.setenv(config_module.CONFIG_DIR_ENV, '   ')

    with pytest.raises(SystemExit, match='set to an empty value'):
        run_command(register, platform_url='http://other.test')

    assert platform.seen is None


def test_a_config_directory_under_a_home_this_machine_does_not_have_names_the_variable(
    platform, run_command, monkeypatch
):
    monkeypatch.setenv(config_module.CONFIG_DIR_ENV, '~no-such-user-on-this-box/config')

    with pytest.raises(SystemExit) as exit_info:
        run_command(register, platform_url='http://other.test')

    assert config_module.CONFIG_DIR_ENV in str(exit_info.value)
    assert '~no-such-user-on-this-box/config' in str(exit_info.value)
    assert platform.seen is None


def test_a_registration_naming_its_platform_reads_no_key_and_no_record(platform, run_command, monkeypatch):
    # A blank key variable and a file that is no record each end a command that needs a key; a
    # registration needs none, and names its platform, so neither is read.
    (config_dir(os.environ)).mkdir(parents=True)
    (config_dir(os.environ) / CONFIG_FILENAME).write_text('not a record')
    monkeypatch.setenv(gateway_module.API_KEY_ENV, '  ')
    platform.answer(EXISTING)

    run_command(register, platform_url='http://other.test')

    assert platform.request.url.path == routes.USERS_REGISTER


def test_a_registration_naming_its_platform_replaces_a_malformed_record(platform, run_command, monkeypatch):
    # A registration sends no key and names its platform, so the record has nothing to give it.
    # Reading one lets a malformed record refuse the very command that rewrites it.
    config_dir(os.environ).mkdir(parents=True)
    (config_dir(os.environ) / CONFIG_FILENAME).write_text('not a record')
    monkeypatch.delenv(gateway_module.API_KEY_ENV)
    monkeypatch.delenv(gateway_module.API_URL_ENV)
    platform.answer(CREATED)

    run_command(register, platform_url='http://other.test')

    assert platform.request.url.path == routes.USERS_REGISTER
    saved = read_config(config_dir(os.environ))
    assert saved is not None and saved.api_key == 'pk_new'


def test_register_refuses_when_no_credential_is_in_the_environment(platform, run_command, monkeypatch):
    monkeypatch.delenv(gateway_module.CREDENTIAL_ENV)
    with pytest.raises(SystemExit) as raised:
        run_command(register)
    assert gateway_module.CREDENTIAL_ENV in str(raised.value)


def test_an_alias_the_request_model_refuses_is_a_refusal_naming_the_field(platform, run_command):
    # configuronic literal-evaluates a CLI value, so `--alias=5` arrives as an int, which the model
    # refuses — a refusal to read rather than a traceback out of the constructor.
    with pytest.raises(SystemExit) as raised:
        run_command(register, alias=5)
    assert 'alias' in str(raised.value)
    assert platform.seen is None


def test_register_says_no_key_came_back_for_an_existing_registration(platform, run_command, capsys):
    platform.answer(EXISTING)

    run_command(register)

    assert 'no key issued' in capsys.readouterr().out
    # A repeat registration mints no key, so no record is written.
    assert read_config(config_dir(os.environ)) is None


PACKAGE = CreditPackage(id=PackageId('package'), credit_units=CREDIT_SCALE, amount_minor=17, currency='jpy')
TARIFF = Tariff.for_rates(10_000_000_000, CREDIT_SCALE, duration_rounding_sec=30)


def _purchase(
    checkout_url: str | None = 'https://checkout.stripe.com/accepted', review_reason: str | None = None
) -> PurchaseView:
    """The purchase `opaque-purchase` of `PACKAGE`."""
    return PurchaseView(
        id=PurchaseId('opaque-purchase'),
        package=PACKAGE,
        initiated_by=USER,
        created_at=WHEN,
        checkout_url=checkout_url,
        review_reason=review_reason,
    )


PURCHASE = _purchase()


def test_account_prints_exact_units_and_operator_configured_package_terms(platform, run_command, capsys):
    answered = _account('acme', CreditBalance(posted_units=CREDIT_SCALE, reserved_units=1))
    platform.answer(answered)
    run_command(account, org='acme')
    assert platform.request.url.path == routes.BILLING_ACCOUNT
    assert platform.request.url.params['org'] == 'acme'
    assert BillingAccount.model_validate_json(capsys.readouterr().out) == answered


def test_buy_sends_a_named_retry_key_and_prints_the_owned_checkout(platform, run_command, capsys):
    platform.answer(PURCHASE)
    run_command(buy, org='acme', package_id='package', transaction_key='retry-key')
    assert platform.request.url.path == routes.BILLING_PURCHASES_CREATE
    assert platform.body == {'org': 'acme', 'package_id': 'package', 'transaction_key': 'retry-key'}
    printed = capsys.readouterr()
    assert PurchaseView.model_validate_json(printed.out) == PURCHASE
    assert printed.err == 'org: acme (from --org)\n'


@pytest.mark.parametrize('field', ['org', 'package_id', 'transaction_key'])
def test_buy_refuses_invalid_input_before_http(field, platform, run_command):
    args = {'org': 'acme', 'package_id': 'package', 'transaction_key': 'retry-key', field: ''}
    with pytest.raises(SystemExit):
        run_command(buy, **args)
    assert platform.seen is None


@pytest.mark.parametrize('field', ['package_id', 'transaction_key'])
def test_buy_without_an_org_refuses_invalid_input_before_http(field, platform, run_command):
    args = {'package_id': 'package', 'transaction_key': 'retry-key', field: ''}
    with pytest.raises(SystemExit, match=f'^{field} must not be empty$'):
        run_command(buy, **args)
    assert platform.seen is None


@pytest.mark.parametrize('field', ['org', 'package_id', 'transaction_key'])
def test_buy_explains_how_to_quote_numeric_text_before_http(field, platform, run_command):
    args = {'org': 'acme', 'package_id': 'package', 'transaction_key': 'retry-key', field: 20261005}
    with pytest.raises(SystemExit, match='quote'):
        run_command(buy, **args)
    assert platform.seen is None


def test_purchase_explains_how_to_quote_numeric_text_before_http(platform, run_command):
    with pytest.raises(CommandFailed, match='quote') as exit_info:
        run_command(get_purchase, id=20261005)
    assert platform.seen is None
    assert exit_info.value.fault is Fault.USAGE


@pytest.mark.parametrize('argument', ['1_000', '0x1F', '1e3', '20261005'])
def test_buy_quote_hint_preserves_the_original_retry_key(argument, platform, run_command):
    token = parser.DefaultParseValue(argument)
    with pytest.raises(SystemExit) as raised:
        run_command(buy, org='acme', package_id='package', transaction_key=token)
    assert 'original argument' in str(raised.value)
    assert str(token) not in str(raised.value)
    assert platform.seen is None


def test_purchase_reads_an_opaque_id_without_creating_another_checkout(platform, run_command, capsys):
    platform.answer(_purchase(checkout_url=None, review_reason='payment review'))
    run_command(get_purchase, id='opaque-purchase')
    assert platform.request.method == 'GET'
    assert platform.request.url.path == routes.BILLING_PURCHASES_GET
    assert platform.request.url.params['id'] == 'opaque-purchase'
    assert PurchaseView.model_validate_json(capsys.readouterr().out).review_reason == 'payment review'


def test_purchases_reads_the_named_member_account(platform, run_command, capsys):
    platform.answer(PurchaseListResponse(purchases=[PURCHASE], next=PurchaseId('opaque-purchase')))
    run_command(list_purchases, org='acme')
    assert platform.request.url.path == routes.BILLING_PURCHASES_LIST
    assert platform.request.url.params['org'] == 'acme'
    printed = capsys.readouterr()
    assert platform.request.url.params['limit'] == '50'
    page = PurchaseListResponse.model_validate_json(printed.out)
    assert page == PurchaseListResponse(purchases=[PURCHASE], next=PurchaseId('opaque-purchase'))
    assert printed.err == 'org: acme (from --org)\n'


def _me(*organizations: str) -> MeResponse:
    """The profile of the user `a0`, a member of `organizations`."""
    return MeResponse(user_id=USER, tenant='t', plan='p', quota=[], organizations=[OrgSlug(o) for o in organizations])


ME = _me('acme', 'user-a0')
# The wire form with no membership list at all, which a profile from an older platform sends.
ME_WITHOUT_MEMBERSHIPS = _me().model_dump(mode='json', exclude_defaults=True)


def _account(org: str, balance: CreditBalance | None = None) -> BillingAccount:
    """The pay-as-you-go account of `org`, empty unless `balance` is given."""
    return BillingAccount(
        org=OrgSlug(org),
        mode=BillingMode.pay_as_you_go,
        billing_role=BillingRole.spender,
        balance=balance or CreditBalance(posted_units=0, reserved_units=0),
        tariff=TARIFF,
        packages=(PACKAGE,),
    )


ACCOUNT = _account('user-a0')
NO_PURCHASES = PurchaseListResponse(purchases=[], next=None)


@pytest.mark.parametrize(
    ('command', 'args', 'route', 'answer'),
    [
        (account, {}, routes.BILLING_ACCOUNT, ACCOUNT),
        (buy, {'package_id': 'package', 'transaction_key': 'retry-key'}, routes.BILLING_PURCHASES_CREATE, PURCHASE),
        (list_purchases, {}, routes.BILLING_PURCHASES_LIST, NO_PURCHASES),
    ],
)
@pytest.mark.parametrize('org', ['user-a0', 'acme'])
def test_a_credit_command_without_an_org_uses_the_sole_membership(
    command, args, route, answer, org, platform, run_command, capsys
):
    if command is account:
        answer = _account(org)
    platform.answer_by_route({routes.USERS_ME: (_me(org), 200), route: (answer, 200)})

    run_command(command, **args)

    assert platform.paths == [routes.USERS_ME, route]
    sent = platform.body['org'] if platform.request.method == 'POST' else platform.request.url.params['org']
    assert sent == org
    printed = capsys.readouterr()
    assert json.loads(printed.out)
    assert printed.err == f'org: {org} (sole organization)\n'


def test_a_named_org_is_used_without_fetching_memberships(platform, run_command):
    platform.answer(ACCOUNT)

    run_command(account, org='acme')

    assert platform.paths == [routes.BILLING_ACCOUNT]


@pytest.mark.parametrize(
    ('command', 'args'),
    [(account, {}), (buy, {'package_id': 'package', 'transaction_key': 'retry-key'}), (list_purchases, {})],
)
@pytest.mark.parametrize(
    'me', [ME, _me(), ME_WITHOUT_MEMBERSHIPS], ids=['several-memberships', 'empty-memberships', 'missing-memberships']
)
def test_credit_commands_require_an_org_without_a_sole_membership(command, args, me, platform, run_command):
    platform.answer(me)

    with pytest.raises(CommandFailed, match='--org') as exit_info:
        run_command(command, **args)

    assert platform.paths == [routes.USERS_ME]
    assert exit_info.value.fault is Fault.USAGE


@pytest.mark.parametrize(
    ('command', 'args'),
    [(account, {'org': 'acme'}), (get_purchase, {'id': 'opaque-purchase'}), (list_purchases, {'org': 'acme'})],
    ids=['account', 'get-purchase', 'list-purchases'],
)
def test_a_credit_read_under_json_prints_the_refusal_envelope(command, args, platform, run_command, capsys):
    envelope = refusal(ErrorCode.forbidden, 'not a member')
    platform.answer(envelope, status=403)

    with pytest.raises(SystemExit) as exit_info:
        run_command(command, json=True, **args)

    assert json.loads(capsys.readouterr().out) == envelope
    assert exit_info.value.code == Fault.CALLER


def test_a_purchase_for_no_org_is_never_created(platform, run_command):
    platform.answer(ME_WITHOUT_MEMBERSHIPS)

    with pytest.raises(SystemExit, match='--org'):
        run_command(buy, package_id='package', transaction_key='retry-key')

    assert platform.paths == [routes.USERS_ME]


def test_credit_commands_are_in_the_real_account_tree():
    assert commands['credits'] == {
        'account': account,
        'buy': buy,
        'get-purchase': get_purchase,
        'list-purchases': list_purchases,
    }


def test_purchase_history_cli_reads_the_page_after_an_opaque_purchase(platform, run_command, capsys):
    platform.answer(NO_PURCHASES)
    run_command(list_purchases, org='acme', after='opaque-purchase', limit=3)
    assert dict(platform.request.url.params) == {'org': 'acme', 'after': 'opaque-purchase', 'limit': '3'}
    printed = capsys.readouterr()
    assert PurchaseListResponse.model_validate_json(printed.out) == NO_PURCHASES
    assert printed.err == 'org: acme (from --org)\n'


def test_sole_organization_purchase_history_cli_keeps_the_continuation(platform, run_command, capsys):
    platform.answer_by_route({
        routes.USERS_ME: (_me('user-a0'), 200),
        routes.BILLING_PURCHASES_LIST: (NO_PURCHASES, 200),
    })
    run_command(list_purchases, after='last-purchase', limit=2)
    assert platform.paths == [routes.USERS_ME, routes.BILLING_PURCHASES_LIST]
    assert dict(platform.request.url.params) == {'org': 'user-a0', 'after': 'last-purchase', 'limit': '2'}
    printed = capsys.readouterr()
    assert PurchaseListResponse.model_validate_json(printed.out) == NO_PURCHASES
    assert printed.err == 'org: user-a0 (sole organization)\n'


@pytest.mark.parametrize('limit', [0, -1, True, 1.5, '50', None])
def test_purchase_history_refuses_invalid_limit_before_http(platform, run_command, limit):
    with pytest.raises(SystemExit, match='limit'):
        run_command(list_purchases, org='acme', limit=limit)
    assert platform.seen is None


@pytest.mark.parametrize('limit', [0, -1, True, 1.5, '50', None])
def test_personal_purchase_history_refuses_invalid_limit_before_http(platform, run_command, limit):
    with pytest.raises(SystemExit, match='limit'):
        run_command(list_purchases, limit=limit)
    assert platform.seen is None


@pytest.mark.parametrize('command', [account, list_purchases])
def test_personal_credit_commands_hide_values_from_malformed_profile(command, platform, run_command):
    private_value = 'private-response-marker'
    # rules-allow: unchecked-field-name — the fixture sends bad wire data on purpose: the name is the wire contract
    platform.answer({**ME.model_dump(mode='json'), 'user_id': private_value})
    with pytest.raises(SystemExit) as raised:
        run_command(command)
    message = str(raised.value)
    assert 'the platform answered with a response the client cannot read' in message
    assert private_value not in message
    assert 'input_value' not in message
    assert platform.paths == [routes.USERS_ME]


@pytest.mark.parametrize('command', [get_purchase, list_purchases])
def test_purchase_commands_hide_values_from_malformed_response(command, platform, run_command):
    private_value = 'private-response-marker'
    # rules-allow: unchecked-field-name — the fixture sends bad wire data on purpose: the name is the wire contract
    malformed = {**PURCHASE.model_dump(mode='json'), 'initiated_by': private_value}
    if command is get_purchase:
        platform.answer(malformed)
        args = {'id': 'opaque-purchase'}
    else:
        # rules-allow: unchecked-field-name — the fixture sends bad wire data on purpose: the name is the wire contract
        platform.answer({'purchases': [malformed], 'next': None})
        args = {'org': 'acme'}
    with pytest.raises(SystemExit) as raised:
        run_command(command, **args)
    message = str(raised.value)
    assert 'the platform answered with a response the client cannot read' in message
    assert private_value not in message
    assert 'input_value' not in message


def test_purchase_history_refuses_an_answer_without_continuation(platform, run_command):
    # rules-allow: unchecked-field-name — the fixture sends bad wire data on purpose: the name is the wire contract
    platform.answer({'purchases': []})
    with pytest.raises(SystemExit, match='cannot read: missing'):
        run_command(list_purchases, org='acme')


def test_billing_response_hides_extra_field_names(platform, run_command):
    private_key = 'private-response-key-marker'
    leaking = {**PACKAGE.model_dump(mode='json'), private_key: 'private-value'}
    # rules-allow: unchecked-field-name — the fixture sends bad wire data on purpose: the name is the wire contract
    platform.answer({**ACCOUNT.model_dump(mode='json'), 'packages': [leaking]})
    with pytest.raises(SystemExit) as raised:
        run_command(account, org='acme')
    message = str(raised.value)
    assert 'the platform answered with a response the client cannot read' in message
    assert 'extra_forbidden' in message
    assert private_key not in message


def test_personal_discovery_hides_response_validation_field_names(platform, monkeypatch, run_command):
    private_key = 'private-response-key-marker'

    def unreadable_profile(self):
        raise ValidationError.from_exception_data(
            'MeResponse', [{'type': 'extra_forbidden', 'loc': (private_key,), 'input': 'private-value'}]
        )

    monkeypatch.setattr(PlatformClient, 'me', unreadable_profile)
    with pytest.raises(SystemExit) as raised:
        run_command(account)
    message = str(raised.value)
    assert 'the platform answered with a response the client cannot read' in message
    assert 'extra_forbidden' in message
    assert private_key not in message
