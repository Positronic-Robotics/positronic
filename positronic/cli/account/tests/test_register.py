"""`positronic account register`, over a stub platform transport."""

import json
import os

import pytest
from platform_client import config as config_module
from platform_client import routes
from platform_client.billing import CREDIT_SCALE, Tariff
from platform_client.config import CONFIG_FILENAME, Config, config_dir, read_config, write_config
from platform_client.ids import ApiKey

from positronic.cli.account import commands
from positronic.cli.account import gateway as gateway_module
from positronic.cli.account.credits import account, buy, purchase, purchases
from positronic.cli.account.register import register


def test_register_saves_the_minted_key_in_the_record(platform, run_command, capsys):
    platform.answer({
        'user_id': 'a0',
        'artifact_location': 's3://b/users/a0/',
        'api_key': 'pk_new',
        'key_status': 'created',
    })

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
    platform.answer({
        'user_id': 'a0',
        'artifact_location': 's3://b/users/a0/',
        'api_key': 'pk_new',
        'key_status': 'created',
    })

    run_command(register, platform_url='http://other.test')

    saved = read_config(config_dir(os.environ))
    assert saved is not None and saved.platform_url.startswith('http://other.test')


def test_a_registration_goes_to_the_platform_it_names_whatever_the_record_holds(platform, run_command, monkeypatch):
    # A registration sends no key, so the record has nothing to say about where it goes.
    write_config(config_dir(os.environ), Config(platform_url='http://gateway.test', api_key=ApiKey('pk_old')))
    monkeypatch.delenv(gateway_module.API_KEY_ENV)
    monkeypatch.delenv(gateway_module.API_URL_ENV)
    platform.answer({'user_id': 'a0', 'artifact_location': 's3://b/users/a0/', 'key_status': 'existing'})

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


def test_a_registration_naming_its_platform_reads_no_key_and_no_record(platform, run_command, monkeypatch):
    # A blank key variable and a file that is no record each end a command that needs a key; a
    # registration needs none, and names its platform, so neither is read.
    (config_dir(os.environ)).mkdir(parents=True)
    (config_dir(os.environ) / CONFIG_FILENAME).write_text('not a record')
    monkeypatch.setenv(gateway_module.API_KEY_ENV, '  ')
    platform.answer({'user_id': 'a0', 'artifact_location': 's3://b/users/a0/', 'key_status': 'existing'})

    run_command(register, platform_url='http://other.test')

    assert platform.request.url.path == routes.USERS_REGISTER


def test_a_registration_naming_its_platform_replaces_a_malformed_record(platform, run_command, monkeypatch):
    # A registration sends no key and names its platform, so the record has nothing to give it.
    # Reading one lets a malformed record refuse the very command that rewrites it.
    config_dir(os.environ).mkdir(parents=True)
    (config_dir(os.environ) / CONFIG_FILENAME).write_text('not a record')
    monkeypatch.delenv(gateway_module.API_KEY_ENV)
    monkeypatch.delenv(gateway_module.API_URL_ENV)
    platform.answer({
        'user_id': 'a0',
        'artifact_location': 's3://b/users/a0/',
        'api_key': 'pk_new',
        'key_status': 'created',
    })

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
    platform.answer({'user_id': 'a0', 'artifact_location': 's3://b/users/a0/', 'key_status': 'existing'})

    run_command(register)

    assert 'no key issued' in capsys.readouterr().out
    # A repeat registration mints no key, so no record is written.
    assert read_config(config_dir(os.environ)) is None


PACKAGE = {'id': 'package', 'credit_units': CREDIT_SCALE, 'amount_minor': 17, 'currency': 'jpy'}
PURCHASE = {
    'id': 'opaque-purchase',
    'package': PACKAGE,
    'initiated_by': 'a0',
    'created_at': '2026-03-04T05:06:07Z',
    'checkout_url': 'https://checkout.stripe.com/accepted',
}


def test_account_prints_exact_units_and_operator_configured_package_terms(platform, run_command, capsys):
    body = {
        'org': 'acme',
        'mode': 'prepaid',
        'billing_role': 'spender',
        'balance': {'posted_units': CREDIT_SCALE, 'reserved_units': 1, 'available_units': CREDIT_SCALE - 1},
        'tariff': Tariff.for_rates(10_000_000_000, CREDIT_SCALE).model_dump(),
        'packages': [PACKAGE],
    }
    platform.answer(body)
    run_command(account, org='acme')
    assert platform.request.url.path == routes.BILLING_ACCOUNT
    assert platform.request.url.params['org'] == 'acme'
    assert json.loads(capsys.readouterr().out) == body


def test_buy_sends_a_named_retry_key_and_prints_the_owned_checkout(platform, run_command, capsys):
    platform.answer(PURCHASE)
    run_command(buy, org='acme', package_id='package', transaction_key='retry-key')
    assert platform.request.url.path == routes.BILLING_PURCHASES_CREATE
    assert platform.body == {'org': 'acme', 'package_id': 'package', 'transaction_key': 'retry-key'}
    assert json.loads(capsys.readouterr().out)['checkout_url'] == PURCHASE['checkout_url']


@pytest.mark.parametrize('field', ['org', 'package_id', 'transaction_key'])
def test_buy_refuses_invalid_input_before_http(field, platform, run_command):
    args = {'org': 'acme', 'package_id': 'package', 'transaction_key': 'retry-key', field: ''}
    with pytest.raises(SystemExit):
        run_command(buy, **args)
    assert platform.seen is None


def test_purchase_reads_an_opaque_id_without_creating_another_checkout(platform, run_command, capsys):
    platform.answer({**PURCHASE, 'checkout_url': None, 'review_reason': 'payment review'})
    run_command(purchase, id='opaque-purchase')
    assert platform.request.method == 'GET'
    assert platform.request.url.path == routes.BILLING_PURCHASES_GET
    assert platform.request.url.params['id'] == 'opaque-purchase'
    assert json.loads(capsys.readouterr().out)['review_reason'] == 'payment review'


def test_purchases_reads_the_named_member_account(platform, run_command, capsys):
    platform.answer({'purchases': [PURCHASE]})
    run_command(purchases, org='acme')
    assert platform.request.url.path == routes.BILLING_PURCHASES_LIST
    assert platform.request.url.params['org'] == 'acme'
    assert len(json.loads(capsys.readouterr().out)['purchases']) == 1


def test_credit_commands_are_in_the_real_account_tree():
    assert commands['credits'] == {'account': account, 'buy': buy, 'purchase': purchase, 'purchases': purchases}
