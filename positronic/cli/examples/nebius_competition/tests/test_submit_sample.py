"""`submit_sample.py`: what its command line refuses before any request."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest
from platform_client.client import API_KEY_ENV
from platform_client.config import CONFIG_DIR_ENV, REGISTER_COMMAND, Config, write_config
from platform_client.ids import ApiKey

SCRIPT = Path(__file__).resolve().parents[1] / 'submit_sample.py'


@pytest.fixture(autouse=True)
def no_saved_record(monkeypatch, tmp_path):
    """A config directory with no record, so the box's own record never reaches a test."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))


@pytest.fixture(scope='module')
def submit_sample() -> dict[str, Any]:
    """The script's globals, loaded from its path: the examples directory is not a package."""
    return runpy.run_path(str(SCRIPT))


def test_an_empty_eval_name_is_a_cli_error_before_any_request(submit_sample: dict[str, Any], monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_secret')
    with pytest.raises(SystemExit) as exit_info:
        submit_sample['main'](['--eval=', '--policy-image=org/policy@sha256:abc'])
    assert str(exit_info.value) == "not an eval name: ''"


def test_an_empty_transaction_key_is_a_cli_error_before_any_request(submit_sample: dict[str, Any], monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_secret')
    with pytest.raises(SystemExit) as exit_info:
        submit_sample['main'](['--eval=a.eval', '--policy-image=org/policy@sha256:abc', '--transaction-key='])
    assert 'transaction_key' in str(exit_info.value)


def test_an_empty_platform_url_is_a_cli_error_before_any_request(submit_sample: dict[str, Any], monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_secret')
    with pytest.raises(SystemExit) as exit_info:
        submit_sample['main'](['--eval=a.eval', '--policy-image=org/policy@sha256:abc', '--platform-url='])
    assert 'base_url is empty' in str(exit_info.value)


def test_the_key_that_register_saved_is_used_when_the_environment_has_none(
    submit_sample: dict[str, Any], monkeypatch, tmp_path
):
    write_config(tmp_path, Config(platform_url='https://gateway.example', api_key=ApiKey('pk_live_saved')))
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    # Past the key check, the empty eval name is the first refusal.
    with pytest.raises(SystemExit) as exit_info:
        submit_sample['main'](['--eval=', '--policy-image=org/policy@sha256:abc'])
    assert str(exit_info.value) == "not an eval name: ''"


def test_with_no_key_and_no_record_the_refusal_names_the_register_command(submit_sample: dict[str, Any], monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(SystemExit) as exit_info:
        submit_sample['main'](['--eval=a.eval', '--policy-image=org/policy@sha256:abc'])
    assert REGISTER_COMMAND in str(exit_info.value)
