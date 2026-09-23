"""`submit_sample.py`: what its command line refuses before any request."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest
from platform_client.client import API_KEY_ENV

SCRIPT = Path(__file__).resolve().parents[1] / 'submit_sample.py'


@pytest.fixture(scope='module')
def submit_sample() -> dict[str, Any]:
    """The script's globals, loaded from its path: the examples directory is not a package."""
    return runpy.run_path(str(SCRIPT))


def test_an_empty_eval_name_is_a_cli_error_before_any_request(submit_sample: dict[str, Any], monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_secret')
    with pytest.raises(SystemExit) as exit_info:
        submit_sample['main'](['--eval=', '--policy-image=org/policy@sha256:abc', '--policy-wire=websocket'])
    assert str(exit_info.value) == "not an eval name: ''"


def test_an_empty_transaction_key_is_a_cli_error_before_any_request(submit_sample: dict[str, Any], monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_secret')
    with pytest.raises(SystemExit) as exit_info:
        submit_sample['main']([
            '--eval=a.eval',
            '--policy-image=org/policy@sha256:abc',
            '--policy-wire=websocket',
            '--transaction-key=',
        ])
    assert 'transaction_key' in str(exit_info.value)


def test_an_empty_platform_url_is_a_cli_error_before_any_request(submit_sample: dict[str, Any], monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, 'pk_live_secret')
    with pytest.raises(SystemExit) as exit_info:
        submit_sample['main']([
            '--eval=a.eval',
            '--policy-image=org/policy@sha256:abc',
            '--policy-wire=websocket',
            '--platform-url=',
        ])
    assert 'base_url is empty' in str(exit_info.value)
