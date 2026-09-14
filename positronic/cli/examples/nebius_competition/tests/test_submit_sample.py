"""`submit_sample.py`: what its command line refuses before any request."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'submit_sample.py'


@pytest.fixture(scope='module')
def submit_sample() -> dict[str, Any]:
    """The script's globals, loaded from its path: the examples directory is not a package."""
    return runpy.run_path(str(SCRIPT))


def test_an_empty_eval_name_is_a_cli_error_before_any_request(submit_sample: dict[str, Any]):
    with pytest.raises(SystemExit) as exit_info:
        submit_sample['main'](['--eval=', '--policy-image=org/policy@sha256:abc'])
    assert str(exit_info.value) == "not an eval name: ''"
