from unittest import mock

import pytest

pytest.importorskip('openpi_client')

from positronic.vendors.openpi.server import OpenpiSubprocess  # noqa: E402


def _subprocess_env(monkeypatch) -> dict[str, str]:
    """The environment ``start`` gives the openpi subprocess."""
    monkeypatch.setattr(OpenpiSubprocess, '_wait_for_ready', lambda self, on_progress: None)
    with mock.patch('subprocess.Popen') as popen:
        OpenpiSubprocess(checkpoint_dir='/checkpoints/exp/1000', config_name='pi05_positronic_lowmem').start()
    return popen.call_args.kwargs['env']


def test_the_subprocess_does_not_preallocate_the_gpu(monkeypatch):
    """JAX takes ~75% of the device at its first use, which leaves a second server on that GPU none."""
    monkeypatch.delenv('XLA_PYTHON_CLIENT_PREALLOCATE', raising=False)

    assert _subprocess_env(monkeypatch)['XLA_PYTHON_CLIENT_PREALLOCATE'] == 'false'


def test_the_preallocation_setting_of_the_environment_reaches_the_subprocess(monkeypatch):
    monkeypatch.setenv('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')

    assert _subprocess_env(monkeypatch)['XLA_PYTHON_CLIENT_PREALLOCATE'] == 'true'
