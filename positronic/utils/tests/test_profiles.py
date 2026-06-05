import subprocess
import sys

import positronic.cfg.phail.v1_0 as phail_v1_0


def test_phail_models_use_public_profile_url():
    urls = list(vars(phail_v1_0.models).values())
    assert urls
    for url in urls:
        assert url.startswith('s3://PUBLIC@positronic-public/phail/v1.0/models/')


def test_registration_reaches_servers_without_cfg_import():
    # Inference servers register the profile only transitively: they import
    # positronic.utils.checkpoints to enumerate checkpoints and never import
    # positronic.cfg.ds. A fresh interpreter importing just that utility must already
    # resolve PUBLIC, or a no-creds serve from s3://PUBLIC@... silently breaks.
    _assert_registers_public('import positronic.utils.checkpoints')


def test_registration_reaches_cfg_ds_consumers():
    # Dataset configs (e.g. cfg.ds.sim) reach the profile only via cfg.ds, which imports
    # positronic.utils for the registration side effect. A fresh interpreter importing just
    # cfg.ds must resolve PUBLIC, or those s3://PUBLIC@... dataset paths silently break.
    _assert_registers_public('import positronic.cfg.ds')


def _assert_registers_public(import_stmt: str) -> None:
    code = f'{import_stmt}\nfrom pos3.profiles import _resolve_profile\nassert _resolve_profile("PUBLIC").public\n'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
