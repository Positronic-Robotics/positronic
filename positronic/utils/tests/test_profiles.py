import subprocess
import sys


def test_registration_reaches_servers_without_cfg_import():
    # Inference servers register the profile only transitively: they import
    # positronic.utils.checkpoints to enumerate checkpoints and never import
    # positronic.cfg.ds. A fresh interpreter importing just that utility must already
    # resolve PUBLIC, or a no-creds serve from s3://PUBLIC@... silently breaks.
    _assert_registers_public('import positronic.utils.checkpoints')


def test_registration_reaches_cfg_ds_consumers():
    # Dataset configs (e.g. cfg.ds.sim) reach the profile only via cfg.ds, which imports
    # positronic.utils purely for the registration side effect. That import looks unused and
    # a fresh interpreter importing just cfg.ds must still resolve PUBLIC, or those
    # s3://PUBLIC@... dataset paths silently break.
    _assert_registers_public('import positronic.cfg.ds')


def _assert_registers_public(import_stmt: str) -> None:
    code = f'{import_stmt}\nfrom pos3.profiles import _resolve_profile\nassert _resolve_profile("PUBLIC").public\n'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
