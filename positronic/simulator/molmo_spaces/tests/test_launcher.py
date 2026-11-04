"""The MolmoSpaces environment install, checked without uv or the network."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

from positronic.simulator.molmo_spaces import launcher

_MAIN_LOCK = Path(__file__).resolve().parents[4] / 'uv.lock'


def test_the_install_passes_the_overrides(tmp_path, monkeypatch):
    installs: list[list[str]] = []
    monkeypatch.setattr(launcher, '_MOLMO_SRC', tmp_path / 'src')
    monkeypatch.setattr(launcher, 'ensure_pinned_checkout', lambda repo, commit, src: src)
    monkeypatch.setattr(launcher, 'ensure_venv', lambda venv, python: venv / 'bin' / 'python')
    monkeypatch.setattr(launcher, 'uv_pip_install', lambda venv, args, cwd=None: installs.append(list(args)))

    launcher.ensure_molmo_venv()

    (args,) = installs
    assert args[args.index('--override') + 1] == str(launcher._MOLMO_OVERRIDES)


def _requirements(path: Path) -> list[Requirement]:
    lines = path.read_text().splitlines()
    return [Requirement(line) for line in lines if line.strip() and not line.startswith('#')]


def test_the_overrides_keep_out_evdev_on_every_platform():
    (evdev,) = _requirements(launcher._MOLMO_OVERRIDES)
    assert evdev.name == 'evdev'
    assert evdev.marker is not None
    for platform in ('linux', 'darwin'):
        assert not evdev.marker.evaluate({'sys_platform': platform})


def _constrained_version(name: str) -> Version:
    (pin,) = (requirement for requirement in _requirements(launcher._MOLMO_CONSTRAINTS) if requirement.name == name)
    (exact,) = pin.specifier
    return Version(exact.version)


def _locked_version(name: str) -> Version:
    (package,) = (package for package in tomllib.loads(_MAIN_LOCK.read_text())['package'] if package['name'] == name)
    return Version(package['version'])


def test_the_constraints_pin_certifi_no_older_than_the_main_lock():
    """certifi holds the trusted CAs, so an older release still trusts a CA that the main lock removed."""
    assert _constrained_version('certifi') >= _locked_version('certifi')
