"""The MolmoSpaces environment install, checked without uv or the network."""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
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


# A server stand-in: it accepts TCP connections on the port of its command until it is terminated.
_POPEN = subprocess.Popen
_LISTEN = 'import socket, sys, time; s = socket.create_server(("localhost", int(sys.argv[1]))); time.sleep(60)'


def _served_command(hook: Path | None, tmp_path: Path, monkeypatch) -> list[str]:
    """The command that ``serve_molmo_spaces`` starts its server with."""
    commands: list[list[str]] = []

    def listen(command: list[str], env: dict[str, str]) -> subprocess.Popen:
        commands.append(command)
        return _POPEN([sys.executable, '-c', _LISTEN, command[command.index('--port') + 1]])

    monkeypatch.setenv(launcher.mapping.ASSETS_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(launcher, 'ensure_molmo_venv', lambda: Path(sys.executable))
    monkeypatch.setattr(launcher.subprocess, 'Popen', listen)
    with launcher.serve_molmo_spaces(hook=hook):
        pass
    (command,) = commands
    return command


def test_a_server_launched_with_a_hook_gets_the_hook_file(tmp_path, monkeypatch):
    without = _served_command(None, tmp_path, monkeypatch)
    with_hook = _served_command(tmp_path / 'hook.py', tmp_path, monkeypatch)

    assert '--hook' not in without
    assert with_hook[with_hook.index('--hook') + 1] == str(tmp_path / 'hook.py')


class _Exited:
    def wait(self) -> int:
        return 0


def test_the_launcher_command_runs_a_server_on_its_port_with_its_hook_until_the_server_exits(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setenv(launcher.mapping.ASSETS_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(launcher, 'ensure_molmo_venv', lambda: Path(sys.executable))
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda command, env: commands.append(command) or _Exited())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['launcher', '--port', '8765', '--hook', 'hook.py'])

    with pytest.raises(SystemExit) as exited:
        launcher.main()

    assert exited.value.code == 0
    (command,) = commands
    assert command[command.index('--port') + 1] == '8765'
    assert command[command.index('--hook') + 1] == str((tmp_path / 'hook.py').resolve())
