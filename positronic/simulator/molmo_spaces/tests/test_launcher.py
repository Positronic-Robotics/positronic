"""The MolmoSpaces environment install, checked without uv or the network."""

from packaging.requirements import Requirement

from positronic.simulator.molmo_spaces import launcher


def test_the_install_passes_the_overrides(tmp_path, monkeypatch):
    installs: list[list[str]] = []
    monkeypatch.setattr(launcher, '_MOLMO_SRC', tmp_path / 'src')
    monkeypatch.setattr(launcher, 'ensure_pinned_checkout', lambda repo, commit, src: src)
    monkeypatch.setattr(launcher, 'ensure_venv', lambda venv, python: venv / 'bin' / 'python')
    monkeypatch.setattr(launcher, 'uv_pip_install', lambda venv, args, cwd=None: installs.append(list(args)))

    launcher.ensure_molmo_venv()

    (args,) = installs
    assert args[args.index('--override') + 1] == str(launcher._MOLMO_OVERRIDES)


def _overrides() -> list[Requirement]:
    lines = launcher._MOLMO_OVERRIDES.read_text().splitlines()
    return [Requirement(line) for line in lines if line.strip() and not line.startswith('#')]


def test_the_overrides_keep_out_evdev_on_every_platform():
    (evdev,) = _overrides()
    assert evdev.name == 'evdev'
    assert evdev.marker is not None
    for platform in ('linux', 'darwin'):
        assert not evdev.marker.evaluate({'sys_platform': platform})
