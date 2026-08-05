"""Regenerate the checked-in `mujoco` type stubs from the installed package.

`mujoco` ships no `.pyi` and no `py.typed`, and re-exports its whole API from binary extension
modules, so a type checker resolves none of it — `mj.MjModel`, `mj.mj_forward` and `mj.mjtObj` all
read as attribute errors. `pybind11-stubgen` recovers the signatures from the pybind11 docstrings;
the result is committed under `stubs/` and reached through `stubPath` in `pyproject.toml`.

Run this after changing the pinned `mujoco` version. `--check` regenerates into a temporary
directory and compares instead of rewriting the stubs.
"""

import argparse
import difflib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco

PACKAGE = 'mujoco'
ROOT = Path(__file__).resolve().parent.parent
STUBS = ROOT / 'stubs' / PACKAGE

# Module constants read off the machine that generated the stubs: the install prefix, the handles of
# the plugins loaded into this process, the host OS. A stub needs their type and nothing else —
# keeping the values would bake one contributor's venv path into the repo and differ on every run.
MACHINE_SPECIFIC = ('HEADERS_DIR', 'PLUGINS_DIR', 'PLUGIN_HANDLES', '_SYSTEM')


def _extension_modules() -> set[str]:
    """Module names a type checker cannot read: the compiled ones, plus the package root.

    The rest of `mujoco` is plain Python that the checker reads from source, so stubbing it would
    only add files that go stale.
    """
    package_dir = Path(mujoco.__file__).parent
    compiled = {p.name.split('.')[0] for p in package_dir.iterdir() if p.suffix in ('.so', '.pyd', '.dylib')}
    return compiled | {'__init__'}


def _normalise(text: str) -> str:
    """Strip machine-specific values and trailing whitespace, and end on a single newline."""
    lines = []
    for line in text.split('\n'):
        name, _, rest = line.rstrip().partition(':')
        if name in MACHINE_SPECIFIC:
            line = f'{name}: {rest.split("=")[0].split("#")[0].strip()}'
        lines.append(line.rstrip())
    return '\n'.join(lines).rstrip('\n') + '\n'


def generate() -> dict[str, str]:
    """Return the stub tree as {filename: contents}, generated fresh from the installed package."""
    # pybind11 leaks raw C++ names into a handful of mujoco docstrings, and pybind11-stubgen renders each
    # as `...` (Unknown). Naming the two patterns keeps the blast radius visible: any other unparseable
    # signature fails generation instead of silently degrading to Unknown.
    cpp_expressions = r'(mujoco::python|std)::.*'
    cpp_names = r'mjs\w+_'
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, '-m', 'pybind11_stubgen', '--output-dir', tmp, '--exit-code']
        cmd += ['--ignore-invalid-expressions', cpp_expressions, '--ignore-unresolved-names', cpp_names, PACKAGE]
        subprocess.run(cmd, check=True)
        keep = _extension_modules()
        generated = Path(tmp) / PACKAGE
        return {p.name: _normalise(p.read_text()) for p in sorted(generated.glob('*.pyi')) if p.stem in keep}


def write(stubs: dict[str, str]) -> None:
    shutil.rmtree(STUBS, ignore_errors=True)
    STUBS.mkdir(parents=True)
    for name, text in stubs.items():
        (STUBS / name).write_text(text)


def diff(stubs: dict[str, str]) -> list[str]:
    """Unified diff of the committed stubs against freshly generated ones, empty when they match."""
    committed = {p.name: p.read_text() for p in sorted(STUBS.glob('*.pyi'))} if STUBS.is_dir() else {}
    out = []
    for name in sorted(set(committed) | set(stubs)):
        out += difflib.unified_diff(
            committed.get(name, '').splitlines(True),
            stubs.get(name, '').splitlines(True),
            f'committed/{name}',
            f'generated/{name}',
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='report staleness instead of rewriting the stubs')
    args = parser.parse_args()

    stubs = generate()
    if not args.check:
        write(stubs)
        print(f'wrote {len(stubs)} stub files to {STUBS.relative_to(ROOT)}')
        return 0

    stale = diff(stubs)
    if not stale:
        return 0
    shown = 40  # enough to see what moved; the whole diff is the regenerated tree itself
    sys.stdout.writelines(stale[:shown])
    if len(stale) > shown:
        print(f'... and {len(stale) - shown} more diff lines')
    print(f'{STUBS.relative_to(ROOT)} does not match the installed {PACKAGE} {mujoco.__version__}.')
    print(f'Regenerate: uv run --extra dev python {Path(__file__).relative_to(ROOT)}')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
