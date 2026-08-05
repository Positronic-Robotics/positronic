"""Regenerate the checked-in `mujoco` type stubs from the installed package.

`mujoco` ships no `.pyi` and no `py.typed`, and re-exports its whole API from binary extension
modules, so a type checker resolves none of it — `mj.MjModel`, `mj.mj_forward` and `mj.mjtObj` all
read as attribute errors. `pybind11-stubgen` recovers the signatures from the pybind11 docstrings;
the result is committed under `stubs/` and reached through `stubPath` in `pyproject.toml`.

Run this after changing the pinned `mujoco` version. `--check` regenerates into a temporary
directory and compares instead of rewriting the stubs.
"""

import argparse
import ast
import difflib
import keyword
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


def _spans(text: str):
    """Yield (index, char, depth) over `text`, with characters inside string literals skipped."""
    depth, quote = 0, ''
    for i, char in enumerate(text):
        if quote:
            if char == quote:
                quote = ''
            continue
        if char in '"\'':
            quote = char
            continue
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        yield i, char, depth


def _split_params(params: str) -> list[str]:
    """Split a parameter list at its own commas — not at those inside annotations or defaults."""
    out, start = [], 0
    for i, char, depth in _spans(params):
        if char == ',' and depth == 0:
            out.append(params[start:i])
            start = i + 1
    return [*out, params[start:]]


def _default_at(param: str) -> int:
    """Index of the `=` introducing this parameter's default, or -1 when it has none."""
    return next((i for i, char, depth in _spans(param) if char == '=' and depth == 0), -1)


def _drop_unusable_defaults(line: str) -> str:
    """Drop a positional parameter's default when a later positional parameter has none.

    pybind11 emits such a default, which Python's grammar rejects; the call it would describe is
    impossible anyway, since the required argument after it leaves nothing to skip.
    """
    opening = line.index('(', line.index('def '))
    closing = next((i for i, char, depth in _spans(line[opening:]) if char == ')' and depth == 0), -1)
    if closing == -1:  # a signature this walker cannot read
        return line
    closing += opening
    params = [p.strip() for p in _split_params(line[opening + 1 : closing])]
    # Keyword-only parameters end the run: past a `*` the grammar already allows either order.
    positional = next((i for i, p in enumerate(params) if p.startswith('*')), len(params))

    required = False
    for i in reversed(range(positional)):
        default = _default_at(params[i])
        if default == -1:
            required = True
        elif required:
            params[i] = params[i][:default].strip()
    return line[: opening + 1] + ', '.join(params) + line[closing:]


def _is_keyword_attribute(line: str) -> bool:
    """Whether `line` annotates an attribute whose name is a Python keyword.

    pybind11 exposes `MjVisual.global`, which no Python source can name and no parser accepts. The
    `global_` alias beside it carries the same type, so the line is dropped rather than renamed.
    """
    name, _, annotation = line.strip().partition(':')
    return keyword.iskeyword(name) and bool(annotation.strip())


def _parseable(text: str, name: str) -> str:
    """Return `text` with every construct Python's grammar rejects removed, or raise if one stays."""
    lines = []
    for line in text.split('\n'):
        if line.lstrip().startswith('def '):
            lines.append(_drop_unusable_defaults(line))
        elif not _is_keyword_attribute(line):
            lines.append(line)
    text = '\n'.join(lines)
    ast.parse(text, filename=name)
    return text


HEADER = f"""\
# Generated from the installed `{PACKAGE}` by utilities/generate_mujoco_stubs.py — do not edit.
# Regenerate: uv run --locked --extra dev python utilities/generate_mujoco_stubs.py
"""


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
        stubs = {p.name: p.read_text() for p in sorted(generated.glob('*.pyi')) if p.stem in keep}
        return {name: HEADER + _parseable(_normalise(text), name) for name, text in stubs.items()}


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
