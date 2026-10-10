import ast
from importlib.util import resolve_name
from pathlib import Path

import positronic.dataset
from pimm.time import EMITTED_WALL, EMITTED_WORLD, RECEIVED_WALL, RECEIVED_WORLD
from positronic.dataset.vector import RECORDED_TIME
from positronic.recording import HARNESS_WALL, HARNESS_WORLD

_DATASET_ROOT = Path(positronic.dataset.__file__).parent


_TELEMETRY_ROOTS = ('telemetry', 'opentelemetry', 'positronic.telemetry', 'positronic.telemetry_keys')


def _is_telemetry_path(path: str) -> bool:
    """True if a dotted import path names a telemetry module — the mechanism, its vocabulary, a bare
    ``telemetry``, or ``opentelemetry`` (or a submodule of any)."""
    return any(path == root or path.startswith(f'{root}.') for root in _TELEMETRY_ROOTS)


def _imports_telemetry(source: str) -> bool:
    """True if the module imports a telemetry module, a bare ``telemetry``, or ``opentelemetry``."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(_is_telemetry_path(alias.name) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            # `from positronic import telemetry` binds the submodule via the imported name, so test both
            # the module path and each `module.name` combination.
            if _is_telemetry_path(module) or any(_is_telemetry_path(f'{module}.{alias.name}') for alias in node.names):
                return True
    return False


def test_dataset_package_has_no_telemetry_dependency():
    offenders = []
    for path in _DATASET_ROOT.rglob('*.py'):
        if 'tests' in path.parts:
            continue
        if _imports_telemetry(path.read_text()):
            offenders.append(str(path.relative_to(_DATASET_ROOT)))
    assert not offenders, f'dataset modules must not import telemetry: {offenders}'


def test_dataset_has_no_application_timeline_dependencies():
    clock_names = {
        HARNESS_WORLD,
        HARNESS_WALL,
        EMITTED_WORLD,
        EMITTED_WALL,
        RECEIVED_WORLD,
        RECEIVED_WALL,
        RECORDED_TIME,
    }
    time_api = {'Time', 'validate_timeline', 'validate_timelines'}
    offenders = []
    for path in _DATASET_ROOT.rglob('*.py'):
        if 'tests' in path.parts:
            continue
        legacy_reader = path == _DATASET_ROOT / 'vector.py'
        package = '.'.join(('positronic', 'dataset', *path.relative_to(_DATASET_ROOT).parts[:-1]))
        for node in ast.walk(ast.parse(path.read_text())):
            imports = []
            line = 0
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
                line = node.lineno
            elif isinstance(node, ast.ImportFrom):
                module = resolve_name('.' * node.level + (node.module or ''), package)
                imports = [f'{module}.{alias.name}' for alias in node.names]
                line = node.lineno
            for name in imports:
                if (
                    name == 'positronic.recording'
                    or name.startswith('positronic.recording.')
                    or name == 'pimm.time'
                    or (name.startswith('pimm.time.') and name.removeprefix('pimm.time.') not in time_api)
                    or (name.endswith('.RECORDED_TIME') and not legacy_reader)
                ):
                    offenders.append(f'{path.relative_to(_DATASET_ROOT)}:{line}: imports {name}')
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in clock_names
                and not (legacy_reader and node.value == RECORDED_TIME)
            ):
                offenders.append(f'{path.relative_to(_DATASET_ROOT)}:{node.lineno}: names {node.value}')
            if isinstance(node, ast.Attribute) and node.attr == 'RECORDED_TIME' and not legacy_reader:
                offenders.append(f'{path.relative_to(_DATASET_ROOT)}:{node.lineno}: uses the legacy timeline')
    assert not offenders, 'Dataset must receive timeline names from its callers:\n' + '\n'.join(offenders)
