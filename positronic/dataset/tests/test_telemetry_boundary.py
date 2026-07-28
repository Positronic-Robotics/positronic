import ast
from pathlib import Path

import positronic.dataset

_DATASET_ROOT = Path(positronic.dataset.__file__).parent


def _is_telemetry_path(path: str) -> bool:
    """True if a dotted import path names a telemetry module: ``positronic.telemetry``, a bare
    ``telemetry``, or ``opentelemetry`` (or a submodule of any)."""
    return (
        path == 'telemetry'
        or path.startswith('telemetry.')
        or path == 'opentelemetry'
        or path.startswith('opentelemetry.')
        or path == 'positronic.telemetry'
        or path.startswith('positronic.telemetry.')
    )


def _imports_telemetry(source: str) -> bool:
    """True if the module imports ``positronic.telemetry``, a bare ``telemetry``, or ``opentelemetry``."""
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
    """The dataset core stays agnostic to telemetry: no module under ``positronic/dataset`` imports
    ``positronic.telemetry`` or ``opentelemetry``. Timing rides in as an opaque ``telemetry_span`` context
    factory (default inert) — naming the injected span a telemetry span is fine; importing telemetry is not."""
    offenders = []
    for path in _DATASET_ROOT.rglob('*.py'):
        if 'tests' in path.parts:
            continue
        if _imports_telemetry(path.read_text()):
            offenders.append(str(path.relative_to(_DATASET_ROOT)))
    assert not offenders, f'dataset modules must not import telemetry: {offenders}'
