import ast
from pathlib import Path

import positronic.dataset

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
    """The dataset core stays agnostic to telemetry: no module under ``positronic/dataset`` imports
    ``positronic.telemetry``, its vocabulary, or ``opentelemetry``. Timing rides in as an opaque
    ``io_context`` context factory (default inert), named after the work it brackets rather than after what
    the caller does with it."""
    offenders = []
    for path in _DATASET_ROOT.rglob('*.py'):
        if 'tests' in path.parts:
            continue
        if _imports_telemetry(path.read_text()):
            offenders.append(str(path.relative_to(_DATASET_ROOT)))
    assert not offenders, f'dataset modules must not import telemetry: {offenders}'
