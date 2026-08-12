import ast
from pathlib import Path

# `positronic/vendors/<x>/` holds the shim for one third-party stack, and two import boundaries
# follow (see "Foreign components plug in through shims" in ARCHITECTURE.md). Nothing outside
# `positronic/vendors/` imports from it: core defines the contract and a vendor adapts to it, so an
# import the other way inverts the dependency and drags the vendor's optional extra into code that
# must work without it. And no vendor imports another: each shim answers to its own upstream, on its
# own pinned deps and often its own interpreter — a helper two of them need belongs in core.

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PACKAGE_ROOT.parent
_VENDORS_ROOT = _PACKAGE_ROOT / 'vendors'
_VENDORS_PACKAGE = 'positronic.vendors'
# First-party trees held to the boundary. The vendors tree itself is walked separately, against the
# cross-vendor rule.
_CORE_ROOTS = (_PACKAGE_ROOT, _REPO_ROOT / 'pimm', _REPO_ROOT / 'utilities')


def _package_parts(path: Path) -> tuple[str, ...]:
    # Dropping the last component covers both cases: a module's package is its directory, and an
    # `__init__.py`'s package is the directory itself (relative imports resolve against it).
    return path.relative_to(_REPO_ROOT).with_suffix('').parts[:-1]


def _imports(path: Path):
    """Yields every imported module name in `path` as (absolute dotted name, line number).

    A `from` import yields one name per alias, joined onto the base module, so a submodule imported
    as `from positronic import vendors` resolves to `positronic.vendors`. Relative imports are
    resolved against the file's package.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                assert node.module is not None  # an absolute `from` import always names a module
                base = node.module.split('.')
            else:
                package = _package_parts(path)
                base = list(package[: len(package) - (node.level - 1)])
                if node.module:
                    base += node.module.split('.')
            for alias in node.names:
                yield '.'.join([*base, alias.name]), node.lineno


def _vendor_imports(path: Path):
    for name, lineno in _imports(path):
        if name == _VENDORS_PACKAGE or name.startswith(_VENDORS_PACKAGE + '.'):
            yield name, lineno


def test_nothing_outside_vendors_imports_a_vendor():
    offenders = []
    for root in _CORE_ROOTS:
        for path in sorted(root.rglob('*.py')):
            if not path.is_relative_to(_VENDORS_ROOT):
                offenders += [
                    f'{path.relative_to(_REPO_ROOT)}:{lineno}: imports {name}' for name, lineno in _vendor_imports(path)
                ]
    assert not offenders, (
        'Only code under `positronic/vendors/` may import from it '
        '(see "Foreign components plug in through shims" in ARCHITECTURE.md):\n' + '\n'.join(offenders)
    )


def test_no_vendor_imports_another_vendor():
    offenders = []
    for path in sorted(_VENDORS_ROOT.rglob('*.py')):
        rel = path.relative_to(_VENDORS_ROOT)
        # Files at the vendors root belong to no vendor, so for them every vendor is foreign.
        vendor = rel.parts[0] if len(rel.parts) > 1 else None
        for name, lineno in _vendor_imports(path):
            # A bare `positronic.vendors` import names the shared parent, which no vendor owns — allowed.
            suffix = name.removeprefix(_VENDORS_PACKAGE + '.')
            if suffix != name and suffix.split('.')[0] != vendor:
                offenders.append(f'{path.relative_to(_REPO_ROOT)}:{lineno}: imports {name}')
    assert not offenders, (
        'A vendor may not import another vendor — a helper both need belongs in core '
        '(see "Foreign components plug in through shims" in ARCHITECTURE.md):\n' + '\n'.join(offenders)
    )
