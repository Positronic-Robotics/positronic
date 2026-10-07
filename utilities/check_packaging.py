"""Verify all core modules are importable from an installed package.

Catches missing package-data files (e.g. .urdf, .xml, templates) that exist
in the source tree but aren't included in the built wheel.  Must be run
against a pip-installed positronic, not the source checkout.
"""

import importlib
import pkgutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for module_name, source_dir in (
    ('positronic', root / 'positronic'),
    ('platform_client', root / 'client' / 'platform_client'),
    ('positronic_wire', root / 'wire' / 'positronic_wire'),
    ('positronic_model_server', root / 'model_server' / 'positronic_model_server'),
    ('eval_vocabulary', root / 'vocabulary' / 'eval_vocabulary'),
):
    module = importlib.import_module(module_name)
    for module_path in module.__path__:
        origin = Path(module_path).resolve()
        if origin.is_relative_to(source_dir):
            raise SystemExit(f'FAIL: {module_name} imports from the source checkout: {origin}')

# Packages that downstream consumers (phail-website, training scripts) import.
CORE_PACKAGES = [
    'positronic.cfg',
    'positronic.dataset',
    'positronic.server',
    'positronic.policy',
    'positronic.utils',
    'positronic.geom',
]

SKIP_PARTS = {'tests', 'vendors'}

failures = []
count = 0

for pkg_name in CORE_PACKAGES:
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception as e:
        failures.append((pkg_name, e))
        continue
    if not hasattr(pkg, '__path__'):
        count += 1
        continue
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg_name + '.'):
        if any(s in info.name.split('.') for s in SKIP_PARTS):
            continue
        try:
            importlib.import_module(info.name)
            count += 1
        except Exception as e:
            failures.append((info.name, e))

if failures:
    for name, e in failures:
        print(f'FAIL: {name}: {e}')
    sys.exit(1)

print(f'{count} modules imported OK')
