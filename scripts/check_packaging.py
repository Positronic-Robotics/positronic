"""Verify all core modules are importable from an installed package.

Catches missing package-data files (e.g. .urdf, .xml, templates) that exist
in the source tree but aren't included in the built wheel.  Must be run
against a pip-installed positronic, not the source checkout.

Usage:
    uv venv /tmp/pkg-test --python 3.11
    uv pip install --python /tmp/pkg-test . jinja2
    /tmp/pkg-test/bin/python scripts/check_packaging.py
"""

import importlib
import pkgutil
import sys

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
