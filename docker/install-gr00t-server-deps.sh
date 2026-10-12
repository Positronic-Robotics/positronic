#!/usr/bin/env bash
set -euo pipefail

UV_OFFLINE=0 uv export --project /opt/positronic-packages --locked --no-dev --no-emit-workspace \
  --no-hashes --no-annotate --no-header --output-file /tmp/gr00t-locked-requirements.txt > /dev/null

# Preserve native versions; use the repository lock for packages absent from the native environment.
/opt/gr00t-venv/bin/python - <<'PY'
import tomllib
from importlib.metadata import distributions
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

native = {canonicalize_name(dist.metadata['Name']): dist.version for dist in distributions()}
constraints = [f'{name}=={version}' for name, version in native.items()]
constraints.extend(
    line for line in Path('/tmp/gr00t-locked-requirements.txt').read_text().splitlines()
    if canonicalize_name(Requirement(line).name) not in native
)
Path('/tmp/gr00t-constraints.txt').write_text('\n'.join(constraints))
projects = [
    tomllib.loads((Path('/opt/positronic-packages') / name / 'pyproject.toml').read_text())['project']
    for name in ('wire', 'model_server')
]
local_names = {canonicalize_name(project['name']) for project in projects}
requirements = []
for project in projects:
    requirements.extend(project['dependencies'])
    for extra in ('websocket', 'grpc'):
        requirements.extend(project['optional-dependencies'][extra])
Path('/tmp/gr00t-requirements.txt').write_text('\n'.join(
    requirement for requirement in requirements if canonicalize_name(Requirement(requirement).name) not in local_names
))
PY
UV_OFFLINE=0 uv pip install --python /opt/gr00t-venv/bin/python --constraint /tmp/gr00t-constraints.txt \
  --requirements /tmp/gr00t-requirements.txt \
  configuronic pos3
rm /tmp/gr00t-constraints.txt /tmp/gr00t-requirements.txt /tmp/gr00t-locked-requirements.txt
