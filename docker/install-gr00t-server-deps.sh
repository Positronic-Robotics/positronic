#!/usr/bin/env bash
set -euo pipefail

# Preserve every package version chosen by the native model environment.
/opt/gr00t-venv/bin/python - <<'PY'
import tomllib
from importlib.metadata import distributions
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

Path('/tmp/gr00t-constraints.txt').write_text('\n'.join(
    f'{dist.metadata["Name"]}=={dist.version}' for dist in distributions()
))
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
  configuronic==0.7.0 pos3==0.3.1
rm /tmp/gr00t-constraints.txt /tmp/gr00t-requirements.txt
