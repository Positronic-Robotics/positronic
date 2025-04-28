#!/bin/bash
#
# Sync requirements with pyproject.toml
#
# Usage: ./sync-reqs.sh [-U]
#
# -U: Additionaly upgrade all packages to the latest version.
#
UPGRADE_FLAG=""
if [[ "$1" == "-U" ]]; then
  UPGRADE_FLAG="-U"
fi

uv pip compile pyproject.toml -o requirements.txt $UPGRADE_FLAG
uv pip compile pyproject.toml -o requirements-hardware.txt --extra hardware --extra lerobot $UPGRADE_FLAG
uv pip compile pyproject.toml -o requirements-all.txt --all-extras $UPGRADE_FLAG
