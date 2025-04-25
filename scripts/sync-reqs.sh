#!/bin/bash

UPGRADE_FLAG=""
if [[ "$1" == "-U" ]]; then
  UPGRADE_FLAG="-U"
fi

uv pip compile pyproject.toml -o requirements.txt $UPGRADE_FLAG
uv pip compile pyproject.toml -o requirements-hardware.txt --extra hardware --extra lerobot $UPGRADE_FLAG
uv pip compile pyproject.toml -o requirements-all.txt --all-extras $UPGRADE_FLAG
