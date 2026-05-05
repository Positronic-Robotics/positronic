#!/usr/bin/env bash
# Delete a Nebius Serverless Endpoint by name (releases compute + public IP).
#
# To pause an endpoint without releasing its static IP, use
# `nebius ai endpoint stop <id>` directly instead — `start` resumes it later.

set -euo pipefail

PARENT_ID="${NEBIUS_PARENT_ID:-project-e00f38wexevrr52b8j}"

if [ $# -ne 1 ]; then
  echo "Usage: bash workflows/nebius/stop.sh <endpoint-name>" >&2
  exit 1
fi
NAME="$1"

ID=$(nebius ai endpoint list --parent-id "$PARENT_ID" --format json \
  | jq -r --arg n "$NAME" '.items[]? | select(.metadata.name==$n) | .metadata.id')

if [ -z "$ID" ]; then
  echo "No endpoint named '$NAME' found in $PARENT_ID" >&2
  exit 1
fi

echo "Deleting endpoint '$NAME' ($ID)..."
nebius ai endpoint delete "$ID"
echo "Released endpoint and public IP."
