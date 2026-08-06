#!/usr/bin/env bash
# Delete a Nebius Serverless Endpoint by name (releases its compute).
#
# Deleting retires the endpoint's managed URL; a re-created endpoint of the same
# name gets a new one. To keep the URL, use `nebius ai endpoint stop <id>`
# directly instead — it releases the compute too, and `start` resumes on the
# same URL.

set -euo pipefail
source "$(dirname "$0")/common.sh"

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
echo "Released endpoint and its managed URL."
