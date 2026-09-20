#!/usr/bin/env bash
# Read a public image's manifest digest and compressed size the way the platform does: anonymously.
# Pin the digest it prints, as `--policy-image=<repo>@<digest>`. The platform counts the size it
# prints against its 30 GB budget.
#
# Usage
#   docker/read_image_digest.sh <you>/<image>:<tag>          # a Docker Hub repository
#
# Needs curl and jq. The platform sees the same 401 or 404: the image is not public, or the name is
# wrong.
set -euo pipefail

ref="${1:?usage: $0 <you>/<image>:<tag>}"
repo="${ref%%:*}"
tag="${ref##*:}"
if [ "$repo" = "$ref" ]; then
  tag=latest
fi

token=$(curl -fsS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${repo}:pull" | jq -r .token)
headers=$(mktemp)
manifest=$(mktemp)
trap 'rm -f "$headers" "$manifest"' EXIT

status=$(curl -sS -o "$manifest" -D "$headers" -w '%{http_code}' \
  -H "Authorization: Bearer ${token}" \
  -H 'Accept: application/vnd.oci.image.manifest.v1+json' \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  "https://registry-1.docker.io/v2/${repo}/manifests/${tag}")
if [ "$status" != 200 ]; then
  echo "HTTP ${status} for ${ref}: the image is not public, or the name is wrong" >&2
  exit 1
fi

digest=$(grep -i '^docker-content-digest:' "$headers" | tr -d '\r' | awk '{print $2}')
bytes=$(jq '([.layers[].size] | add) + .config.size' "$manifest")
echo "digest            ${digest}"
echo "compressed_bytes  ${bytes}"
echo "compressed_gb     $(awk -v b="$bytes" 'BEGIN {printf "%.2f", b / 1e9}')"
