#!/usr/bin/env bash
# Read a public image's manifest digest and compressed size the way the platform does: anonymously.
# Pin the digest it prints, as `--policy-image=<repo>@<digest>`. The platform counts the size it
# prints against its 50 GB budget. When the tag names an index, the size is its linux/amd64 image's.
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

read_manifest() {  # <tag or digest>: the manifest into $manifest, its headers into $headers
  local status
  status=$(curl -sS -o "$manifest" -D "$headers" -w '%{http_code}' \
    -H "Authorization: Bearer ${token}" \
    -H 'Accept: application/vnd.oci.image.index.v1+json' \
    -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
    -H 'Accept: application/vnd.oci.image.manifest.v1+json' \
    -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
    "https://registry-1.docker.io/v2/${repo}/manifests/$1")
  if [ "$status" != 200 ]; then
    echo "HTTP ${status} for ${repo}:$1: the image is not public, or the name is wrong" >&2
    exit 1
  fi
}

read_manifest "$tag"
digest=$(grep -i '^docker-content-digest:' "$headers" | tr -d '\r' | awk '{print $2}')
if jq -e '.manifests' "$manifest" > /dev/null; then
  amd64=$(jq -r '[.manifests[] | select(.platform.os == "linux" and .platform.architecture == "amd64")][0].digest // empty' "$manifest")
  if [ -z "$amd64" ]; then
    echo "${ref} names no linux/amd64 image" >&2
    exit 1
  fi
  read_manifest "$amd64"
fi
bytes=$(jq '([.layers[].size] | add) + .config.size' "$manifest")
echo "digest            ${digest}"
echo "compressed_bytes  ${bytes}"
echo "compressed_gb     $(awk -v b="$bytes" 'BEGIN {printf "%.2f", b / 1e9}')"
