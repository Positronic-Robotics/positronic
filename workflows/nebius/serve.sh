#!/usr/bin/env bash
# Submit a Nebius Serverless Endpoint running a vendor inference server.
#
# After creation, polls until the endpoint reports an address and prints connection
# details. The container itself takes ~10-15 min more to finish uv sync and
# load the model into GPU memory after the address appears.
#
# Hardcoded: GPU platform, MysteryBox secret names, S3 endpoint URL,
# container port. Vendor selects image + uv extra. Override-able via env:
# NEBIUS_PARENT_ID, NEBIUS_SUBNET_ID, NEBIUS_PUBLIC.

set -euo pipefail

PARENT_ID="${NEBIUS_PARENT_ID:-project-e00f38wexevrr52b8j}"
SUBNET_ID="${NEBIUS_SUBNET_ID:-vpcsubnet-e00pk1j1x6hjmr4m92}"
# Shared filesystem (RWX) holding the uv / HF / openpi caches across cold starts.
# pos3's cache stays on local disk (~/.cache/positronic/s3) — never redirected here.
CACHE_FS="${NEBIUS_CACHE_FS:-computefilesystem-e00f6jyfr5wkawyrab}"
# Docker image tag pulled by the endpoint. `make push-*` only updates `:latest`
# under CI; locally it pushes `:<branch>`/`:<sha>`. To serve a branch build:
# `make push-<x> IMAGE_TAG=<branch>` then run with `NEBIUS_IMAGE_TAG=<branch>`.
IMAGE_TAG="${NEBIUS_IMAGE_TAG:-latest}"
# Nebius GPU preset. Default is one H100; multi-GPU presets must match the server's GPU count
# (DreamZero's --num_gpus runs torchrun --nproc_per_node, so an 8-GPU server needs an 8-GPU preset).
PRESET="${NEBIUS_PRESET:-1gpu-16vcpu-200gb}"

if [ $# -lt 2 ]; then
  cat >&2 <<'EOF'
Usage: bash workflows/nebius/serve.sh <vendor> <endpoint-name> [server args...]

Vendors: lerobot_0_3_3 | lerobot | openpi | gr00t | dreamzero | molmoact2

The endpoint name must be unique in the project (lowercase alphanumeric + dashes).
Remaining arguments forward to positronic.vendors.<vendor>.server.

Examples:

  # ACT public demo checkpoint (no S3 credentials needed inside the container)
  bash workflows/nebius/serve.sh lerobot_0_3_3 my-act-demo demo

  # Your own ACT checkpoint
  bash workflows/nebius/serve.sh lerobot_0_3_3 act-server ee \
    --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/lerobot/<exp_name>/

  # SmolVLA / lerobot 0.4.x checkpoint
  bash workflows/nebius/serve.sh lerobot smolvla-server ee \
    --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/smolvla/<exp_name>/

  # OpenPI (ee_frame is the EE frame the checkpoint speaks; None means the rig's default)
  bash workflows/nebius/serve.sh openpi pi-server ee \
    --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/openpi/<exp_name>/ \
    --pipeline.ee_frame=None

  # GR00T
  bash workflows/nebius/serve.sh gr00t groot-server ee_rot6d_rel \
    --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/groot/<exp_name>/
EOF
  exit 1
fi

VENDOR="$1"
NAME="$2"
shift 2

case "$VENDOR" in
  lerobot_0_3_3) IMAGE="positro/positronic:${IMAGE_TAG}"; EXTRA="--extra lerobot_0_3_3 " ;;
  lerobot)       IMAGE="positro/positronic:${IMAGE_TAG}"; EXTRA="--extra lerobot " ;;
  # openpi.server imports `openpi_client` at module top → needs --extra openpi
  openpi)        IMAGE="positro/openpi:${IMAGE_TAG}";     EXTRA="--extra openpi " ;;
  gr00t)         IMAGE="positro/gr00t:${IMAGE_TAG}";      EXTRA="" ;;
  # dreamzero.server imports `huggingface_hub` at module top → needs --extra dreamzero
  dreamzero)     IMAGE="positro/dreamzero:${IMAGE_TAG}";  EXTRA="--extra dreamzero " ;;
  molmoact2)     IMAGE="positro/positronic:${IMAGE_TAG}"; EXTRA="--extra molmoact2 " ;;
  *)
    echo "Unknown vendor: '$VENDOR'. Supported: lerobot_0_3_3 | lerobot | openpi | gr00t | dreamzero | molmoact2" >&2
    exit 1
    ;;
esac

# Serverless endpoints have no native idle/scale-to-zero, so opt the server into
# self-shutdown (the base default is no timeout). Override the window with
# NEBIUS_IDLE_TIMEOUT_MIN; skip injection if the caller already passed one.
case " $* " in
  *" --idle_timeout_min="*|*" --idle_timeout_min "*) ;;
  *) set -- "$@" "--idle_timeout_min=${NEBIUS_IDLE_TIMEOUT_MIN:-20}" ;;
esac

SERVER_ARGS="run --python 3.13 ${EXTRA}python -m positronic.vendors.${VENDOR}.server $*"

# A public IP is only needed by a consumer outside the cloud. The project's public-IP quota is small and
# permanently occupied by the long-lived instances, and a --public create against a full quota waits for an
# allocation that never comes — the endpoint sits in PROVISIONING with no error. An in-cloud consumer on the
# same subnet reaches the endpoint on its private IP, so set NEBIUS_PUBLIC=0 there.
PUBLIC_FLAG=(--public)
[ "${NEBIUS_PUBLIC:-1}" = "0" ] && PUBLIC_FLAG=()

echo "Creating $VENDOR endpoint '$NAME'..."
nebius ai endpoint create \
  --parent-id "$PARENT_ID" \
  --subnet-id "$SUBNET_ID" \
  --name "$NAME" \
  --image "$IMAGE" \
  --container-command uv \
  --args "$SERVER_ARGS" \
  --container-port 8000 \
  --platform gpu-h100-sxm \
  --preset "$PRESET" \
  --working-dir /positronic \
  --volume "${CACHE_FS}:/cache:rw" \
  --env UV_CACHE_DIR=/cache/uv \
  --env HF_HOME=/cache/hf \
  --env OPENPI_DATA_HOME=/cache/openpi \
  --env-secret AWS_ACCESS_KEY_ID=positronic-serverless-aws-access-key-id \
  --env-secret AWS_SECRET_ACCESS_KEY=positronic-serverless-aws-secret-access-key \
  --env AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud:443 \
  --env AWS_DEFAULT_REGION=eu-north1 \
  "${PUBLIC_FLAG[@]}" >/dev/null

ID=$(nebius ai endpoint list --parent-id "$PARENT_ID" --format json \
  | jq -r --arg n "$NAME" '.items[]? | select(.metadata.name==$n) | .metadata.id')

if [ -z "$ID" ]; then
  echo "Endpoint create did not return a known resource for name '$NAME'." >&2
  exit 1
fi

echo "Endpoint ID: $ID"

# The two fields carry different shapes: a private address is bare `host:port`, a public one is a full
# tunnel URL with its own scheme. Prefixing a scheme unconditionally corrupts the public form.
if [ "${NEBIUS_PUBLIC:-1}" = "0" ]; then
  FIELD=private_endpoints
  SCHEME=http://
else
  FIELD=public_endpoints
  SCHEME=
fi

echo "Waiting for the $FIELD address (typically <1 min)..."
ADDRESS=""
for i in $(seq 1 30); do
  ADDRESS=$(nebius ai endpoint get "$ID" --format json 2>/dev/null \
    | jq -r --arg f "$FIELD" '.status[$f][0]? // empty')
  if [ -n "$ADDRESS" ]; then break; fi
  sleep 10
done

if [ -z "$ADDRESS" ]; then
  echo "No $FIELD address within 5 min. Check: nebius ai endpoint get $ID" >&2
  exit 1
fi

URL="$SCHEME$ADDRESS"

cat <<BANNER

==============================================================
  Endpoint URL:  $URL
  Endpoint ID:   $ID
  Endpoint name: $NAME
  Vendor:        $VENDOR
==============================================================

The container is still warming up (image pull + uv sync + checkpoint load,
~10-15 min total). Follow startup logs:

  nebius ai endpoint logs $ID --follow

Once the model is loaded, sanity-check with:

  curl $URL/api/v1/models

To release the endpoint and its address:

  bash workflows/nebius/stop.sh $NAME

BANNER
