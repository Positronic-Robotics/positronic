#!/usr/bin/env bash
# Submit a Nebius Serverless Endpoint running a vendor inference server.
#
# After creation, polls until a public IP is allocated and prints connection
# details. The container itself takes ~10-15 min more to finish uv sync and
# load the model into GPU memory after the IP appears.
#
# Hardcoded: GPU platform/preset, MysteryBox secret names, S3 endpoint URL,
# container port. Vendor selects image + uv extra. Override-able via env:
# NEBIUS_PARENT_ID, NEBIUS_SUBNET_ID.

set -euo pipefail

PARENT_ID="${NEBIUS_PARENT_ID:-project-e00f38wexevrr52b8j}"
SUBNET_ID="${NEBIUS_SUBNET_ID:-vpcsubnet-e00pk1j1x6hjmr4m92}"
# Shared filesystem (RWX) holding the uv / HF / openpi caches across cold starts.
# pos3's cache stays on local disk (~/.cache/positronic/s3) — never redirected here.
CACHE_FS="${NEBIUS_CACHE_FS:-computefilesystem-e00f6jyfr5wkawyrab}"

if [ $# -lt 2 ]; then
  cat >&2 <<'EOF'
Usage: bash workflows/nebius/serve.sh <vendor> <endpoint-name> [server args...]

Vendors: lerobot_0_3_3 | lerobot | openpi | gr00t

The endpoint name must be unique in the project (lowercase alphanumeric + dashes).
Remaining arguments forward to positronic.vendors.<vendor>.server.

Examples:

  # ACT public demo checkpoint (no S3 credentials needed inside the container)
  bash workflows/nebius/serve.sh lerobot_0_3_3 my-act-demo demo

  # Your own ACT checkpoint
  bash workflows/nebius/serve.sh lerobot_0_3_3 act-server serve \
    --checkpoints_dir=s3://<your-bucket>/checkpoints/lerobot/<exp_name>/

  # SmolVLA / lerobot 0.4.x checkpoint
  bash workflows/nebius/serve.sh lerobot smolvla-server serve \
    --checkpoints_dir=s3://<your-bucket>/checkpoints/smolvla/<exp_name>/

  # OpenPI
  bash workflows/nebius/serve.sh openpi pi-server serve \
    --checkpoints_dir=s3://<your-bucket>/checkpoints/openpi/<exp_name>/

  # GR00T
  bash workflows/nebius/serve.sh gr00t groot-server ee_rot6d_rel \
    --checkpoints_dir=s3://<your-bucket>/checkpoints/groot/<exp_name>/
EOF
  exit 1
fi

VENDOR="$1"
NAME="$2"
shift 2

case "$VENDOR" in
  lerobot_0_3_3) IMAGE="positro/positronic:latest"; EXTRA="--extra lerobot_0_3_3 " ;;
  lerobot)       IMAGE="positro/positronic:latest"; EXTRA="--extra lerobot " ;;
  # openpi.server imports `openpi_client` at module top → needs --extra openpi
  openpi)        IMAGE="positro/openpi:latest";     EXTRA="--extra openpi " ;;
  gr00t)         IMAGE="positro/gr00t:latest";      EXTRA="" ;;
  *)
    echo "Unknown vendor: '$VENDOR'. Supported: lerobot_0_3_3 | lerobot | openpi | gr00t" >&2
    exit 1
    ;;
esac

SERVER_ARGS="run --python 3.11 ${EXTRA}python -m positronic.vendors.${VENDOR}.server $*"

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
  --preset 1gpu-16vcpu-200gb \
  --working-dir /positronic \
  --volume "${CACHE_FS}:/cache:rw" \
  --env UV_CACHE_DIR=/cache/uv \
  --env HF_HOME=/cache/hf \
  --env OPENPI_DATA_HOME=/cache/openpi \
  --env-secret AWS_ACCESS_KEY_ID=positronic-serverless-aws-access-key-id \
  --env-secret AWS_SECRET_ACCESS_KEY=positronic-serverless-aws-secret-access-key \
  --env AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud:443 \
  --env AWS_DEFAULT_REGION=eu-north1 \
  --public >/dev/null

ID=$(nebius ai endpoint list --parent-id "$PARENT_ID" --format json \
  | jq -r --arg n "$NAME" '.items[]? | select(.metadata.name==$n) | .metadata.id')

if [ -z "$ID" ]; then
  echo "Endpoint create did not return a known resource for name '$NAME'." >&2
  exit 1
fi

echo "Endpoint ID: $ID"
echo "Waiting for public IP (typically <1 min)..."

IP=""
for i in $(seq 1 30); do
  IP=$(nebius ai endpoint get "$ID" --format json 2>/dev/null \
    | jq -r '.status.public_endpoints[0]? // empty')
  if [ -n "$IP" ]; then break; fi
  sleep 10
done

if [ -z "$IP" ]; then
  echo "Public IP not allocated within 5 min. Check: nebius ai endpoint get $ID" >&2
  exit 1
fi

cat <<BANNER

==============================================================
  Endpoint URL:  http://$IP
  Endpoint ID:   $ID
  Endpoint name: $NAME
  Vendor:        $VENDOR
==============================================================

The container is still warming up (image pull + uv sync + checkpoint load,
~10-15 min total). Follow startup logs:

  nebius ai endpoint logs $ID --follow

Once the model is loaded, sanity-check with:

  curl http://$IP/api/v1/models

To release the endpoint and its public IP:

  bash workflows/nebius/stop.sh $NAME

BANNER
