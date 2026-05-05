#!/usr/bin/env bash
# Convert a Positronic dataset into a vendor's LeRobot dataset format
# as a Nebius Serverless Job.
#
# Hardcoded: CPU platform/preset, MysteryBox secret names, S3 endpoint URL.
# Vendor selects image + uv extra. Override-able via env: NEBIUS_PARENT_ID,
# NEBIUS_SUBNET_ID.

set -euo pipefail

PARENT_ID="${NEBIUS_PARENT_ID:-project-e00f38wexevrr52b8j}"
SUBNET_ID="${NEBIUS_SUBNET_ID:-vpcsubnet-e00pk1j1x6hjmr4m92}"

if [ $# -lt 1 ]; then
  cat >&2 <<'EOF'
Usage: bash workflows/nebius/convert.sh <vendor> [convert args...]

Vendors: lerobot_0_3_3 | lerobot

Forwards remaining arguments to positronic.vendors.<vendor>.to_lerobot convert.
Example:

  bash workflows/nebius/convert.sh lerobot_0_3_3 \
    --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
    --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
    --output_dir=s3://<your-bucket>/sim_stack_cubes_lerobot/

OpenPI and GR00T do not have a converter — they consume LeRobot 0.3.3 datasets
produced by `lerobot_0_3_3`.
EOF
  exit 1
fi

VENDOR="$1"
shift

case "$VENDOR" in
  lerobot_0_3_3) EXTRA="--extra lerobot_0_3_3 " ;;
  lerobot)       EXTRA="--extra lerobot " ;;
  *)
    echo "Unknown vendor: '$VENDOR'. Supported: lerobot_0_3_3 | lerobot" >&2
    exit 1
    ;;
esac

JOB_NAME="${VENDOR//_/-}-convert-$(date +%Y%m%d-%H%M%S)"
CONVERT_ARGS="run --python 3.11 ${EXTRA}python -m positronic.vendors.${VENDOR}.to_lerobot convert $*"

nebius ai job create \
  --parent-id "$PARENT_ID" \
  --subnet-id "$SUBNET_ID" \
  --name "$JOB_NAME" \
  --image positro/positronic:latest \
  --container-command uv \
  --args "$CONVERT_ARGS" \
  --platform cpu-e2 \
  --preset 8vcpu-32gb \
  --timeout 4h \
  --working-dir /positronic \
  --env-secret AWS_ACCESS_KEY_ID=positronic-serverless-aws-access-key-id \
  --env-secret AWS_SECRET_ACCESS_KEY=positronic-serverless-aws-secret-access-key \
  --env AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud:443 \
  --env AWS_DEFAULT_REGION=eu-north1
