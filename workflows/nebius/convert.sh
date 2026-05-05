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

Vendors: lerobot_0_3_3 | lerobot | openpi | gr00t

Forwards remaining arguments to the converter recommended by each vendor's
README (lerobot 0.3.3 for ACT/OpenPI/GR00T; lerobot 0.4.x for SmolVLA). The
caller picks a vendor-specific codec via `--dataset.codec=...`.

Examples:

  bash workflows/nebius/convert.sh lerobot_0_3_3 \
    --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
    --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
    --output_dir=s3://<your-bucket>/sim_stack_cubes_lerobot/

  bash workflows/nebius/convert.sh openpi \
    --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
    --dataset.codec=@positronic.vendors.openpi.codecs.ee \
    --output_dir=s3://<your-bucket>/sim_stack_cubes_openpi/

  bash workflows/nebius/convert.sh gr00t \
    --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
    --dataset.codec=@positronic.vendors.gr00t.codecs.ee_rot6d_joints \
    --output_dir=s3://<your-bucket>/sim_stack_cubes_gr00t/
EOF
  exit 1
fi

VENDOR="$1"
shift

# OpenPI and GR00T don't ship their own converter — they re-use the lerobot_0_3_3
# converter with their own codec namespaces (per each vendor's README).
case "$VENDOR" in
  lerobot_0_3_3|openpi|gr00t) CONVERTER_MODULE="positronic.vendors.lerobot_0_3_3.to_lerobot"; EXTRA="--extra lerobot_0_3_3 " ;;
  lerobot)                    CONVERTER_MODULE="positronic.vendors.lerobot.to_lerobot";       EXTRA="--extra lerobot " ;;
  *)
    echo "Unknown vendor: '$VENDOR'. Supported: lerobot_0_3_3 | lerobot | openpi | gr00t" >&2
    exit 1
    ;;
esac

JOB_NAME="${VENDOR//_/-}-convert-$(date +%Y%m%d-%H%M%S)"
CONVERT_ARGS="run --python 3.11 ${EXTRA}python -m ${CONVERTER_MODULE} convert $*"

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
