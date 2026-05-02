#!/usr/bin/env bash
# Convert a Positronic dataset into LeRobot 0.3.3 format as a Nebius Serverless Job.
#
# Hardcoded: image, CPU platform/preset, MysteryBox secret names, S3 endpoint URL.
# Override-able via env: NEBIUS_PARENT_ID, NEBIUS_SUBNET_ID.
#
# All flags after the script name are forwarded to:
#     python -m positronic.vendors.lerobot_0_3_3.to_lerobot convert

set -euo pipefail

PARENT_ID="${NEBIUS_PARENT_ID:-project-e00f38wexevrr52b8j}"
SUBNET_ID="${NEBIUS_SUBNET_ID:-vpcsubnet-e00pk1j1x6hjmr4m92}"

if [ $# -eq 0 ]; then
  cat >&2 <<'EOF'
Usage: bash workflows/nebius/convert.sh [convert args...]

Forwards all arguments to positronic.vendors.lerobot_0_3_3.to_lerobot convert. Example:

  bash workflows/nebius/convert.sh \
    --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
    --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
    --output_dir=s3://<your-bucket>/sim_stack_cubes_lerobot/
EOF
  exit 1
fi

JOB_NAME="convert-$(date +%Y%m%d-%H%M%S)"
CONVERT_ARGS="run --python 3.11 --extra lerobot_0_3_3 python -m positronic.vendors.lerobot_0_3_3.to_lerobot convert $*"

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
