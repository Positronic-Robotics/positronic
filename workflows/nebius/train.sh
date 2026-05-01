#!/usr/bin/env bash
# Submit ACT (lerobot 0.3.3) training as a Nebius Serverless Job.
#
# Hardcoded: image, GPU platform/preset, MysteryBox secret names, S3 endpoint URL.
# Override-able via env: NEBIUS_PARENT_ID, NEBIUS_SUBNET_ID.
#
# All flags after the script name are forwarded to:
#     python -m positronic.vendors.lerobot_0_3_3.train

set -euo pipefail

PARENT_ID="${NEBIUS_PARENT_ID:-project-e00f38wexevrr52b8j}"
SUBNET_ID="${NEBIUS_SUBNET_ID:-vpcsubnet-e00pk1j1x6hjmr4m92}"

if [ $# -eq 0 ]; then
  cat >&2 <<'EOF'
Usage: bash workflows/nebius/train.sh [train args...]

Forwards all arguments to positronic.vendors.lerobot_0_3_3.train. Example:

  bash workflows/nebius/train.sh \
    --input_path=s3://interim/sim_stack/lerobot/ee/ \
    --exp_name=act_sim_stack_$(date +%Y%m%d_%H%M%S) \
    --output_dir=s3://checkpoints/sim_stack/lerobot/serverless/ \
    --num_train_steps=200 --save_freq=100
EOF
  exit 1
fi

JOB_NAME="act-train-$(date +%Y%m%d-%H%M%S)"
TRAIN_ARGS="run --python 3.11 --extra lerobot_0_3_3 python -m positronic.vendors.lerobot_0_3_3.train $*"

nebius ai job create \
  --parent-id "$PARENT_ID" \
  --subnet-id "$SUBNET_ID" \
  --name "$JOB_NAME" \
  --image positro/positronic:latest \
  --container-command uv \
  --args "$TRAIN_ARGS" \
  --platform gpu-h100-sxm \
  --preset 1gpu-16vcpu-200gb \
  --timeout 24h \
  --working-dir /positronic \
  --env-secret AWS_ACCESS_KEY_ID=positronic-serverless-aws-access-key-id \
  --env-secret AWS_SECRET_ACCESS_KEY=positronic-serverless-aws-secret-access-key \
  --env-secret WANDB_API_KEY=positronic-serverless-wandb-api-key \
  --env AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud:443 \
  --env AWS_DEFAULT_REGION=eu-north1
