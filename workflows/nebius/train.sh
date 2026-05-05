#!/usr/bin/env bash
# Submit a vendor training run as a Nebius Serverless Job.
#
# The bucket referenced by --input_path=s3://... is mounted via Mountpoint-S3
# (FUSE) at /mnt/input, and --input_path is rewritten to a path under that mount.
# This skips the dataset download into local cache and streams reads on demand.
#
# --output_dir stays as an s3:// URL handled by pos3 — vendor checkpoint savers
# tend to use symlinks, which Mountpoint-S3 does not support.
#
# Hardcoded: GPU platform/preset, MysteryBox secret names, S3 endpoint URL.
# Vendor selects image + uv extra. Override-able via env: NEBIUS_PARENT_ID,
# NEBIUS_SUBNET_ID.

set -euo pipefail

PARENT_ID="${NEBIUS_PARENT_ID:-project-e00f38wexevrr52b8j}"
SUBNET_ID="${NEBIUS_SUBNET_ID:-vpcsubnet-e00pk1j1x6hjmr4m92}"

if [ $# -lt 1 ]; then
  cat >&2 <<'EOF'
Usage: bash workflows/nebius/train.sh <vendor> [train args...]

Vendors: lerobot_0_3_3 | lerobot | openpi | gr00t

Forwards remaining arguments to positronic.vendors.<vendor>.train. Example:

  bash workflows/nebius/train.sh lerobot_0_3_3 \
    --input_path=s3://<your-bucket>/sim_stack_cubes_lerobot/ \
    --exp_name=act_sim_stack_v1 \
    --output_dir=s3://<your-bucket>/checkpoints/lerobot/ \
    --num_train_steps=50000 --save_freq=10000
EOF
  exit 1
fi

VENDOR="$1"
shift

case "$VENDOR" in
  lerobot_0_3_3) IMAGE="positro/positronic:latest"; EXTRA="--extra lerobot_0_3_3 " ;;
  lerobot)       IMAGE="positro/positronic:latest"; EXTRA="--extra lerobot " ;;
  openpi)        IMAGE="positro/openpi:latest";     EXTRA="" ;;
  gr00t)         IMAGE="positro/gr00t:latest";      EXTRA="" ;;
  *)
    echo "Unknown vendor: '$VENDOR'. Supported: lerobot_0_3_3 | lerobot | openpi | gr00t" >&2
    exit 1
    ;;
esac

# Rewrite --input_path=s3://bucket/key/ → /mnt/input/key/, plan an S3 mount.
INPUT_BUCKET=""
NEW_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --input_path=s3://*)
      val="${arg#--input_path=}"
      INPUT_BUCKET="${val#s3://}"
      INPUT_BUCKET="${INPUT_BUCKET%%/*}"
      key="${val#s3://${INPUT_BUCKET}}"
      key="${key#/}"
      NEW_ARGS+=("--input_path=/mnt/input/${key}")
      ;;
    *)
      NEW_ARGS+=("$arg")
      ;;
  esac
done

VOLUME_FLAGS=()
if [ -n "$INPUT_BUCKET" ]; then
  VOLUME_FLAGS+=(--volume "s3://${INPUT_BUCKET}:/mnt/input:ro:default@positronic-serverless-s3-creds")
fi

JOB_NAME="${VENDOR//_/-}-train-$(date +%Y%m%d-%H%M%S)"
TRAIN_ARGS="run --python 3.11 ${EXTRA}python -m positronic.vendors.${VENDOR}.train ${NEW_ARGS[*]}"

nebius ai job create \
  --parent-id "$PARENT_ID" \
  --subnet-id "$SUBNET_ID" \
  --name "$JOB_NAME" \
  --image "$IMAGE" \
  --container-command uv \
  --args "$TRAIN_ARGS" \
  --platform gpu-h100-sxm \
  --preset 1gpu-16vcpu-200gb \
  --timeout 24h \
  --working-dir /positronic \
  ${VOLUME_FLAGS[@]+"${VOLUME_FLAGS[@]}"} \
  --env-secret AWS_ACCESS_KEY_ID=positronic-serverless-aws-access-key-id \
  --env-secret AWS_SECRET_ACCESS_KEY=positronic-serverless-aws-secret-access-key \
  --env-secret WANDB_API_KEY=positronic-serverless-wandb-api-key \
  --env AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud:443 \
  --env AWS_DEFAULT_REGION=eu-north1
