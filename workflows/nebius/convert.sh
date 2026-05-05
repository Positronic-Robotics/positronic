#!/usr/bin/env bash
# Convert a Positronic dataset into a vendor's LeRobot dataset format
# as a Nebius Serverless Job.
#
# Hardcoded: CPU platform/preset, MysteryBox secret names, S3 endpoint URL.
# Vendor selects image + uv extra. Override-able via env: NEBIUS_PARENT_ID,
# NEBIUS_SUBNET_ID.
#
# OpenPI also requires normalization stats. When vendor=openpi, this script
# blocks until the convert job completes, then submits a second job (in the
# `positro/openpi` image) that runs `compute_norm_stats.py` and writes assets
# to <output_dir>/stats/. Use `--stats_path=<output_dir>/stats/` when training.

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

# OpenPI needs the dataset path so we can chain a stats job after convert finishes.
OUTPUT_DIR=""
for arg in "$@"; do
  case "$arg" in
    --output_dir=*) OUTPUT_DIR="${arg#--output_dir=}" ;;
  esac
done

if [ "$VENDOR" = "openpi" ] && [ -z "$OUTPUT_DIR" ]; then
  echo "openpi convert requires --output_dir=s3://... so a stats job can be chained." >&2
  exit 1
fi

JOB_NAME="${VENDOR//_/-}-convert-$(date +%Y%m%d-%H%M%S)"
CONVERT_ARGS="run --python 3.11 ${EXTRA}python -m ${CONVERTER_MODULE} convert $*"

CREATE_OUT=$(nebius ai job create \
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
  --env AWS_DEFAULT_REGION=eu-north1)

echo "$CREATE_OUT"

# For non-openpi vendors we're done — the create call already streamed the job ID
# and follow-up commands.
if [ "$VENDOR" != "openpi" ]; then
  exit 0
fi

CONVERT_ID=$(echo "$CREATE_OUT" | grep -oE 'aijob-[a-z0-9]+' | head -1)
if [ -z "$CONVERT_ID" ]; then
  echo "Could not parse convert job id from create output." >&2
  exit 1
fi

echo
echo "Waiting for convert job $CONVERT_ID to complete before submitting openpi stats..."
while true; do
  STATE=$(nebius ai job get "$CONVERT_ID" --format json 2>/dev/null | jq -r '.status.state // ""')
  case "$STATE" in
    COMPLETED)
      echo "Convert COMPLETED."
      break
      ;;
    FAILED|CANCELLED)
      echo "Convert finished with state $STATE — not submitting stats job." >&2
      exit 1
      ;;
    "")
      echo "Could not read job state; retrying..."
      ;;
    *)
      printf '\rConvert state: %-20s' "$STATE"
      ;;
  esac
  sleep 30
done

# Stats must be a SIBLING of the dataset, not a child — pos3 refuses when an
# upload destination is inside the dataset it's also downloading.
DATASET_TRIM="${OUTPUT_DIR%/}"
STATS_PATH="${DATASET_TRIM%/*}/stats/"
STATS_JOB_NAME="openpi-stats-$(date +%Y%m%d-%H%M%S)"
STATS_ARGS="run --python 3.11 python -m positronic.vendors.openpi.stats --input_path=${OUTPUT_DIR} --output_path=${STATS_PATH}"

echo
echo "Submitting openpi stats job (image positro/openpi)..."
nebius ai job create \
  --parent-id "$PARENT_ID" \
  --subnet-id "$SUBNET_ID" \
  --name "$STATS_JOB_NAME" \
  --image positro/openpi:latest \
  --container-command uv \
  --args "$STATS_ARGS" \
  --platform cpu-e2 \
  --preset 8vcpu-32gb \
  --timeout 4h \
  --working-dir /positronic \
  --env-secret AWS_ACCESS_KEY_ID=positronic-serverless-aws-access-key-id \
  --env-secret AWS_SECRET_ACCESS_KEY=positronic-serverless-aws-secret-access-key \
  --env AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud:443 \
  --env AWS_DEFAULT_REGION=eu-north1

cat <<EOM

Stats output will land at: ${STATS_PATH}
When training, pass:
  --input_path=${OUTPUT_DIR} --stats_path=${STATS_PATH}
EOM
