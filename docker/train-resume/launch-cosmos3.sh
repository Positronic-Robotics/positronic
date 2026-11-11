#!/bin/bash
# Start or resume a Cosmos3 SFT run from /data/run/config.toml. The run resumes from the last checkpoint under
# /data/output, because the TOML's `job.name` stays the same at each launch.
#
# Usage (in the training image; /opt/train/resume calls it)
#   /opt/train/launch
#
# Environment: DROID_ROOT (the dataset directory, required), NPROC_PER_NODE (default: every GPU),
# EXTRA_TAIL_OVERRIDES (space-separated Hydra overrides). The image sets BASE_CHECKPOINT_PATH and WAN_VAE_PATH.
set -euo pipefail
: "${DROID_ROOT:?run.env sets no DROID_ROOT}"
export DROID_ROOT
cd /workspace
TAIL=()
if [[ -n "${EXTRA_TAIL_OVERRIDES:-}" ]]; then
    read -r -a overrides <<< "$EXTRA_TAIL_OVERRIDES"
    TAIL=(-- "${overrides[@]}")
fi
IMAGINAIRE_OUTPUT_ROOT=/data/output PYTHONPATH=. torchrun \
    --nproc_per_node="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}" --master_port="${MASTER_PORT:-50012}" \
    -m cosmos_framework.scripts.train --sft-toml=/data/run/config.toml "${TAIL[@]}" \
    2>&1 | tee -a "/data/run/logs/train-$(date -u +%Y%m%dT%H%M%SZ).log"
exit "${PIPESTATUS[0]}"
