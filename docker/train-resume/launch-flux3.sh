#!/bin/bash
# Start or resume a FLUX 3 Action run from /data/run/train.json. `resume=auto` restores the last complete checkpoint
# under the config's `output_dir`.
#
# Usage (in the training image; /opt/train/resume calls it)
#   /opt/train/launch
#
# Environment: NPROC_PER_NODE (default: every GPU).
set -euo pipefail
cd /opt/flux-action
.venv/bin/torchrun --nproc_per_node="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}" \
    -m flux_action.cli train --config /data/run/train.json --override resume=auto \
    2>&1 | tee -a "/data/run/logs/train-$(date -u +%Y%m%dT%H%M%SZ).log"
exit "${PIPESTATUS[0]}"
