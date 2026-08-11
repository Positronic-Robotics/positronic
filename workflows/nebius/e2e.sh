#!/usr/bin/env bash
# Full-pipeline validation harness for one vendor:
#   convert (+ openpi stats) → train (200 steps) → serve → served-endpoint contract → teardown
#
# Submits jobs via the user-facing primitives (convert.sh / train.sh / serve.sh / stop.sh),
# polls Nebius for completion, and reports a status line per stage. Intended for CI or for
# verifying a vendor's pipeline still works after an image / dependency / script change.
#
# Cost: ~$2-5 per run (cold-start dominated; the actual compute is short).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$SCRIPT_DIR/common.sh"

S3_BASE="${E2E_S3_BASE:-s3://tmp/e2e_validation}"
EXP_NAME="${E2E_EXP_NAME:-e2e_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${E2E_LOG_ROOT:-/tmp/e2e_logs}"
DATASET="${E2E_DATASET:-@positronic.cfg.ds.sim.sim_stack_cubes}"

VENDOR="${1:-}"
if [ -z "$VENDOR" ]; then
  cat >&2 <<'EOF'
Usage: bash workflows/nebius/e2e.sh <vendor>

Vendors: lerobot_0_3_3 | lerobot | openpi | gr00t

Optional env:
  E2E_S3_BASE     S3 base for outputs (default: s3://tmp/e2e_validation)
  E2E_EXP_NAME    Experiment name (default: e2e_<timestamp>)
  E2E_LOG_ROOT    Local log dir (default: /tmp/e2e_logs)
  E2E_DATASET     Dataset config target (default: @positronic.cfg.ds.sim.sim_stack_cubes)

To run all four in parallel:
  for v in lerobot_0_3_3 lerobot openpi gr00t; do bash workflows/nebius/e2e.sh "$v" & done; wait

The shared cache FS (NEBIUS_CACHE_FS) is RWX, but a cold (empty) cache plus a
4-way parallel fan-out means several jobs racing to populate the same entries.
On a fresh cache, seed it first with one vendor, then fan out the rest warm:
  bash workflows/nebius/e2e.sh lerobot_0_3_3
  for v in lerobot openpi gr00t; do bash workflows/nebius/e2e.sh "$v" & done; wait
EOF
  exit 1
fi

case "$VENDOR" in
  lerobot_0_3_3) CODEC=positronic.vendors.lerobot_0_3_3.codecs.ee ;;
  lerobot)       CODEC=positronic.vendors.lerobot.codecs.ee ;;
  openpi)        CODEC=positronic.vendors.openpi.codecs.ee ;;
  gr00t)         CODEC=positronic.vendors.gr00t.codecs.ee_rot6d ;;
  *) echo "Unknown vendor '$VENDOR'. Supported: lerobot_0_3_3 | lerobot | openpi | gr00t" >&2; exit 1 ;;
esac

mkdir -p "$LOG_ROOT"
DATASET_DIR="$S3_BASE/$VENDOR/dataset/"
CKPT_DIR="$S3_BASE/$VENDOR/checkpoints/"
ENDPOINT_NAME="e2e-${VENDOR//_/-}-$(date +%H%M%S)"
LOG="$LOG_ROOT/$VENDOR.log"

note() {
  local ts; ts=$(date +%H:%M:%S)
  echo "[$ts][$VENDOR] $*" | tee -a "$LOG"
}

extract_job_id() { echo "$1" | grep -oE 'aijob-[a-z0-9]+' | head -1; }

wait_job() {
  local job_id="$1"; local stage="$2"
  while true; do
    local state
    state=$(nebius ai job get "$job_id" --format json 2>/dev/null | jq -r '.status.state // ""')
    case "$state" in
      COMPLETED)        note "$stage $job_id COMPLETED"; return 0 ;;
      FAILED|CANCELLED) note "$stage $job_id $state";    return 1 ;;
    esac
    sleep 30
  done
}

# ---- 1. Convert (convert.sh openpi additionally chains a stats job) ----
note "convert -> $DATASET_DIR"
CONVERT_OUT=$(bash "$SCRIPT_DIR/convert.sh" "$VENDOR" \
  "--dataset.dataset=${DATASET}" \
  "--dataset.codec=@${CODEC}" \
  "--output_dir=${DATASET_DIR}" 2>&1)
echo "$CONVERT_OUT" >> "$LOG"
CONVERT_ID=$(extract_job_id "$CONVERT_OUT")
[ -z "$CONVERT_ID" ] && { note "FAIL: no convert job id"; exit 1; }
note "convert job: $CONVERT_ID"
wait_job "$CONVERT_ID" "convert" || exit 1

# OpenPI: convert.sh has already submitted a stats job by this point. Find it by name.
STATS_PATH=""
if [ "$VENDOR" = "openpi" ]; then
  sleep 15
  STATS_ID=$(nebius ai job list --parent-id "$PARENT_ID" --format json | \
    jq -r '.items[]? | select(.metadata.name | startswith("openpi-stats-")) | "\(.metadata.created_at)|\(.metadata.id)"' | \
    sort | tail -1 | cut -d'|' -f2)
  [ -z "$STATS_ID" ] && { note "FAIL: stats job not found"; exit 1; }
  note "stats job: $STATS_ID"
  wait_job "$STATS_ID" "stats" || exit 1
  # Sibling of the dataset (see convert.sh — pos3 forbids upload-inside-download).
  STATS_PATH="${DATASET_DIR%/}"
  STATS_PATH="${STATS_PATH%/*}/stats/"
  # train consumes the assets subdirectory that openpi.stats writes inside STATS_PATH.
  STATS_TRAIN_INPUT="${STATS_PATH}assets/"
fi

# ---- 2. Train (200 short steps; goal is wiring, not loss) ----
note "train -> $CKPT_DIR"
case "$VENDOR" in
  lerobot_0_3_3)
    TRAIN_OUT=$(bash "$SCRIPT_DIR/train.sh" lerobot_0_3_3 \
      "--input_path=$DATASET_DIR" \
      "--exp_name=$EXP_NAME" \
      "--output_dir=$CKPT_DIR" \
      --num_train_steps=200 --save_freq=100 2>&1)
    SERVE_SUBCMD=(ee --pipeline.source.checkpoints_dir="$CKPT_DIR$EXP_NAME/")
    ;;
  lerobot)
    TRAIN_OUT=$(bash "$SCRIPT_DIR/train.sh" lerobot expert_only \
      "--input_path=$DATASET_DIR" \
      "--exp_name=$EXP_NAME" \
      "--output_dir=$CKPT_DIR" \
      --num_train_steps=200 --save_freq=100 2>&1)
    SERVE_SUBCMD=(ee --pipeline.source.checkpoints_dir="$CKPT_DIR$EXP_NAME/")
    ;;
  openpi)
    TRAIN_OUT=$(bash "$SCRIPT_DIR/train.sh" openpi \
      "--input_path=$DATASET_DIR" \
      "--stats_path=$STATS_TRAIN_INPUT" \
      "--output_path=$CKPT_DIR" \
      "--exp_name=$EXP_NAME" \
      --num_train_steps=500 2>&1)
    # openpi.train writes to <output_path>/<config_name>/<exp_name>/
    # The run trains in the rig's default frame, so the checkpoint declares no transform.
    SERVE_SUBCMD=(
      ee
      --pipeline.source.checkpoints_dir="${CKPT_DIR%/}/pi05_positronic_lowmem/$EXP_NAME/"
      --pipeline.ee_frame=None
    )
    ;;
  gr00t)
    TRAIN_OUT=$(bash "$SCRIPT_DIR/train.sh" gr00t \
      "--input_path=$DATASET_DIR" \
      "--output_path=$CKPT_DIR" \
      "--exp_name=$EXP_NAME" \
      --num_train_steps=200 --save_steps=100 \
      --modality_config=ee_rot6d 2>&1)
    SERVE_SUBCMD=(ee_rot6d --pipeline.source.checkpoints_dir="$CKPT_DIR$EXP_NAME/")
    ;;
esac
echo "$TRAIN_OUT" >> "$LOG"
TRAIN_ID=$(extract_job_id "$TRAIN_OUT")
[ -z "$TRAIN_ID" ] && { note "FAIL: no train job id"; exit 1; }
note "train job: $TRAIN_ID"
wait_job "$TRAIN_ID" "train" || exit 1

# ---- 3. Serve ----
note "serve -> $ENDPOINT_NAME"
# Armed before the endpoint can exist: from here on every exit, including a `serve.sh` that created
# one and then failed, has to release the GPU rather than leave it billing.
teardown() {
  local status=$?   # whatever the run was already exiting with; a teardown failure may worsen it, never hide it
  note "teardown -> $ENDPOINT_NAME"
  if bash "$SCRIPT_DIR/stop.sh" "$ENDPOINT_NAME" >> "$LOG" 2>&1; then
    note "DONE"
  else
    note "FAIL: teardown left $ENDPOINT_NAME alive and billing; delete it by hand"
    status=1
  fi
  exit "$status"
}
# The trap deletes by name, so the name has to be ours before it is armed: an endpoint already holding
# it belongs to another run, and `serve.sh` would refuse it while the trap deleted it anyway.
if nebius ai endpoint list --parent-id "$PARENT_ID" --format json \
  | jq -e --arg n "$ENDPOINT_NAME" '.items[]? | select(.metadata.name==$n)' >/dev/null 2>&1; then
  note "FAIL: an endpoint named $ENDPOINT_NAME already exists; it is not this run's to delete"
  exit 1
fi
trap teardown EXIT
SERVE_OUT=$(bash "$SCRIPT_DIR/serve.sh" "$VENDOR" "$ENDPOINT_NAME" "${SERVE_SUBCMD[@]}" 2>&1)
echo "$SERVE_OUT" >> "$LOG"
SERVE_URL=$(echo "$SERVE_OUT" | awk '/Endpoint URL:/ {print $3; exit}')
[ -z "$SERVE_URL" ] && { note "FAIL: no serve URL"; exit 1; }
note "serve URL: $SERVE_URL"

# ---- 4. Wait out the warm-up (up to 25 min), then check the served-endpoint contract ----
SECRET_ID=$(nebius mysterybox secret list --parent-id "$PARENT_ID" --format json \
  | jq -r --arg n "$AUTH_TOKEN_SECRET" '.items[]? | select(.metadata.name==$n) | .metadata.id')
[ -z "$SECRET_ID" ] && { note "FAIL: no MysteryBox secret named $AUTH_TOKEN_SECRET"; exit 1; }
AUTH_TOKEN=$(nebius mysterybox payload get-by-key --secret-id "$SECRET_ID" --key "$AUTH_TOKEN_KEY" --format json \
  | jq -r '.data.string_value')
case "$AUTH_TOKEN" in
  ''|null) note "FAIL: no $AUTH_TOKEN_KEY payload in $AUTH_TOKEN_SECRET"; exit 1 ;;
esac

# Only a 200 means the model is loaded and serving. A warming endpoint answers 502/503 and a bad token
# answers 401, both with a body of their own, so "got bytes back" would call either one warm.
RESP=""
CODE=""
for i in $(seq 1 50); do
  OUT=$(curl --max-time 5 -s -w '\n%{http_code}' -H "Authorization: Bearer $AUTH_TOKEN" "$SERVE_URL/api/v1/models" || true)
  CODE=${OUT##*$'\n'}
  if [ "$CODE" = "200" ]; then RESP=${OUT%$'\n'*}; break; fi
  sleep 30
done
STATUS=0
if [ "$CODE" = "200" ]; then
  note "models: $RESP"
  # That was an HTTP route. Sessions are WebSockets, and a managed ingress can carry the two differently,
  # so the endpoint answers the same assertions the suite otherwise makes against a server of its own.
  if POSITRONIC_ENDPOINT_URL="$SERVE_URL" AUTH_TOKEN="$AUTH_TOKEN" \
      uv run --directory "$SCRIPT_DIR/../.." --locked \
      pytest positronic/offboard/tests/test_server.py -m endpoint --no-cov >> "$LOG" 2>&1; then
    note "endpoint contract: OK"
  else
    note "endpoint contract: FAILED (see $LOG)"
    STATUS=1
  fi
else
  note "warm-up: FAILED, last status ${CODE:-none}"
  STATUS=1
fi

exit "$STATUS"
