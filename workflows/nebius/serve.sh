#!/usr/bin/env bash
# Submit a Nebius Serverless Endpoint running a vendor inference server.
#
# Usage
#   bash workflows/nebius/serve.sh <vendor> <endpoint-name> [server args...]
#   NEBIUS_PRESET=8gpu-128vcpu-1600gb bash workflows/nebius/serve.sh dreamzero dz-server droid --pipeline.source.num_gpus=8
#   NEBIUS_GRPC_PORT= bash workflows/nebius/serve.sh lerobot ws-only ee \
#     --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/smolvla/<exp_name>/   # the websocket wire alone
#
# The endpoint gets no public IP: Nebius fronts every HTTP container port with a
# managed https:// URL, which is what this polls for and prints. The container
# itself takes ~10-15 min more to finish uv sync and load the model into GPU
# memory after the URL appears.
#
# The endpoint serves the websocket wire on 8000, and the gRPC wire on the port
# `--grpc_port` names. The create declares the gRPC port as an ordinary HTTP port;
# a `/tcp` port gets a front gRPC refuses. The offboard README says what each
# front does to a session.
#
# The managed URL is assigned, never chosen, and a delete plus re-create of the
# same name gets a new one; `stop.sh` deletes, `nebius ai endpoint stop`/`start`
# keeps the URL. See the README's "A managed URL is assigned, not chosen".
#
# The server is gated on a bearer token (AUTH_TOKEN, from MysteryBox). Auth stays
# in-process: `nebius ai endpoint create --auth token` strips the WebSocket
# upgrade headers and passes no inference session.
#
# Hardcoded: GPU platform, websocket port. Vendor selects image + uv extra. One
# setting of its own, via env: NEBIUS_PRESET. Everything shared with the other
# scripts here lives in common.sh.

set -euo pipefail
source "$(dirname "$0")/common.sh"

# Nebius GPU preset. Default is one H100; multi-GPU presets must match the server's GPU count
# (DreamZero's --num_gpus runs torchrun --nproc_per_node, so an 8-GPU server needs an 8-GPU preset).
PRESET="${NEBIUS_PRESET:-1gpu-16vcpu-200gb}"

if [ $# -lt 2 ]; then
  cat >&2 <<'EOF'
Usage: bash workflows/nebius/serve.sh <vendor> <endpoint-name> [server args...]

Vendors: lerobot_0_3_3 | lerobot | openpi | gr00t | dreamzero | molmoact2

The endpoint name must be unique in the project (lowercase alphanumeric + dashes).
Remaining arguments forward to positronic.vendors.<vendor>.server.

Examples:

  # ACT public demo checkpoint (no S3 credentials needed inside the container)
  bash workflows/nebius/serve.sh lerobot_0_3_3 my-act-demo demo

  # Your own ACT checkpoint
  bash workflows/nebius/serve.sh lerobot_0_3_3 act-server ee \
    --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/lerobot/<exp_name>/

  # SmolVLA / lerobot 0.4.x checkpoint
  bash workflows/nebius/serve.sh lerobot smolvla-server ee \
    --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/smolvla/<exp_name>/

  # OpenPI (ee_frame is the EE frame the checkpoint speaks; None means the rig's default)
  bash workflows/nebius/serve.sh openpi pi-server ee \
    --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/openpi/<exp_name>/ \
    --pipeline.ee_frame=None

  # GR00T
  bash workflows/nebius/serve.sh gr00t groot-server ee_rot6d_rel \
    --pipeline.source.checkpoints_dir=s3://<your-bucket>/checkpoints/groot/<exp_name>/
EOF
  exit 1
fi

VENDOR="$1"
NAME="$2"
shift 2

case "$VENDOR" in
  lerobot_0_3_3) IMAGE="positro/positronic:${IMAGE_TAG}"; EXTRA="--extra lerobot_0_3_3 " ;;
  lerobot)       IMAGE="positro/positronic:${IMAGE_TAG}"; EXTRA="--extra lerobot " ;;
  # openpi.server imports `openpi_client` at module top → needs --extra openpi
  openpi)        IMAGE="positro/openpi:${IMAGE_TAG}";     EXTRA="--extra openpi " ;;
  gr00t)         IMAGE="positro/gr00t:${IMAGE_TAG}";      EXTRA="" ;;
  # dreamzero.server imports `huggingface_hub` at module top → needs --extra dreamzero
  dreamzero)     IMAGE="positro/dreamzero:${IMAGE_TAG}";  EXTRA="--extra dreamzero " ;;
  molmoact2)     IMAGE="positro/positronic:${IMAGE_TAG}"; EXTRA="--extra molmoact2 " ;;
  *)
    echo "Unknown vendor: '$VENDOR'. Supported: lerobot_0_3_3 | lerobot | openpi | gr00t | dreamzero | molmoact2" >&2
    exit 1
    ;;
esac

# Serverless endpoints have no native idle/scale-to-zero, so opt the server into
# self-shutdown (the base default is no timeout). Override the window with
# NEBIUS_IDLE_TIMEOUT_MIN; skip injection if the caller already passed one.
case " $* " in
  *" --idle_timeout_min="*|*" --idle_timeout_min "*) ;;
  *) set -- "$@" "--idle_timeout_min=${NEBIUS_IDLE_TIMEOUT_MIN:-20}" ;;
esac

# The websocket port: the create declares it, and the poll below selects the managed URL that fronts it.
WS_PORT=8000

# gRPC is the server's opt-in wire, so the endpoint declares its port only where one is served. A
# caller's own --grpc_port names it; NEBIUS_GRPC_PORT= (empty) serves the websocket wire alone.
ARGS=" $* "
case "$ARGS" in
  *" --grpc_port="*) GRPC_PORT=${ARGS#*--grpc_port=}; GRPC_PORT=${GRPC_PORT%% *} ;;
  *" --grpc_port "*) GRPC_PORT=${ARGS#*--grpc_port }; GRPC_PORT=${GRPC_PORT%% *} ;;
  *)
    GRPC_PORT=${NEBIUS_GRPC_PORT-9000}
    if [ -n "$GRPC_PORT" ]; then set -- "$@" "--grpc_port=${GRPC_PORT}"; fi
    ;;
esac

PORT_ARGS=(--container-port "${WS_PORT}")
if [ -n "$GRPC_PORT" ]; then PORT_ARGS+=(--container-port "${GRPC_PORT}"); fi

SERVER_ARGS="run --python 3.13 ${EXTRA}python -m positronic.vendors.${VENDOR}.server $*"

echo "Creating $VENDOR endpoint '$NAME'..."
nebius ai endpoint create \
  --parent-id "$PARENT_ID" \
  --subnet-id "$SUBNET_ID" \
  --name "$NAME" \
  --image "$IMAGE" \
  --container-command uv \
  --args "$SERVER_ARGS" \
  "${PORT_ARGS[@]}" \
  --platform gpu-h100-sxm \
  --preset "$PRESET" \
  --working-dir /positronic \
  --volume "${CACHE_FS}:/cache:rw" \
  --env UV_CACHE_DIR=/cache/uv \
  --env HF_HOME=/cache/hf \
  --env OPENPI_DATA_HOME=/cache/openpi \
  --env-secret "${AUTH_TOKEN_KEY}=${AUTH_TOKEN_SECRET}" \
  "${S3_ENV_FLAGS[@]}"
# Left on stdout, not discarded: `create` reports `Endpoint ID:` as soon as the resource exists and can
# still fail afterwards — a container that will not start does exactly that — so a caller logging this
# output learns what to release even on the paths where this script never reaches its banner.

ID=$(nebius ai endpoint list --parent-id "$PARENT_ID" --format json \
  | jq -r --arg n "$NAME" '.items[]? | select(.metadata.name==$n) | .metadata.id')

if [ -z "$ID" ]; then
  echo "Endpoint create did not return a known resource for name '$NAME'." >&2
  exit 1
fi

echo "Endpoint ID: $ID"
echo "Waiting for the managed HTTPS URL(s) (typically <1 min)..."

# Each managed URL names the container port it fronts, and that prefix tells the two wires apart.
# This field also carries bare `IP:port` entries, which serve no TLS and would put the bearer token
# on the wire in cleartext: take the https:// ones, and fail with no fallback.
url_for_port() {
  printf '%s' "$1" | jq -r "[.status.public_endpoints[]? | select(startswith(\"https://port$2-\"))] | first // empty"
}

URL=""
GRPC_HOST=""
for i in $(seq 1 30); do
  # One read per pass: a port's tunnel can be published in a later status update than another's, so
  # every declared port waits out the same budget rather than the first one ending it.
  ENDPOINTS=$(nebius ai endpoint get "$ID" --format json 2>/dev/null)
  URL=$(url_for_port "$ENDPOINTS" "$WS_PORT")
  if [ -n "$GRPC_PORT" ]; then
    GRPC_HOST=$(url_for_port "$ENDPOINTS" "$GRPC_PORT" | sed 's|^https://||')
  fi
  if [ -n "$URL" ] && { [ -z "$GRPC_PORT" ] || [ -n "$GRPC_HOST" ]; }; then break; fi
  sleep 10
done

if [ -z "$URL" ]; then
  echo "No managed https:// URL for port ${WS_PORT} within 5 min. Check: nebius ai endpoint get $ID" >&2
  exit 1
fi
if [ -n "$GRPC_PORT" ] && [ -z "$GRPC_HOST" ]; then
  echo "No managed https:// URL for port ${GRPC_PORT} within 5 min. Check: nebius ai endpoint get $ID" >&2
  exit 1
fi

GRPC_BANNER=""
POLICY_URL="$URL"
POLICY_NOTE="Point a rig at the websocket wire:"
if [ -n "$GRPC_PORT" ]; then
  GRPC_URL="grpcs://${GRPC_HOST}:443"
  GRPC_BANNER="  gRPC URL:      ${GRPC_URL}
"
  POLICY_URL="$GRPC_URL"
  POLICY_NOTE="Point a rig at either wire; through this front an 846 KiB observation
round-trips in about 6 ms over gRPC and about 60 ms over the websocket:"
fi

cat <<BANNER

==============================================================
  Endpoint URL:  $URL
${GRPC_BANNER}  Endpoint ID:   $ID
  Endpoint name: $NAME
  Vendor:        $VENDOR
==============================================================

The container is still warming up (image pull + uv sync + checkpoint load,
~10-15 min total). Follow startup logs:

  nebius ai endpoint logs $ID --follow

Once the model is loaded, sanity-check with (see workflows/nebius/README.md
for loading AUTH_TOKEN out of MysteryBox):

  curl -H "Authorization: Bearer \$AUTH_TOKEN" $URL/api/v1/models

$POLICY_NOTE

  --policy=.authed_remote --policy.url='$POLICY_URL'

To release the endpoint:

  bash workflows/nebius/stop.sh $NAME

BANNER
