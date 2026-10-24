#!/usr/bin/env bash
# Score a served policy on a public MolmoSpaces eval: wait for the policy server, run the eval, print the score.
# This is the entrypoint of positro/molmospaces (Dockerfile.molmospaces).
#
# Usage
#   docker run --rm --gpus all -e POLICY_HOST=<host> -e AUTH_TOKEN positro/molmospaces:main
#   docker run --rm --network host -e POLICY_HOST=localhost -e POLICY_PORT=8000 -e POLICY_WIRE=websocket \
#       -e AUTH_TOKEN positro/molmospaces:main [more `positronic eval run` arguments]
#
# Environment
#   POLICY_HOST    the host of the policy server, required
#   AUTH_TOKEN     the bearer token of the policy server, required
#   POLICY_PORT    default 443
#   POLICY_WIRE    websocket_tls (https, the default) or websocket (http)
#   EVAL           a public eval, default molmo.franka_pick_mini_smoke
#   POLICY_WAIT_S  how long to wait for the server to answer its keepalive route, default 1800
#   OUTPUT_DIR     where the episodes go, default /molmospaces/evals
#
# The first run downloads the MolmoSpaces assets, about 13 GB, into MLSPACES_CACHE_DIR. The last line that
# the eval prints is the score: `score <share> over <n> episodes, <u> unscored`.
set -euo pipefail

: "${POLICY_HOST:?set POLICY_HOST to the host of the policy server}"
: "${AUTH_TOKEN:?set AUTH_TOKEN to the bearer token of the policy server}"
port="${POLICY_PORT:-443}"
wire="${POLICY_WIRE:-websocket_tls}"
eval_name="${EVAL:-molmo.franka_pick_mini_smoke}"
wait_s="${POLICY_WAIT_S:-1800}"
output_dir="${OUTPUT_DIR:-/molmospaces/evals}"

case "$wire" in
  websocket_tls) scheme=https ;;
  websocket) scheme=http ;;
  *) echo "POLICY_WIRE must be websocket_tls or websocket, not $wire" >&2; exit 2 ;;
esac
keepalive="$scheme://$POLICY_HOST:$port/api/v1/keepalive"

# curl reads the header from stdin, so the token is not in its command line.
answered() {
  printf 'header = "Authorization: Bearer %s"\n' "$AUTH_TOKEN" \
    | curl --config - --silent --show-error --fail --max-time 10 -X POST "$keepalive"
}

echo "Waiting up to ${wait_s}s for $keepalive"
deadline=$((SECONDS + wait_s))
# With --fail, a failed call prints nothing to stdout, so $answer holds the error of the last try.
until answer=$(answered 2>&1); do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "The policy server did not answer $keepalive in ${wait_s}s. The last error: $answer" >&2
    exit 1
  fi
  sleep 10
done
echo "The policy server answered: $answer"

# MuJoCo renders on the GPU through the NVIDIA EGL library. A host that does not give the container that
# library gets software EGL, which renders on the CPU and is slower.
if [ -z "${MUJOCO_GL:-}" ]; then
  export MUJOCO_GL=egl
  if ! ldconfig -p | grep -q libEGL_nvidia; then
    echo "No NVIDIA EGL library in the container: MuJoCo renders on the CPU"
    export EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
  fi
fi

mkdir -p "$MLSPACES_ASSETS_DIR" "$MLSPACES_CACHE_DIR" "$output_dir"
exec positronic eval run --eval="$eval_name" \
  --policy=.authed_remote --policy.wire="$wire" \
  --policy.address.host="$POLICY_HOST" --policy.address.port="$port" \
  --output_dir="$output_dir" "$@"
