#!/usr/bin/env bash
# Start a policy image the way the platform does, and wait until its server answers.
#
# Usage
#   bash docker/test_policy_image.sh <image> [arg...]          # with no GPU, for up to 900 s
#   bash docker/test_policy_image.sh --gpu <image> [arg...]    # with GPU 0, for up to 600 s
#   bash docker/test_policy_image.sh <molmoact2 image> --model.device_map=cpu
#
# Each arg after the image goes to the image's start command.
#
# The container gets the token `test` in AUTH_TOKEN, no capabilities, and a Docker network with no egress.
# Every 5 s, the script sends POST /api/v1/keepalive to port 8000 from outside the container, as the
# platform does. It prints one result line and the end of the log, and exits with 0 only when the server
# answers 200 with the keepalive answer.
# Needs Docker on amd64 Linux, curl and python3.
set -euo pipefail

gpu_flags=()
limit=900
if [ "${1:-}" = --gpu ]; then
  # The platform gives the image one GPU, so a model that spreads over more GPUs must fail here too.
  gpu_flags=(--gpus device=0)
  limit=600
  shift
fi
image="${1:?usage: $0 [--gpu] <image> [arg...]}"
shift
# The policy contract, as positronic names it: `policy_container.POLICY_PORT` and `AUTH_TOKEN_ENV`,
# `positronic_model_server.protocol.AUTH_HEADER` and `bearer`, `positronic_wire.wire.KEEPALIVE_PATH` and
# `ALIVE_SECONDS`, and `positronic_wire.websocket` `_MAX_KEEPALIVE_BODY_BYTES`.
port=8000
token_env=AUTH_TOKEN
auth_header=Authorization
keepalive_path=/api/v1/keepalive
alive_key=alive_seconds
max_answer_bytes=16384
token=test
auth_value="Bearer $token"

for tool in curl python3; do
  command -v "$tool" >/dev/null || { echo "This test needs $tool." >&2; exit 1; }
done
if [ "$(uname -s)" != Linux ] || [ "$(docker version -f '{{.Server.Arch}}')" != amd64 ]; then
  echo "Run this test with Docker on an amd64 Linux machine. Docker on another machine emulates or hides it." >&2
  exit 1
fi
# The script calls the container's address on its network. Rootless Docker keeps that address in its own
# namespace, Docker Desktop keeps it in a VM, and an internal network publishes no port.
if docker info -f '{{json .SecurityOptions}}' | grep -q 'name=rootless' ||
  [ "$(docker info -f '{{.OperatingSystem}}')" = 'Docker Desktop' ]; then
  echo "This Docker is rootless or Docker Desktop, and the host cannot reach the container." >&2
  echo "Run the test with the system Docker Engine." >&2
  exit 1
fi
# A network of this test alone: a container on a shared network could get files from the other containers on it.
network="policy-test-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
cid=""
body=$(mktemp)
curl_errors=$(mktemp)
# Remove the files $1 and $2, the container $4 when there is one, and the network $3.
cleanup() {
  local container=$4
  rm -f "$1" "$2"
  [ -z "$container" ] || docker rm -f "$container" >/dev/null
  docker network rm "$3" >/dev/null
}
docker network create --internal "$network" >/dev/null
trap 'cleanup "$body" "$curl_errors" "$network" "$cid"' EXIT
cid=$(docker run -d --pull=always --network "$network" --cap-drop ALL --security-opt no-new-privileges \
  "${gpu_flags[@]}" -e "$token_env=$token" "$image" "$@")
ip=$(docker inspect -f "{{(index .NetworkSettings.Networks \"$network\").IPAddress}}" "$cid")

# A socket that listens on port $2 of 127.0.0.1 or ::1, in /proc/net/tcp of the container $1. The platform cannot
# reach it. An image with no `cat` gives no answer, and the test then waits as for any closed port.
loopback_only() {
  local container=$1 hex
  hex=$(printf '%04X' "$2")
  docker exec "$container" cat /proc/net/tcp /proc/net/tcp6 2>/dev/null |
    awk -v v4="0100007F:$hex" -v v6="00000000000000000000000001000000:$hex" \
      '$4 == "0A" && ($2 == v4 || $2 == v6) {found = 1} END {exit !found}'
}

# Succeed when the first $3 bytes of the file $1 hold the keepalive answer as `positronic_wire.websocket` reads
# it: a JSON object whose key $2 is an integer or null. A 200 with any other body fails the run on the platform.
keepalive_answer() {
  python3 -c 'import json, sys
try:
    with open(sys.argv[1], "rb") as f:
        alive = json.loads(f.read(int(sys.argv[3])))[sys.argv[2]]
except (ValueError, LookupError, TypeError, RecursionError):
    sys.exit(1)
sys.exit(0 if alive is None or type(alive) is int else 1)' "$1" "$2" "$3"
}

# POST once to the URL $1 with the header $2, with the files $3 for the body and $4 for curl's errors. Print
# `silent` for no answer, `cold <code>` for a server that is not ready, `ok` for a 200, or `fail <reason>`.
probe() {
  local status=0 code body=$3 errors=$4
  code=$(curl -sS --noproxy '*' -o "$body" -w '%{http_code}' -m 5 -X POST -H "$2" "$1" 2>"$errors") || status=$?
  # curl exits with 7 on a closed port, 28 on no answer in 5 s, and 52 or 56 on a dropped connection. These
  # ignore the HTTP code, as a server can send its headers and then stop.
  if [ "$status" = 7 ] || [ "$status" = 28 ] || [ "$status" = 52 ] || [ "$status" = 56 ]; then
    echo silent
  elif [ "$status" != 0 ]; then
    echo "fail curl failed: $(cat "$errors")"
  elif [ "$code" = 429 ] || [ "$code" -ge 500 ]; then
    # The platform reads these as a server that is not ready yet, and asks again until its deadline.
    echo "cold $code"
  elif [ "$code" != 200 ]; then
    echo "fail the server answered HTTP $code"
  else
    echo ok
  fi
}

start=$SECONDS
passed=false
waiting_on=""
while :; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$cid")" != true ]; then
    result="the server exited"
    break
  fi
  if [ $((SECONDS - start)) -ge "$limit" ]; then
    result="no answer in $limit s$waiting_on"
    break
  fi
  outcome=$(probe "http://$ip:$port$keepalive_path" "$auth_header: $auth_value" "$body" "$curl_errors")
  case $outcome in
    ok)
      if keepalive_answer "$body" "$alive_key" "$max_answer_bytes"; then
        result="the server answered: $(cat "$body")"
        passed=true
      else
        result="the server answered 200 without {\"$alive_key\": <an integer or null>}: $(head -c 200 "$body")"
      fi
      break ;;
    fail*)
      result=${outcome#fail }
      break ;;
    cold*)
      waiting_on=", the last answer was HTTP ${outcome#cold }" ;;
    silent)
      if loopback_only "$cid" "$port"; then
        result="the server listens only on 127.0.0.1, and the platform connects from outside the container"
        break
      fi ;;
  esac
  sleep 5
done
echo "$result, after $((SECONDS - start)) s"
echo "--- the last 30 lines of the log ---"
docker logs --tail 30 "$cid" 2>&1
[ "$passed" = true ]
