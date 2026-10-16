#!/usr/bin/env bash
# Start a policy image the way the platform does, and wait until its server answers.
#
# Usage
#   bash docker/test_policy_image.sh <image>          # with no GPU, for up to 900 s
#   bash docker/test_policy_image.sh --gpu <image>    # with all GPUs, for up to 600 s
#
# The container gets the token `test` in AUTH_TOKEN, no capabilities, and a Docker network with no egress.
# Every 5 s, the script sends POST /api/v1/keepalive to port 8000 from outside the container, as the
# platform does. It prints one result line and the end of the log, and exits with 0 only on an HTTP 200.
# Needs Docker on amd64 Linux, and curl.
set -euo pipefail

gpu_flags=()
limit=900
if [ "${1:-}" = --gpu ]; then
  gpu_flags=(--gpus all)
  limit=600
  shift
fi
image="${1:?usage: $0 [--gpu] <image>}"
network=policy-test

if [ "$(uname -s)" != Linux ] || [ "$(docker version -f '{{.Server.Arch}}')" != amd64 ]; then
  echo "Run this test with Docker on an amd64 Linux machine. Docker on another machine emulates or hides it." >&2
  exit 1
fi
docker network inspect "$network" >/dev/null 2>&1 || docker network create --internal "$network" >/dev/null
cid=$(docker run -d --pull=always --network "$network" --cap-drop ALL --security-opt no-new-privileges \
  "${gpu_flags[@]}" -e AUTH_TOKEN=test "$image")
body=$(mktemp)
trap 'docker rm -f "$cid" >/dev/null; rm -f "$body"' EXIT
ip=$(docker inspect -f "{{(index .NetworkSettings.Networks \"$network\").IPAddress}}" "$cid")

# A socket that listens on port 8000 (1F40) of 127.0.0.1 or ::1, in /proc/net/tcp of the container. The platform
# cannot reach it. An image with no `cat` gives no answer, and the test then waits as for any closed port.
loopback_only() {
  docker exec "$cid" cat /proc/net/tcp /proc/net/tcp6 2>/dev/null |
    awk '$4 == "0A" && ($2 == "0100007F:1F40" || $2 == "00000000000000000000000001000000:1F40") {found = 1}
         END {exit !found}'
}

start=$SECONDS
code=000
while :; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$cid")" != true ]; then
    result="the server exited"
    break
  fi
  if [ $((SECONDS - start)) -ge "$limit" ]; then
    result="no answer in $limit s"
    break
  fi
  code=$(curl -s -o "$body" -w '%{http_code}' -m 5 -X POST -H 'Authorization: Bearer test' \
    "http://$ip:8000/api/v1/keepalive") || code=000
  if [ "$code" = 200 ]; then
    result="the server answered: $(cat "$body")"
    break
  elif [ "$code" != 000 ]; then
    result="the server answered HTTP $code"
    break
  elif loopback_only; then
    result="the server listens only on 127.0.0.1, and the platform connects from outside the container"
    break
  fi
  sleep 5
done
echo "$result, after $((SECONDS - start)) s"
echo "--- the last 30 lines of the log ---"
docker logs --tail 30 "$cid" 2>&1
[ "$code" = 200 ]
