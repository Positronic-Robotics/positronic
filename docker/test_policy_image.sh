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
# The policy contract, as positronic names it: `policy_container.POLICY_PORT` and `AUTH_TOKEN_ENV`,
# `positronic_model_server.protocol.AUTH_HEADER` and `bearer`, and `positronic_wire.wire.KEEPALIVE_PATH`.
port=8000
token_env=AUTH_TOKEN
auth_header=Authorization
keepalive_path=/api/v1/keepalive
token=test

if [ "$(uname -s)" != Linux ] || [ "$(docker version -f '{{.Server.Arch}}')" != amd64 ]; then
  echo "Run this test with Docker on an amd64 Linux machine. Docker on another machine emulates or hides it." >&2
  exit 1
fi
docker network inspect "$network" >/dev/null 2>&1 || docker network create --internal "$network" >/dev/null
if [ "$(docker network inspect -f '{{.Internal}}' "$network")" != true ]; then
  echo "The Docker network $network has egress. Remove it with: docker network rm $network" >&2
  exit 1
fi
cid=$(docker run -d --pull=always --network "$network" --cap-drop ALL --security-opt no-new-privileges \
  "${gpu_flags[@]}" -e "$token_env=$token" "$image")
body=$(mktemp)
curl_errors=$(mktemp)
trap 'docker rm -f "$cid" >/dev/null; rm -f "$body" "$curl_errors"' EXIT
ip=$(docker inspect -f "{{(index .NetworkSettings.Networks \"$network\").IPAddress}}" "$cid")

# A socket that listens on $port of 127.0.0.1 or ::1, in /proc/net/tcp of the container. The platform cannot
# reach it. An image with no `cat` gives no answer, and the test then waits as for any closed port.
loopback_only() {
  local hex
  hex=$(printf '%04X' "$port")
  docker exec "$cid" cat /proc/net/tcp /proc/net/tcp6 2>/dev/null |
    awk -v v4="0100007F:$hex" -v v6="00000000000000000000000001000000:$hex" \
      '$4 == "0A" && ($2 == v4 || $2 == v6) {found = 1} END {exit !found}'
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
  # curl exits with 7 on a closed port, 28 on no answer in 5 s, and 52 or 56 on a dropped connection.
  status=0
  code=$(curl -sS --noproxy '*' -o "$body" -w '%{http_code}' -m 5 -X POST -H "$auth_header: Bearer $token" \
    "http://$ip:$port$keepalive_path" 2>"$curl_errors") || status=$?
  if [ "$status" != 0 ] && [ "$status" != 7 ] && [ "$status" != 28 ] && [ "$status" != 52 ] && [ "$status" != 56 ]; then
    result="curl failed: $(cat "$curl_errors")"
    break
  fi
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
