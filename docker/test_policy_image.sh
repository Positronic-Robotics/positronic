#!/usr/bin/env bash
# Start a policy image the way the platform does, and wait until its server answers.
#
# Usage
#   bash docker/test_policy_image.sh <image>          # with no GPU, for up to 900 s
#   bash docker/test_policy_image.sh --gpu <image>    # with GPU 0, for up to 600 s
#
# The container gets the token `test` in AUTH_TOKEN, no capabilities, and a Docker network with no egress.
# Every 5 s, the script sends POST /api/v1/keepalive to port 8000 from outside the container, as the
# platform does. It prints one result line and the end of the log, and exits with 0 only when the server
# answers 200 with the keepalive answer.
# Needs Docker on amd64 Linux, and curl.
set -euo pipefail

gpu_flags=()
limit=900
if [ "${1:-}" = --gpu ]; then
  # The platform gives the image one GPU, so a model that spreads over more GPUs must fail here too.
  gpu_flags=(--gpus device=0)
  limit=600
  shift
fi
image="${1:?usage: $0 [--gpu] <image>}"
network=policy-test
# The policy contract, as positronic names it: `policy_container.POLICY_PORT` and `AUTH_TOKEN_ENV`,
# `positronic_model_server.protocol.AUTH_HEADER` and `bearer`, and `positronic_wire.wire.KEEPALIVE_PATH` and
# `ALIVE_SECONDS`.
port=8000
token_env=AUTH_TOKEN
auth_header=Authorization
keepalive_path=/api/v1/keepalive
alive_key=alive_seconds
token=test
auth_value="Bearer $token"
# The keepalive answer as `positronic_wire.websocket` reads it: a JSON object whose $alive_key is an integer or
# null. A 200 with any other body fails the run on the platform.
keepalive_answer="^[[:space:]]*\\{.*\"$alive_key\"[[:space:]]*:[[:space:]]*(null|-?[0-9]+)[[:space:]]*[,}]"

if [ "$(uname -s)" != Linux ] || [ "$(docker version -f '{{.Server.Arch}}')" != amd64 ]; then
  echo "Run this test with Docker on an amd64 Linux machine. Docker on another machine emulates or hides it." >&2
  exit 1
fi
# Rootless Docker keeps the container's address inside its own namespace, and an internal network publishes no port.
if docker info -f '{{json .SecurityOptions}}' | grep -q 'name=rootless'; then
  echo "This Docker is rootless, and the host cannot reach the container. Run the test with the system Docker." >&2
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

# A socket that listens on port $2 of 127.0.0.1 or ::1, in /proc/net/tcp of the container $1. The platform cannot
# reach it. An image with no `cat` gives no answer, and the test then waits as for any closed port.
loopback_only() {
  local container=$1 hex
  hex=$(printf '%04X' "$2")
  docker exec "$container" cat /proc/net/tcp /proc/net/tcp6 2>/dev/null |
    awk -v v4="0100007F:$hex" -v v6="00000000000000000000000001000000:$hex" \
      '$4 == "0A" && ($2 == v4 || $2 == v6) {found = 1} END {exit !found}'
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
  # curl exits with 7 on a closed port, 28 on no answer in 5 s, and 52 or 56 on a dropped connection.
  status=0
  code=$(curl -sS --noproxy '*' -o "$body" -w '%{http_code}' -m 5 -X POST -H "$auth_header: $auth_value" \
    "http://$ip:$port$keepalive_path" 2>"$curl_errors") || status=$?
  # These waits ignore the HTTP code, as a server can send its headers and then stop.
  if [ "$status" = 7 ] || [ "$status" = 28 ] || [ "$status" = 52 ] || [ "$status" = 56 ]; then
    if loopback_only "$cid" "$port"; then
      result="the server listens only on 127.0.0.1, and the platform connects from outside the container"
      break
    fi
    sleep 5
  elif [ "$status" != 0 ]; then
    result="curl failed: $(cat "$curl_errors")"
    break
  elif [ "$code" = 200 ] && tr -d '\n' <"$body" | grep -Eq "$keepalive_answer"; then
    result="the server answered: $(cat "$body")"
    passed=true
    break
  elif [ "$code" = 429 ] || [ "$code" -ge 500 ]; then
    # The platform reads these as a server that is not ready yet, and asks again until its deadline.
    waiting_on=", the last answer was HTTP $code"
    sleep 5
  elif [ "$code" = 200 ]; then
    result="the server answered 200 without {\"$alive_key\": <an integer or null>}: $(head -c 200 "$body")"
    break
  else
    result="the server answered HTTP $code"
    break
  fi
done
echo "$result, after $((SECONDS - start)) s"
echo "--- the last 30 lines of the log ---"
docker logs --tail 30 "$cid" 2>&1
[ "$passed" = true ]
