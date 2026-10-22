#!/usr/bin/env bash
# Start a policy image as a Nebius Serverless job, and wait until its server answers.
#
# Usage
#   bash docker/test_policy_image_serverless.sh <image>
#   bash docker/test_policy_image_serverless.sh --gpu <image>
#   bash docker/test_policy_image_serverless.sh --registry-username=<user> \
#       --registry-password-file=<file> <image>
#
# The job runs the image with no command, so the image ENTRYPOINT starts the server. The container
# gets the token `test` in AUTH_TOKEN. Every 5 s, the script sends POST /api/v1/keepalive to port 8000
# of the container, from the machine under the job. It prints one result line and the end of the log,
# and exits with 0 only when the server answers 200 with the keepalive answer.
# The job has a network. A failure that appears only with no egress does not appear here.
# docker/test_policy_image.sh is that test.
# The script deletes the job when it exits. The job costs money until that delete.
# Needs the nebius CLI with a profile, ssh, curl, python3 and jq.
#
# Settings, as environment variables. With --gpu, the defaults are gpu-l40s-a and 1gpu-8vcpu-32gb.
#   NEBIUS_JOB_NAME   name of the job, default policy-image-test
#   NEBIUS_PLATFORM   default cpu-d3
#   NEBIUS_PRESET     default 8vcpu-32gb
#   NEBIUS_DISK_SIZE  disk of the job, default 500Gi
#   NEBIUS_SSH_KEY    private key, default ~/.ssh/id_ed25519. A key with a passphrase must be in ssh-agent.
set -euo pipefail

gpu=false
limit=900
platform="${NEBIUS_PLATFORM:-cpu-d3}"
preset="${NEBIUS_PRESET:-8vcpu-32gb}"
registry_user=""
password_file=""
image=""
usage() {
  echo "usage: $0 [--gpu] [--registry-username=<user> --registry-password-file=<file>] <image>" >&2
  exit 2
}
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)
      gpu=true
      limit=600
      [ -n "${NEBIUS_PLATFORM:-}" ] || platform=gpu-l40s-a
      [ -n "${NEBIUS_PRESET:-}" ] || preset=1gpu-8vcpu-32gb
      shift ;;
    --registry-username)
      [ -n "${2:-}" ] || usage
      registry_user=$2
      shift 2 ;;
    --registry-username=*)
      registry_user="${1#*=}"
      shift ;;
    --registry-password-file)
      [ -n "${2:-}" ] || usage
      password_file=$2
      shift 2 ;;
    --registry-password-file=*)
      password_file="${1#*=}"
      shift ;;
    --)
      shift
      break ;;
    -*)
      usage ;;
    *)
      [ -z "$image" ] || usage
      image=$1
      shift ;;
  esac
done
[ -n "$image" ] || usage
if [ -n "$registry_user" ] && [ -z "$password_file" ]; then
  echo "A private image needs --registry-password-file." >&2
  exit 2
fi
if [ -z "$registry_user" ] && [ -n "$password_file" ]; then
  echo "A password file needs --registry-username." >&2
  exit 2
fi

name="${NEBIUS_JOB_NAME:-policy-image-test}"
disk_size="${NEBIUS_DISK_SIZE:-500Gi}"
ssh_key="${NEBIUS_SSH_KEY:-$HOME/.ssh/id_ed25519}"
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

for tool in nebius ssh curl python3 jq; do
  command -v "$tool" >/dev/null || { echo "This test needs $tool." >&2; exit 1; }
done

project=$(nebius config get parent-id)
job_id=""
host_ip=""
cname=""
cip=""
body=$(mktemp)
curl_errors=$(mktemp)
create_log=$(mktemp)
create_err=$(mktemp)
control_path="/tmp/nebius-ssh-$$"
ssh_opts=(-i "$ssh_key" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o StrictHostKeyChecking=accept-new
  -o ControlMaster=auto -o ControlPersist=120 -o "ControlPath=$control_path")

# Delete the job when one was created, then remove the local files.
cleanup() {
  local status=$?
  ssh "${ssh_opts[@]}" -O exit "nebius@${host_ip:-none}" >/dev/null 2>&1 || true
  rm -f "$body" "$curl_errors" "$create_log" "$create_err" "$control_path"
  if [ -n "$job_id" ]; then
    echo "The job costs money until Nebius deletes it."
    nebius ai job delete "$job_id" >/dev/null || echo "Nebius could not delete the job $job_id." >&2
    echo "Deleted the job $name."
    job_id=""
  fi
  exit "$status"
}
trap cleanup EXIT

# Print the public key of $ssh_key as `<type> <key>`, with no comment. ssh cannot ask for a passphrase,
# so the key must be readable without one, or be in the agent.
public_key() {
  local key blob
  if key=$(ssh-keygen -y -P "" -f "$ssh_key" 2>/dev/null); then
    key=$(cut -d' ' -f1,2 <<<"$key")
    # ssh offers the key in <key>.pub when the file exists, so a stale file makes the server refuse it.
    if [ -f "$ssh_key.pub" ] && [ "$(cut -d' ' -f1,2 "$ssh_key.pub")" != "$key" ]; then
      echo "$ssh_key.pub is not the public key of $ssh_key. Make it again with:" >&2
      echo "  ssh-keygen -y -f $(printf %q "$ssh_key") > $(printf %q "$ssh_key.pub")" >&2
      return 1
    fi
    echo "$key"
    return
  fi
  if [ ! -f "$ssh_key" ] && [ ! -f "$ssh_key.pub" ]; then
    echo "No key at $ssh_key. Run ssh-keygen -t ed25519, or set NEBIUS_SSH_KEY." >&2
    return 1
  fi
  if [ ! -f "$ssh_key.pub" ]; then
    echo "No public key at $ssh_key.pub. Make it with:" >&2
    echo "  ssh-keygen -y -f $(printf %q "$ssh_key") > $(printf %q "$ssh_key.pub")" >&2
    return 1
  fi
  blob=$(cut -d' ' -f2 "$ssh_key.pub")
  if [ -z "$blob" ] || ! ssh-add -L 2>/dev/null | cut -d' ' -f2 | grep -qxF "$blob"; then
    echo "ssh cannot use $ssh_key without a passphrase. Run: ssh-add $(printf %q "$ssh_key")" >&2
    return 1
  fi
  cut -d' ' -f1,2 "$ssh_key.pub"
}

# The id of the job named $name, or nothing when Nebius has no such job.
existing_job() {
  local errors id
  errors=$(mktemp)
  if ! id=$(nebius ai job get-by-name --parent-id "$project" --name "$name" --format jsonpath='{.metadata.id}' \
    2>"$errors"); then
    if ! grep -q 'code = NotFound' "$errors"; then
      cat "$errors" >&2
      rm -f "$errors"
      return 1
    fi
    id=""
  fi
  rm -f "$errors"
  printf '%s' "$id"
}

# Run one command on the machine under the job.
ssh_host() {
  ssh "${ssh_opts[@]}" "nebius@$host_ip" "$@"
}

# Succeed when the first $3 bytes of the file $1 hold the keepalive answer as `positronic_wire.websocket`
# reads it: a JSON object whose key $2 is an integer or null. A 200 with any other body fails the run.
keepalive_answer() {
  python3 -c 'import json, sys
try:
    with open(sys.argv[1], "rb") as f:
        alive = json.loads(f.read(int(sys.argv[3])))[sys.argv[2]]
except (ValueError, LookupError, TypeError, RecursionError):
    sys.exit(1)
sys.exit(0 if alive is None or type(alive) is int else 1)' "$1" "$2" "$3"
}

# POST once from the machine under the job. Print `silent`, `cold <code>`, `ok`, or `fail <reason>`.
probe() {
  local status=0 line curl_status code
  line=$(ssh "${ssh_opts[@]}" "nebius@$host_ip" bash -s -- "$cip" "$port" "$keepalive_path" \
    "$auth_header: $auth_value" <<'EOF'
ip=$1 port=$2 path=$3 header=$4
status=0
code=$(curl -sS --noproxy '*' -o /tmp/policy-keepalive-body -w '%{http_code}' -m 5 -X POST \
  -H "$header" "http://${ip}:${port}${path}" 2>/tmp/policy-keepalive-err) || status=$?
printf '%s %s\n' "$status" "${code:-000}"
EOF
) || status=$?
  if [ "$status" != 0 ]; then
    echo "fail curl failed: ssh to $host_ip failed"
    return
  fi
  curl_status=${line%% *}
  code=${line#* }
  ssh "${ssh_opts[@]}" "nebius@$host_ip" "cat /tmp/policy-keepalive-body 2>/dev/null || true" >"$body" || true
  ssh "${ssh_opts[@]}" "nebius@$host_ip" "cat /tmp/policy-keepalive-err 2>/dev/null || true" >"$curl_errors" || true
  # curl exits with 7 on a closed port, 28 on no answer in 5 s, and 52 or 56 on a dropped connection.
  if [ "$curl_status" = 7 ] || [ "$curl_status" = 28 ] || [ "$curl_status" = 52 ] || [ "$curl_status" = 56 ]; then
    echo silent
  elif [ "$curl_status" != 0 ]; then
    echo "fail curl failed: $(cat "$curl_errors")"
  elif [ "$code" = 429 ] || [ "${code:-0}" -ge 500 ] 2>/dev/null; then
    echo "cold $code"
  elif [ "$code" != 200 ]; then
    echo "fail the server answered HTTP $code"
  else
    echo ok
  fi
}

# A socket that listens on port 8000 of 127.0.0.1 or ::1. The platform cannot reach it.
loopback_only() {
  local hex
  hex=$(printf '%04X' "$port")
  ssh_host sudo -n docker exec "$cname" cat /proc/net/tcp /proc/net/tcp6 2>/dev/null |
    awk -v v4="0100007F:$hex" -v v6="00000000000000000000000001000000:$hex" \
      '$4 == "0A" && ($2 == v4 || $2 == v6) {found = 1} END {exit !found}'
}

# True when the workload container is still running.
container_running() {
  [ "$(ssh_host sudo -n docker inspect -f '{{.State.Running}}' "$cname" 2>/dev/null || true)" = true ]
}

# Read an aijob id from the create output in $1.
job_id_from() {
  local raw json id
  raw=$(cat "$1")
  json=$(printf '%s\n' "$raw" | sed -n '/^{/,$p')
  id=""
  if [ -n "$json" ]; then
    id=$(jq -r '.resource_id // .metadata.id // empty' <<<"$json" 2>/dev/null || true)
  fi
  if [ -z "$id" ]; then
    id=$(printf '%s\n' "$raw" | sed -n 's/.*\(aijob-[a-z0-9]*\).*/\1/p' | head -1)
  fi
  printf '%s' "$id"
}

pub=$(public_key) || exit 1
if ! existing=$(existing_job); then
  exit 1
fi
if [ -n "$existing" ]; then
  echo "The job $name exists. Delete it with: nebius ai job delete $existing" >&2
  exit 1
fi

registry_password=""
if [ -n "$password_file" ]; then
  [ -f "$password_file" ] || { echo "No password file at $password_file." >&2; exit 1; }
  # One trailing line ending is not part of the password. The same rule is in the competition guide.
  registry_password=$(python3 -c 'import pathlib,sys
p=pathlib.Path(sys.argv[1]).read_bytes()
if p.endswith(b"\r\n"):
    p=p[:-2]
elif p.endswith(b"\n"):
    p=p[:-1]
sys.stdout.buffer.write(p)' "$password_file")
fi

echo "Creating the job $name for $image ($platform $preset, disk $disk_size)."
echo "The job costs money until this script deletes it."
# Bash 3.2 treats an empty "${array[@]}" as unset under `set -u`, and an `if` then exits 0.
create_args=(--parent-id "$project" --name "$name" --image "$image" --env "$token_env=$token"
  --timeout 1h --platform "$platform" --preset "$preset" --disk-size "$disk_size" --ssh-key "$pub")
if [ -n "$registry_user" ]; then
  create_args+=(--registry-username "$registry_user" --registry-password "$registry_password")
fi
if ! nebius ai job create "${create_args[@]}" --async --format json >"$create_log" 2>"$create_err"; then
  cat "$create_err" >&2
  job_id=$(job_id_from "$create_log")
  [ -n "$job_id" ] || job_id=$(job_id_from "$create_err")
  echo "Nebius did not create the job." >&2
  exit 1
fi
job_id=$(job_id_from "$create_log")
[ -n "$job_id" ] || job_id=$(job_id_from "$create_err")
[ -n "$job_id" ] || { echo "Nebius did not return a job id." >&2; cat "$create_log" "$create_err" >&2; exit 1; }
echo "Job id: $job_id"

# Image pull sits in this wait. The probe limit below starts after the container has an address.
start_deadline=$((SECONDS + 2400))
state=""
while :; do
  job_json=$(nebius ai job get "$job_id" --format json)
  next=$(jq -r '.status.state // empty' <<<"$job_json")
  if [ "$next" != "$state" ]; then
    state=$next
    echo "The job is $state."
  fi
  case "$state" in
    COMPLETED|FAILED|ERROR|CANCELLED)
      echo "the server exited, after 0 s"
      echo "--- the job log ---"
      nebius ai job logs "$job_id" || true
      exit 1 ;;
  esac
  host_ip=$(jq -r '.status.instances[0].public_ip // empty' <<<"$job_json")
  if [ "$state" = RUNNING ] && [ -n "$host_ip" ]; then
    break
  fi
  [ "$SECONDS" -lt "$start_deadline" ] || { echo "The job was not running in 2400 s." >&2; exit 1; }
  sleep 10
done

if [ -f "$HOME/.ssh/known_hosts" ]; then
  ssh-keygen -R "$host_ip" >/dev/null 2>&1 || true
fi
ssh_deadline=$((SECONDS + 180))
while :; do
  ssh "${ssh_opts[@]}" "nebius@$host_ip" true >/dev/null 2>&1 && break
  [ "$SECONDS" -lt "$ssh_deadline" ] || { echo "No ssh to $host_ip." >&2; exit 1; }
  sleep 5
done

cname="${job_id}-job-1"
while :; do
  run_fmt='{{.State.Running}}'
  ip_fmt='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
  running=$(ssh_host sudo -n docker inspect -f "$run_fmt" "$cname" 2>/dev/null || true)
  cip=$(ssh_host sudo -n docker inspect -f "$ip_fmt" "$cname" 2>/dev/null || true)
  if [ "$running" = true ] && [ -n "$cip" ]; then
    break
  fi
  # A stopped container still has an address. An absent container has neither.
  if [ "$running" = false ]; then
    echo "the server exited, after 0 s"
    echo "--- the last 30 lines of the log ---"
    ssh_host sudo -n docker logs --tail 30 "$cname" 2>&1 || nebius ai job logs "$job_id" || true
    exit 1
  fi
  job_state=$(nebius ai job get "$job_id" --format jsonpath='{.status.state}')
  case "$job_state" in
    COMPLETED|FAILED|ERROR|CANCELLED)
      echo "the server exited, after 0 s"
      echo "--- the job log ---"
      nebius ai job logs "$job_id" || true
      exit 1 ;;
  esac
  [ "$SECONDS" -lt "$start_deadline" ] || { echo "The container had no address in the start wait." >&2; exit 1; }
  sleep 5
done

start=$SECONDS
passed=false
waiting_on=""
result=""
while :; do
  if ! container_running; then
    result="the server exited"
    break
  fi
  if [ $((SECONDS - start)) -ge "$limit" ]; then
    result="no answer in $limit s$waiting_on"
    break
  fi
  outcome=$(probe)
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
      if loopback_only; then
        result="the server listens only on 127.0.0.1, and the platform connects from outside the container"
        break
      fi ;;
  esac
  sleep 5
done
echo "$result, after $((SECONDS - start)) s"
echo "--- the last 30 lines of the log ---"
ssh_host sudo -n docker logs --tail 30 "$cname" 2>&1 || nebius ai job logs "$job_id" || true
[ "$passed" = true ]
