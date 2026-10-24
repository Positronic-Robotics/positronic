#!/usr/bin/env bash
# Create a Nebius Serverless job to build and test a policy image on, and delete it.
#
# Usage
#   bash docker/nebius_serverless.sh create     # create the job, wait for it, and print the ssh command
#   bash docker/nebius_serverless.sh delete     # delete the job and its disk
#
# The job runs `sleep` on ubuntu:24.04. Docker is on the machine under the job, not in the container.
# Connect with the printed ssh command. The script adds user `nebius` to the `docker` group, so `docker`
# works in that connection. The job container has no Docker. `nebius ai job ssh` opens that container.
# Needs the nebius CLI with a profile, and ssh. The job is in the project of the profile.
#
# Settings, as environment variables:
#   NEBIUS_JOB_NAME   name of the job, default policy-image-build
#   NEBIUS_PLATFORM   default gpu-l40s-a; cpu-d3 gives a job with no GPU
#   NEBIUS_PRESET     default 1gpu-8vcpu-32gb; 8vcpu-32gb on cpu-d3
#   NEBIUS_DISK_SIZE  disk of the job, default 500Gi
#   NEBIUS_TIMEOUT    when the job ends if you do not delete it, default 8h
#   NEBIUS_SSH_KEY    private key, default ~/.ssh/id_ed25519. A key with a passphrase must be in ssh-agent.
set -euo pipefail

name="${NEBIUS_JOB_NAME:-policy-image-build}"
platform="${NEBIUS_PLATFORM:-gpu-l40s-a}"
preset="${NEBIUS_PRESET:-1gpu-8vcpu-32gb}"
disk_size="${NEBIUS_DISK_SIZE:-500Gi}"
timeout="${NEBIUS_TIMEOUT:-8h}"
ssh_key="${NEBIUS_SSH_KEY:-$HOME/.ssh/id_ed25519}"
# The ssh command to print. Plain ssh finds only a key at a default path.
ssh_cmd=ssh
[ -z "${NEBIUS_SSH_KEY:-}" ] || ssh_cmd="ssh -i $(printf %q "$ssh_key") -o IdentitiesOnly=yes"
script="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
delete_cmd="bash $(printf %q "$script") delete"
[ -z "${NEBIUS_JOB_NAME:-}" ] || delete_cmd="NEBIUS_JOB_NAME=$(printf %q "$name") $delete_cmd"
ssh_opts=(-i "$ssh_key" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o StrictHostKeyChecking=accept-new)

for tool in nebius ssh jq; do
  command -v "$tool" >/dev/null || { echo "This script needs $tool." >&2; exit 1; }
done
project=$(nebius config get parent-id)

# Show the error $1, and the command that deletes the job. Then stop.
fail() {
  echo "$1" >&2
  echo "The job costs money until you delete it. Delete it with: $delete_cmd" >&2
  exit 1
}

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

# Print the job JSON, or nothing when no job has the name.
job_json() {
  local errors
  errors=$(mktemp)
  if ! nebius ai job get-by-name --parent-id "$project" --name "$name" --format json 2>"$errors"; then
    if ! grep -q 'code = NotFound' "$errors"; then
      cat "$errors" >&2
      rm -f "$errors"
      return 1
    fi
  fi
  rm -f "$errors"
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

create() {
  local pub created id state ip json create_log create_err deadline
  pub=$(public_key) || exit 1
  json=$(job_json) || exit 1
  created=false
  state=$(jq -r '.status.state // empty' <<<"${json:-null}")
  id=$(jq -r '.metadata.id // empty' <<<"${json:-null}")
  if [ -n "$state" ] && [ "$state" != RUNNING ] && [ "$state" != STARTING ] && [ "$state" != PROVISIONING ] &&
    [ "$state" != IMAGE_PULLING ]; then
    echo "The job $name exists, and its state is $state. Delete it with: $delete_cmd" >&2
    exit 1
  elif [ -n "$state" ]; then
    echo "The job $name exists. Waiting for it."
  else
    create_log=$(mktemp)
    create_err=$(mktemp)
    echo "Creating the job $name: $platform $preset, disk $disk_size, timeout $timeout."
    echo "If you stop this command, delete the job with: $delete_cmd"
    if ! nebius ai job create --parent-id "$project" --name "$name" --image ubuntu:24.04 \
      --container-command sleep --args infinity --timeout "$timeout" --platform "$platform" \
      --preset "$preset" --disk-size "$disk_size" --ssh-key "$pub" --async --format json \
      >"$create_log" 2>"$create_err"; then
      cat "$create_err" >&2
      id=$(job_id_from "$create_log")
      [ -n "$id" ] || id=$(job_id_from "$create_err")
      if [ -n "$id" ]; then
        echo "Nebius did not start the job. Deleting what it created." >&2
        nebius ai job delete "$id" >/dev/null || fail "Nebius could not delete the job."
      fi
      rm -f "$create_log" "$create_err"
      exit 1
    fi
    id=$(job_id_from "$create_log")
    [ -n "$id" ] || id=$(job_id_from "$create_err")
    rm -f "$create_log" "$create_err"
    [ -n "$id" ] || fail "Nebius did not return a job id."
    created=true
  fi

  # Serverless retries a missing GPU for 30 minutes, then the job goes to ERROR.
  deadline=$((SECONDS + 1800))
  ip=""
  while :; do
    json=$(nebius ai job get "$id" --format json)
    state=$(jq -r '.status.state // empty' <<<"$json")
    case "$state" in
      FAILED|ERROR|CANCELLED)
        echo "Nebius did not start the job ($state). Deleting what it created." >&2
        nebius ai job logs "$id" >&2 || true
        nebius ai job delete "$id" >/dev/null || echo "Nebius could not delete the job $id." >&2
        exit 1 ;;
    esac
    ip=$(jq -r '.status.instances[0].public_ip // empty' <<<"$json")
    [ "$state" = RUNNING ] && [ -n "$ip" ] && break
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "The job got no public address in 1800 s. Deleting it." >&2
      nebius ai job delete "$id" >/dev/null || echo "Nebius could not delete the job $id." >&2
      exit 1
    fi
    sleep 10
  done
  # Nebius can give a new job an address it used before. A known_hosts line for it is from another machine.
  if [ "$created" = true ] && [ -f "$HOME/.ssh/known_hosts" ]; then
    ssh-keygen -R "$ip" >/dev/null 2>&1 || fail "ssh-keygen could not remove $ip from ~/.ssh/known_hosts."
  fi
  ssh_deadline=$((SECONDS + 180))
  while :; do
    ssh "${ssh_opts[@]}" "nebius@$ip" true >/dev/null 2>&1 && break
    [ "$SECONDS" -lt "$ssh_deadline" ] || fail "No ssh to $ip in 180 s."
    sleep 5
  done
  # The account is not in the docker group until this runs. The next ssh sees the group.
  # docker/test_policy_image.sh calls docker, so sudo docker is not enough for that script.
  if ! ssh "${ssh_opts[@]}" "nebius@$ip" sudo -n usermod -aG docker nebius; then
    echo "Could not add nebius to the docker group. Use sudo docker on the host." >&2
    echo "docker/test_policy_image.sh calls docker, and it will fail until the group contains nebius." >&2
  fi
  echo "The job is ready. Connect to it with:"
  echo "  $ssh_cmd nebius@$ip"
  echo "The job costs money until you delete it, also when it does nothing. Delete it with:"
  echo "  $delete_cmd"
}

delete() {
  local json id
  json=$(job_json) || exit 1
  id=$(jq -r '.metadata.id // empty' <<<"${json:-null}")
  if [ -z "$id" ]; then
    echo "No job has the name $name."
    return
  fi
  nebius ai job delete "$id" >/dev/null
  echo "Deleted the job $name and its disk."
}

case "${1:-}" in
  create) create ;;
  delete) delete ;;
  *) echo "usage: $0 create|delete" >&2; exit 2 ;;
esac
