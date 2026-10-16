#!/usr/bin/env bash
# Create a Nebius VM to build and test a policy image on, and delete it.
#
# Usage
#   bash docker/nebius_vm.sh create     # create the VM, wait for it, and print the ssh command
#   bash docker/nebius_vm.sh delete     # delete the VM and its disk
#
# Needs the nebius CLI with a profile, and ssh. The VM is in the project of the profile. It boots the
# Nebius image `ubuntu24.04-cuda12`, which has Docker and the NVIDIA container toolkit. Its user
# `ubuntu` gets your public key, the `docker` group and uv.
#
# Settings, as environment variables:
#   NEBIUS_VM_NAME   name of the VM, default policy-image-vm
#   NEBIUS_PLATFORM  default gpu-l40s-a; cpu-d3 gives a VM with no GPU
#   NEBIUS_PRESET    default 1gpu-8vcpu-32gb; 8vcpu-32gb on cpu-d3
#   NEBIUS_DISK_GIB  size of the boot disk, default 200
#   NEBIUS_SSH_KEY   private key, default ~/.ssh/id_ed25519; the VM gets the public key at <key>.pub
set -euo pipefail

name="${NEBIUS_VM_NAME:-policy-image-vm}"
platform="${NEBIUS_PLATFORM:-gpu-l40s-a}"
preset="${NEBIUS_PRESET:-1gpu-8vcpu-32gb}"
disk_gib="${NEBIUS_DISK_GIB:-200}"
ssh_key="${NEBIUS_SSH_KEY:-$HOME/.ssh/id_ed25519}"
# The ssh command to print. Plain ssh finds only a key at a default path.
ssh_cmd=ssh
[ -z "${NEBIUS_SSH_KEY:-}" ] || ssh_cmd="ssh -i $(printf %q "$ssh_key")"
# The delete command to print. It must name the same VM.
delete_cmd="bash $(printf %q "$0") delete"
[ -z "${NEBIUS_VM_NAME:-}" ] || delete_cmd="NEBIUS_VM_NAME=$(printf %q "$name") $delete_cmd"

project=$(nebius config get parent-id)
errors=$(mktemp)
trap 'rm -f "$errors"' EXIT

# Print the field of the VM that the jsonpath $1 names. Print nothing when no VM has the name.
vm_field() {
  if ! nebius compute instance get-by-name --parent-id "$project" --name "$name" --format jsonpath="$1" \
    2>"$errors"; then
    grep -q 'code = NotFound' "$errors" || { cat "$errors" >&2; return 1; }
  fi
}

# Print the public address of the VM, or nothing. The jsonpath fails on a field that is not there, so it
# reads the whole object.
vm_address() {
  vm_field '{.status.network_interfaces[0].public_ip_address}' | sed -n 's/.*"address": *"\([0-9.]*\)\/.*/\1/p'
}

# Create the VM. When Nebius does not start it, delete it and stop.
create_vm() {
  subnet=$(nebius vpc subnet list --parent-id "$project" --format jsonpath='{.items[0].metadata.id}')
  # The image has the user `ubuntu` already, and cloud-init does not change the groups of a user that exists.
  user_data="#cloud-config
ssh_authorized_keys:
  - \"$(cat "$ssh_key.pub")\"
runcmd:
  - [usermod, -aG, docker, ubuntu]
  - [su, -l, ubuntu, -c, 'curl -LsSf -o /tmp/uv-install.sh https://astral.sh/uv/install.sh && sh /tmp/uv-install.sh']"
  # A label that only this run knows. After a failed create, it shows that the VM is from this run.
  run_label=$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')
  echo "Creating the VM $name: $platform $preset, $disk_gib GiB disk."
  if nebius compute instance create --parent-id "$project" --name "$name" --labels "nebius-vm-run=$run_label" \
    --resources-platform "$platform" --resources-preset "$preset" \
    --boot-disk-attach-mode read_write --boot-disk-managed-disk-name "$name-boot" \
    --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-cuda12 \
    --boot-disk-managed-disk-size-gibibytes "$disk_gib" --boot-disk-managed-disk-type network_ssd \
    --network-interfaces "[{\"name\":\"eth0\",\"subnet_id\":\"$subnet\",\"ip_address\":{},\"public_ip_address\":{}}]" \
    --cloud-init-user-data "$user_data" >/dev/null; then
    return
  fi
  # A VM that Nebius did not start stays in the project, stopped. A VM with the name but not the label is from
  # another run.
  metadata=$(vm_field '{.metadata}')
  if grep -q "$run_label" <<<"$metadata"; then
    echo "Nebius did not start the VM. Deleting what it created." >&2
    id=$(sed -n 's/.*"id":"\(computeinstance-[a-z0-9]*\)".*/\1/p' <<<"$metadata")
    nebius compute instance delete --id "$id" >/dev/null
  fi
  exit 1
}

create() {
  if [ ! -f "$ssh_key.pub" ]; then
    echo "No public key at $ssh_key.pub. Run ssh-keygen -t ed25519, or set NEBIUS_SSH_KEY." >&2
    exit 1
  fi
  # The ssh below cannot ask for a passphrase. Check the key before the VM costs money.
  if ! ssh-keygen -y -P "" -f "$ssh_key" >/dev/null 2>&1 &&
    ! ssh-add -L 2>/dev/null | grep -qF "$(cut -d' ' -f2 "$ssh_key.pub")"; then
    echo "ssh cannot use $ssh_key without a passphrase. Run: ssh-add $(printf %q "$ssh_key")" >&2
    exit 1
  fi
  state=$(vm_field '{.status.state}')
  if [ -n "$state" ] && [ "$state" != RUNNING ] && [ "$state" != STARTING ] && [ "$state" != CREATING ]; then
    echo "The VM $name exists, and its state is $state. Delete it with: $delete_cmd" >&2
    exit 1
  elif [ -n "$state" ]; then
    echo "The VM $name exists. Waiting for it."
  else
    create_vm
  fi

  deadline=$((SECONDS + 900))
  until ip=$(vm_address) && [ -n "$ip" ]; do
    [ "$SECONDS" -lt "$deadline" ] || { echo "The VM got no public address in 900 s." >&2; exit 1; }
    sleep 10
  done
  # ssh exits with 255 when it cannot connect, and `timeout` on the VM exits with 124. Any other failure comes
  # from cloud-init. The `timeout` runs on the VM, because macOS has no `timeout`.
  ssh_opts=(-i "$ssh_key" -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
  while :; do
    status=0
    ssh "${ssh_opts[@]}" "ubuntu@$ip" timeout $((deadline > SECONDS ? deadline - SECONDS : 1)) \
      cloud-init status --wait >/dev/null 2>"$errors" || status=$?
    [ "$status" = 255 ] || break
    [ "$SECONDS" -lt "$deadline" ] || { cat "$errors" >&2; echo "No ssh to $ip in 900 s." >&2; exit 1; }
    sleep 10
  done
  if [ "$status" = 124 ]; then
    echo "The VM setup did not finish in 900 s." >&2
    exit 1
  elif [ "$status" != 0 ]; then
    echo "The VM setup failed. Read the log with: $ssh_cmd ubuntu@$ip sudo cat /var/log/cloud-init-output.log" >&2
    exit 1
  fi
  echo "The VM is ready. Connect to it with:"
  echo "  $ssh_cmd ubuntu@$ip"
  echo "The VM costs money until you delete it, also when it does nothing. Delete it with:"
  echo "  $delete_cmd"
}

delete() {
  id=$(vm_field '{.metadata.id}')
  if [ -z "$id" ]; then
    echo "No VM has the name $name."
    return
  fi
  ip=$(vm_address)
  nebius compute instance delete --id "$id" >/dev/null
  # Nebius gives the address to other VMs later, and each new VM has a new host key.
  if [ -n "$ip" ] && [ -f "$HOME/.ssh/known_hosts" ]; then
    ssh-keygen -R "$ip" >/dev/null 2>&1
  fi
  echo "Deleted the VM $name and its disk."
}

case "${1:-}" in
  create) create ;;
  delete) delete ;;
  *) echo "usage: $0 create|delete" >&2; exit 2 ;;
esac
