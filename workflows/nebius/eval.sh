#!/usr/bin/env bash
# Submit a simulator eval (`positronic eval run`) as a Nebius Serverless Job.
#
# The job pulls `positro/robolab` and boots the benchmark's env server inside the
# job container (Isaac Sim for RoboLab — needs an RTX-class GPU, so the platform
# is L40S, not H100). The policy is remote: serve it first (e.g. `serve.sh openpi
# ...`) and point `--policy.host` at the endpoint IP.
#
# Cache: unlike the other scripts, the shared filesystem mounts at /root/.cache,
# not /cache. The env-server launcher keeps its pinned checkout and venv at fixed
# $HOME-relative paths (~/.cache/positronic/robolab/src), and they dominate
# cold-start (~25 GB with the Isaac stack). One mount persists checkout, venv and
# uv cache together; uv's default cache dir (~/.cache/uv) lands on the same <fs>/uv
# subdir the UV_CACHE_DIR=/cache/uv jobs use, so the wheel cache stays shared.
# The pos3 stays-on-local-disk rule loses nothing here: eval jobs stream no S3
# datasets, and an s3:// --output_dir is a write-through pos3.sync at episode end.
#
# Concurrency: two cold jobs racing to populate the same checkout/venv can
# corrupt it — seed the cache with one run before fanning out (same rule as the
# shared uv cache, see e2e.sh).
#
# Hardcoded: MysteryBox secret names, S3 endpoint URL. Override-able via env:
# NEBIUS_PARENT_ID, NEBIUS_SUBNET_ID, NEBIUS_PLATFORM, NEBIUS_PRESET,
# NEBIUS_IMAGE_REPO, NEBIUS_IMAGE_TAG, NEBIUS_CACHE_FS, NEBIUS_JOB_TIMEOUT.
#
# NEBIUS_IMAGE_REPO defaults to the Docker Hub `positro/robolab`; set it to an
# in-region registry path (e.g. cr.eu-north1.nebius.cloud/<project-id>/robolab)
# to pull the image from the Nebius Container Registry instead.

set -euo pipefail

PARENT_ID="${NEBIUS_PARENT_ID:-project-e00f38wexevrr52b8j}"
SUBNET_ID="${NEBIUS_SUBNET_ID:-vpcsubnet-e00pk1j1x6hjmr4m92}"
CACHE_FS="${NEBIUS_CACHE_FS:-computefilesystem-e00f6jyfr5wkawyrab}"
IMAGE_REPO="${NEBIUS_IMAGE_REPO:-positro/robolab}"
IMAGE_TAG="${NEBIUS_IMAGE_TAG:-latest}"
# RTX-class platform: Isaac Sim's RTX renderer needs RT cores. gpu-l40s-d is the
# Intel-host L40S; gpu-l40s-a is the AMD-host variant with the same GPU.
PLATFORM="${NEBIUS_PLATFORM:-gpu-l40s-d}"
PRESET="${NEBIUS_PRESET:-1gpu-16vcpu-96gb}"
JOB_TIMEOUT="${NEBIUS_JOB_TIMEOUT:-24h}"

if [ $# -lt 1 ]; then
  cat >&2 <<'EOF'
Usage: bash workflows/nebius/eval.sh [eval run args...]

Forwards all arguments to `positronic eval run`. Serve the policy first
(workflows/nebius/serve.sh) and pass its endpoint IP. Example:

  bash workflows/nebius/eval.sh \
    --eval=@positronic.cfg.eval.sim.robolab.banana_in_bowl \
    --eval.trial_count=10 \
    --policy=@positronic.cfg.policy.remote \
    --policy.host=<endpoint-ip> \
    --policy.resize=None \
    --output_dir=s3://<your-bucket>/evals/robolab_banana/
EOF
  exit 1
fi

JOB_NAME="robolab-eval-$(date +%Y%m%d-%H%M%S)"
EVAL_ARGS="run --python 3.13 positronic eval run $*"

# NEBIUS_PREEMPTIBLE=1 runs on a preemptible VM — markedly cheaper, and often the only L40S capacity available.
PREEMPTIBLE_FLAGS=()
[ -n "${NEBIUS_PREEMPTIBLE:-}" ] && PREEMPTIBLE_FLAGS+=(--preemptible)

nebius ai job create \
  --parent-id "$PARENT_ID" \
  --subnet-id "$SUBNET_ID" \
  --name "$JOB_NAME" \
  --image "${IMAGE_REPO}:${IMAGE_TAG}" \
  --container-command uv \
  --args "$EVAL_ARGS" \
  --platform "$PLATFORM" \
  --preset "$PRESET" \
  ${PREEMPTIBLE_FLAGS[@]+"${PREEMPTIBLE_FLAGS[@]}"} \
  --timeout "$JOB_TIMEOUT" \
  --working-dir /positronic \
  --volume "${CACHE_FS}:/root/.cache:rw" \
  --env PYTHONUNBUFFERED=1 \
  --env-secret AWS_ACCESS_KEY_ID=positronic-serverless-aws-access-key-id \
  --env-secret AWS_SECRET_ACCESS_KEY=positronic-serverless-aws-secret-access-key \
  --env AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud:443 \
  --env AWS_DEFAULT_REGION=eu-north1
