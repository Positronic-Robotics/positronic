#!/usr/bin/env bash
# Settings shared by every script in this directory. Sourced, never executed.
#
# What belongs here: a value the scripts must agree on, because they address the same cloud
# resource from different sides — the project a job is created in and an endpoint is looked up in,
# the secret `serve.sh` injects and `e2e.sh` reads back.
#
# What does not: a value each script sets for itself. NEBIUS_PRESET is one GPU for training and
# serving but a smaller one for eval; NEBIUS_JOB_TIMEOUT is 4h for a conversion and 24h for a run.
# Those live beside the script that means them, so hoisting one script's default never becomes
# another's.

# Nebius project and VPC subnet every job and endpoint is created in. The defaults are
# Positronic-internal; external users must override both.
PARENT_ID="${NEBIUS_PARENT_ID:-project-e00f38wexevrr52b8j}"
SUBNET_ID="${NEBIUS_SUBNET_ID:-vpcsubnet-e00pk1j1x6hjmr4m92}"

# Shared filesystem (RWX) holding the uv / HF / openpi caches across cold starts. An ID, not a name
# — `--volume` rejects names. pos3's cache stays on local disk (~/.cache/positronic/s3).
CACHE_FS="${NEBIUS_CACHE_FS:-computefilesystem-e00f6jyfr5wkawyrab}"

# Docker image tag every job and endpoint pulls. `make push-*` only updates `:latest` under CI;
# locally it pushes `:<branch>`/`:<sha>`. To run a branch build: `make push-<x> IMAGE_TAG=<branch>`
# then `NEBIUS_IMAGE_TAG=<branch>`.
IMAGE_TAG="${NEBIUS_IMAGE_TAG:-latest}"

# The bearer token gating served endpoints: a MysteryBox secret name, and the payload key inside it.
# `serve.sh` and `eval.sh` inject it under that same key as the container's env var, which is where
# the server and `positronic.cfg.policy.authed_remote` read it; `e2e.sh` reads the value back to
# authenticate its smoke check. Nebius' own `--token-secret` requires the key to be AUTH_TOKEN.
AUTH_TOKEN_SECRET="${NEBIUS_AUTH_TOKEN_SECRET:-positronic-serverless-inference-token}"
AUTH_TOKEN_KEY=AUTH_TOKEN

# S3 credentials and endpoint for pos3, as `nebius ai job|endpoint create` flags. Expand into a
# create call with "${S3_ENV_FLAGS[@]}".
S3_ENV_FLAGS=(
  --env-secret AWS_ACCESS_KEY_ID=positronic-serverless-aws-access-key-id
  --env-secret AWS_SECRET_ACCESS_KEY=positronic-serverless-aws-secret-access-key
  --env AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud:443
  --env AWS_DEFAULT_REGION=eu-north1
)
