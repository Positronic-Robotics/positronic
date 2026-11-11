#!/bin/sh
# Relaunch the run that /data/run holds, unless the run has written its final marker. When the launcher exits 0,
# write the marker, so a later boot launches nothing.
#
# Usage (train-resume.service runs it in the training image at each boot)
#   docker run --gpus all --env-file /data/run/run.env -v /data:/data <image> /opt/train/resume
#
# /data/run/run.env sets:
#   TRAIN_IMAGE    the image, pinned by digest (the unit reads it)
#   RUN            the run id
#   FINAL_MARKER   s3://<bucket>/<key> of the final marker
#   S3_ENDPOINT    the S3 endpoint of that bucket; S3_REGION its signing region
# /data/keys/s3.json holds `access_key_id` and `secret_access_key` for the marker's bucket.
# An executable /data/run/run.sh replaces the image's own launcher.
set -eu
: "${RUN:?run.env sets no RUN}"
: "${FINAL_MARKER:?run.env sets no FINAL_MARKER}"
mkdir -p /data/run/logs
marker() { /opt/train/venv/bin/python /opt/train/marker.py "$1"; }

# Only an existing marker stops the launch. A launch after the end resumes from the last checkpoint and stops
# at once, where a run left waiting for S3 holds its GPUs until its deadline.
status=0
marker exists || status=$?
if [ "$status" -eq 0 ]; then
    echo "run $RUN: $FINAL_MARKER exists, nothing to launch"
    exit 0
fi
echo "run $RUN: the final marker lookup exits $status (1: not found), launching"

launcher=/opt/train/launch
if [ -x /data/run/run.sh ]; then
    launcher=/data/run/run.sh
fi
# The launcher is a child, so the stop signal of the container reaches it through this trap.
"$launcher" &
pid=$!
trap 'kill -TERM "$pid" 2>/dev/null' TERM INT
status=0
wait "$pid" || status=$?
if [ "$status" -ne 0 ]; then
    echo "run $RUN: $launcher exited $status, no final marker"
    exit "$status"
fi
marker put
echo "run $RUN: $launcher exited 0, final marker written"
