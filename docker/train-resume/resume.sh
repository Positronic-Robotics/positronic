#!/bin/sh
# Relaunch the run that /data/run holds, unless the run has written its final marker.
#
# Usage (train-resume.service runs it in the training image at each boot)
#   docker run --gpus all --env-file /data/run/run.env -v /data:/data <image> /opt/train/resume
#
# /data/run/run.env sets:
#   TRAIN_IMAGE    the image, pinned by digest (the unit reads it)
#   RUN            the run id
#   FINAL_MARKER   s3://<bucket>/<key> that the run writes when it is done
#   S3_ENDPOINT    the S3 endpoint of that bucket; S3_REGION its signing region
# /data/keys/s3.json holds `access_key_id` and `secret_access_key` for the marker's bucket.
# An executable /data/run/run.sh replaces the image's own launcher.
set -eu
: "${RUN:?run.env sets no RUN}"
: "${FINAL_MARKER:?run.env sets no FINAL_MARKER}"
mkdir -p /data/run/logs

# Only a 200 stops the launch. A launch after the end resumes from the last checkpoint and stops at once.
bucket_key="${FINAL_MARKER#s3://}"
code=$(curl -s -o /dev/null -w '%{http_code}' -I --max-time 30 \
    --aws-sigv4 "aws:amz:${S3_REGION:-eu-north1}:s3" \
    --user "$(jq -r .access_key_id /data/keys/s3.json):$(jq -r .secret_access_key /data/keys/s3.json)" \
    "${S3_ENDPOINT:-https://storage.eu-north1.nebius.cloud}/$bucket_key" || true)
if [ "$code" = 200 ]; then
    echo "run $RUN: $FINAL_MARKER exists, nothing to launch"
    exit 0
fi
echo "run $RUN: the final marker answers $code, launching"

if [ -x /data/run/run.sh ]; then
    exec /data/run/run.sh
fi
exec /opt/train/launch
