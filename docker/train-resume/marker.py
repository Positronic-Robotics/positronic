"""Read or write the final marker of a training run in S3.

Usage (in a training image)
  /opt/train/venv/bin/python /opt/train/marker.py exists   # exit 0: it exists, 1: S3 answers not found, 2: other error
  /opt/train/venv/bin/python /opt/train/marker.py put

Environment: FINAL_MARKER (s3://<bucket>/<key>), S3_ENDPOINT, S3_REGION. The key is /data/keys/s3.json, with
`access_key_id` and `secret_access_key`.
"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

KEY_FILE = Path('/data/keys/s3.json')


def main() -> int:
    bucket, _, key = os.environ['FINAL_MARKER'].removeprefix('s3://').partition('/')
    creds = json.loads(KEY_FILE.read_text())
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ.get('S3_ENDPOINT', 'https://storage.eu-north1.nebius.cloud'),
        region_name=os.environ.get('S3_REGION', 'eu-north1'),
        aws_access_key_id=creds['access_key_id'],
        aws_secret_access_key=creds['secret_access_key'],
    )
    if sys.argv[1:] == ['put']:
        s3.put_object(Bucket=bucket, Key=key, Body=f'done {datetime.now(UTC).isoformat()}\n'.encode())
        return 0
    if sys.argv[1:] != ['exists']:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response['Error']['Code'] in ('404', 'NoSuchKey', 'NotFound'):
            return 1
        print(f'final marker lookup failed: {e}', file=sys.stderr)
        return 2
    except BotoCoreError as e:
        print(f'final marker lookup failed: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
