#!/usr/bin/env python3
"""Download an ABC checkpoint and its metadata sidecar from ABC's data mirror, and check the checkpoint's sha256.

It runs in the weights stage of `Dockerfile.serve-abc-dit`, which has only the standard library.

Usage
  fetch_abc_checkpoint.py <url without suffix> <sha256 of the .pt> <output dir>
"""

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

# The mirror refuses Python's default agent; this is the agent of ABC's prepare.py.
USER_AGENT = 'abc-prepare/1.0'
CHUNK_BYTES = 16 << 20


def fetch(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request) as response, path.open('wb') as out:
        shutil.copyfileobj(response, out, CHUNK_BYTES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def main(base_url: str, expected_sha256: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = base_url.rsplit('/', 1)[-1]
    fetch(base_url + '.json', out / f'{name}.json')
    checkpoint = out / f'{name}.pt'
    fetch(base_url + '.pt', checkpoint)
    actual = sha256(checkpoint)
    if actual != expected_sha256:
        raise SystemExit(f'{checkpoint}: sha256 {actual}, expected {expected_sha256}')


if __name__ == '__main__':
    main(*sys.argv[1:])
