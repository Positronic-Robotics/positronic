#!/usr/bin/env python3
"""Check a local image against the platform's three image budgets: compressed size, unpacked size, files.

Usage (from the repository root)
  uv run --package positronic-platform-client --no-dev docker/check_image_budget.py positro/<name>:local

Needs `pigz`. The compressed size is an upper estimate: this compresses each layer at gzip level 1, and a push at
the default level. Exits 1 when the image is over a budget.
"""

import json
import math
import subprocess
import sys
import tarfile
import threading
from collections import Counter
from pathlib import PurePosixPath
from typing import IO

from platform_client.policy_container import (
    COMPRESSED_IMAGE_BYTES,
    IMAGE_BLOCK_BYTES,
    IMAGE_FILES,
    UNPACKED_IMAGE_BYTES,
)


class _CopyingReader:
    """Reads from ``source`` and writes every byte it reads to ``sink``."""

    def __init__(self, source: IO[bytes], sink: IO[bytes]):
        self._source = source
        self._sink = sink

    def read(self, size: int = -1) -> bytes:
        data = self._source.read(size)
        self._sink.write(data)
        return data


def _count_bytes(stream: IO[bytes], total: list[int]) -> None:
    while chunk := stream.read(1 << 20):
        total[0] += len(chunk)


def layer_counts(layer: IO[bytes]) -> tuple[int, int, int]:
    """The gzip level 1 size, the unpacked bytes and the entries of one layer archive.

    Each entry counts at its size rounded up to a whole block, and at least one block.
    """
    pigz = subprocess.Popen(['pigz', '-1'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert pigz.stdin is not None and pigz.stdout is not None
    compressed = [0]
    counter = threading.Thread(target=_count_bytes, args=(pigz.stdout, compressed))
    counter.start()

    reader = _CopyingReader(layer, pigz.stdin)
    unpacked = entries = 0
    with tarfile.open(fileobj=reader, mode='r|*') as archive:
        for entry in archive:
            unpacked += max(1, math.ceil(entry.size / IMAGE_BLOCK_BYTES)) * IMAGE_BLOCK_BYTES
            entries += 1
    # The archive's end-of-file blocks are past its last entry, and the compressed size includes them.
    while reader.read(1 << 20):
        pass
    pigz.stdin.close()
    counter.join()
    if pigz.wait():
        raise RuntimeError(f'pigz exited {pigz.returncode}')
    return compressed[0], unpacked, entries


def main(image: str) -> int:
    inspect = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, check=True)
    inspected = json.loads(inspect.stdout)
    config_digest = inspected[0]['Id'].removeprefix('sha256:')
    # The manifest lists a layer once per occurrence, and the archive stores it once.
    layers = Counter(diff_id.removeprefix('sha256:') for diff_id in inspected[0]['RootFS']['Layers'])

    compressed = unpacked = entries = 0
    found: set[str] = set()
    save = subprocess.Popen(['docker', 'save', image], stdout=subprocess.PIPE)
    assert save.stdout is not None
    with tarfile.open(fileobj=save.stdout, mode='r|') as saved:
        for member in saved:
            # `blobs/sha256/<digest>` in an OCI archive; `<digest>.tar` and `<digest>.json` in a Docker one.
            digest = PurePosixPath(member.name).name.split('.', 1)[0]
            if not member.isfile() or digest in found or (digest != config_digest and digest not in layers):
                continue
            found.add(digest)
            if digest == config_digest:
                compressed += member.size
                continue
            blob = saved.extractfile(member)
            assert blob is not None
            layer_compressed, layer_unpacked, layer_entries = layer_counts(blob)
            compressed += layer_compressed * layers[digest]
            unpacked += layer_unpacked * layers[digest]
            entries += layer_entries * layers[digest]
    if save.wait():
        print(f'docker save exited {save.returncode}', file=sys.stderr)
        return 1
    missing = (set(layers) | {config_digest}) - found
    if missing:
        print(f'docker save carried no blob for {sorted(missing)}', file=sys.stderr)
        return 1

    checks = [
        ('compressed bytes, upper estimate', compressed, COMPRESSED_IMAGE_BYTES),
        ('unpacked bytes', unpacked, UNPACKED_IMAGE_BYTES),
        ('files', entries, IMAGE_FILES),
    ]
    for label, value, budget in checks:
        print(f'{label}: {value:,} of {budget:,}')
    over = [label for label, value, budget in checks if value > budget]
    if over:
        print(f'over the budget: {", ".join(over)}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(f'usage: {sys.argv[0]} <image>')
    sys.exit(main(sys.argv[1]))
