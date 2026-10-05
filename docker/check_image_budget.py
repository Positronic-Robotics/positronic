#!/usr/bin/env python3
"""Check a local image against the platform's three image budgets: compressed size, unpacked size, files.

Usage (from the repository root)
  uv run --package positronic-platform-client --no-dev docker/check_image_budget.py positro/<name>:local

Needs `pigz`. The compressed size is an upper estimate: this compresses at gzip level 1, and a push at the
default level. Exits 1 when the image is over a budget.
"""

import math
import subprocess
import sys
import tarfile
import threading
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


def unpacked_counts(saved: tarfile.TarFile) -> tuple[int, int]:
    """Sum the unpacked bytes and the entries of every layer in a ``docker save`` archive.

    Each entry counts at its size rounded up to a whole block, and at least one block. A blob that is not a
    tar archive is the config or a manifest, and counts nothing.
    """
    unpacked = entries = 0
    for member in saved:
        blob = saved.extractfile(member) if member.isfile() else None
        if blob is None:
            continue
        try:
            layer = tarfile.open(fileobj=blob, mode='r|*')
        except tarfile.ReadError:
            continue
        for entry in layer:
            unpacked += max(1, math.ceil(entry.size / IMAGE_BLOCK_BYTES)) * IMAGE_BLOCK_BYTES
            entries += 1
    return unpacked, entries


def main(image: str) -> int:
    save = subprocess.Popen(['docker', 'save', image], stdout=subprocess.PIPE)
    pigz = subprocess.Popen(['pigz', '-1'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert save.stdout is not None and pigz.stdin is not None and pigz.stdout is not None
    compressed = [0]
    counter = threading.Thread(target=_count_bytes, args=(pigz.stdout, compressed))
    counter.start()

    reader = _CopyingReader(save.stdout, pigz.stdin)
    with tarfile.open(fileobj=reader, mode='r|') as saved:
        unpacked, entries = unpacked_counts(saved)
    # The archive's end-of-file blocks are past the last member, and the compressed count includes them.
    while reader.read(1 << 20):
        pass
    pigz.stdin.close()
    counter.join()
    if save.wait() or pigz.wait():
        print(f'docker save exited {save.returncode}, pigz exited {pigz.returncode}', file=sys.stderr)
        return 1

    checks = [
        ('compressed bytes, upper estimate', compressed[0], COMPRESSED_IMAGE_BYTES),
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
