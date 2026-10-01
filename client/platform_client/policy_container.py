"""The container the platform runs a submitted policy image in: its limits and its image budgets.

Each value is a published contract value. The platform runs every image under it, and
`docs/submit-a-policy-image.md` states it.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, NonNegativeInt

# The port the image serves its inference server on.
POLICY_PORT = 8000
# The one variable the platform sets in the container: the run's bearer token.
AUTH_TOKEN_ENV = 'AUTH_TOKEN'
# The deadline that VM boot, the image pull and the server's start share.
PROVISIONING_DEADLINE_S = 1800.0

# The GPU: one MIG slice of an H100, and the memory it carries.
MIG_PROFILE = '3g.40gb'
VRAM_MIB = 40448

# What the platform pulls and unpacks. `COMPRESSED_IMAGE_BYTES` bounds `ImageManifest.compressed_size`.
# The unpacked budgets count every entry of every layer: its size rounded up to a whole
# `IMAGE_BLOCK_BYTES` block, and at least one block, against `UNPACKED_IMAGE_BYTES`; one entry against
# `IMAGE_FILES`. All three live in an image store of `IMAGE_STORE_BYTES`.
COMPRESSED_IMAGE_BYTES = 30 * 10**9
UNPACKED_IMAGE_BYTES = 75 * 10**9
IMAGE_FILES = 4_500_000
IMAGE_BLOCK_BYTES = 4096
IMAGE_STORE_BYTES = 120 * 2**30

# The container's share of the box. Its swap is zero: the swap limit equals the memory limit.
MEMORY_BYTES = 150 * 2**30
CPUS = 14
PIDS = 4096
# The container log, one file.
LOG_BYTES = 200 * 2**20
# Reads and writes on the disk that holds the image store.
DISK_READ_BYTES_PER_S = 500 * 2**20
DISK_WRITE_BYTES_PER_S = 100 * 2**20
DISK_READ_IOPS = 2000
DISK_WRITE_IOPS = 1000
# What the container may send, policed on the host side of its link. A packet above the rate is dropped.
SEND_BITS_PER_S = 500 * 10**6
SEND_BURST_BYTES = 5 * 2**20


def docker_limit_flags(disk: Path) -> list[str]:
    """The `docker run` flags that hold a container to the limits above.

    `disk` is the block device that holds the image store, which the disk rates bound. The send rate is
    a policer on the host, and no `docker run` flag sets it.
    """
    return [
        '--cap-drop',
        'ALL',
        '--security-opt',
        'no-new-privileges',
        '--memory',
        str(MEMORY_BYTES),
        '--memory-swap',
        str(MEMORY_BYTES),
        '--cpus',
        str(CPUS),
        '--pids-limit',
        str(PIDS),
        '--log-opt',
        f'max-size={LOG_BYTES}',
        '--device-read-bps',
        f'{disk}:{DISK_READ_BYTES_PER_S}',
        '--device-write-bps',
        f'{disk}:{DISK_WRITE_BYTES_PER_S}',
        '--device-read-iops',
        f'{disk}:{DISK_READ_IOPS}',
        '--device-write-iops',
        f'{disk}:{DISK_WRITE_IOPS}',
    ]


class Descriptor(BaseModel):
    """One blob an image manifest names, by the size the registry stores it at."""

    size: NonNegativeInt


class ImageManifest(BaseModel):
    """An image manifest for one platform, OCI or Docker v2: its config blob and its layers."""

    config: Descriptor
    layers: list[Descriptor]

    @property
    def compressed_size(self) -> int:
        """What a pull transfers: the config and every layer, as the registry stores them."""
        return self.config.size + sum(layer.size for layer in self.layers)
