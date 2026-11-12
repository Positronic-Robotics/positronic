"""A stand-in for the fork's interpreter running the author's openpi server. It answers as openpi's server answers.

It writes its arguments, its working directory, its pid and a summary of every request to a log, one JSON object
per line.

Usage
  python author_server_stand_in.py <log> <server script> --host 127.0.0.1 --port <port> ...
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.server import ServerConnection, serve

# The chunk that answers every request: 50 actions of 14 values, per arm 6 joints then the gripper width.
CHUNK = np.arange(50 * 14, dtype=np.float32).reshape(50, 14) / 1000


def _summary(value: Any) -> Any:
    """The shape, the dtype and the mean value of an image, and the value of anything else."""
    if isinstance(value, dict):
        return {key: _summary(item) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'mean': float(value.mean())}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    log_path = Path(sys.argv[1])
    argv = sys.argv[2:]

    def record(entry: dict[str, Any]) -> None:
        with log_path.open('a') as log:
            log.write(json.dumps(entry) + '\n')

    def handle(connection: ServerConnection) -> None:
        packer = msgpack_numpy.Packer()
        connection.send(packer.pack({}))
        for message in connection:
            record({'request': _summary(msgpack_numpy.unpackb(message))})
            connection.send(packer.pack({'actions': CHUNK}))

    record({'argv': argv, 'cwd': os.getcwd(), 'pid': os.getpid()})
    port = int(argv[argv.index('--port') + 1])
    with serve(handle, '127.0.0.1', port, compression=None, max_size=None) as backend:
        backend.serve_forever()


if __name__ == '__main__':
    main()
