"""A stand-in for ABC's interpreter running `backend.py`. It answers as ABC's policy server answers.

It writes its arguments, its pid and a summary of every request to a log, one JSON object per line.

Usage
  python abc_stand_in.py <log> <serve|refuse> -P backend.py --policy.checkpoint-path=<path> \
      --port=<port> ...
"""

import enum
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from positronic_model_server.serialization import deserialize, serialize
from websockets.sync.server import ServerConnection, serve

# The chunk that answers every request: 30 actions of 14 values, per arm 6 joints then the gripper width.
CHUNK = np.arange(30 * 14, dtype=np.float32).reshape(30, 14) / 1000
# What ABC's server sends before it closes the connection on an exception.
REFUSAL = 'Traceback (most recent call last): the policy refused the request'


class Mode(enum.Enum):
    SERVE = 'serve'
    REFUSE = 'refuse'


def _summary(value: Any) -> Any:
    """The shape and the mean value of an image, the shape and the sum of the noise, and the value of anything else."""
    if isinstance(value, dict):
        return {key: _summary(item) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'mean': float(value.mean())}
    if isinstance(value, np.ndarray) and value.ndim == 2:
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sum': float(value.sum())}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    log_token, mode_token, *argv = sys.argv[1:]
    log_path = Path(log_token)
    mode = Mode(mode_token)

    def record(entry: dict[str, Any]) -> None:
        with log_path.open('a') as log:
            log.write(json.dumps(entry) + '\n')

    def handle(connection: ServerConnection) -> None:
        connection.send(serialize({}))
        for message in connection:
            record({'request': _summary(deserialize(message))})
            if mode is Mode.REFUSE:
                connection.send(REFUSAL)
                connection.close(1011)
                return
            connection.send(serialize({'actions': CHUNK}))

    record({'argv': argv, 'pid': os.getpid()})
    port = int(next(arg for arg in argv if arg.startswith('--port=')).removeprefix('--port='))
    with serve(handle, '127.0.0.1', port, compression=None, max_size=None) as backend:
        backend.serve_forever()


if __name__ == '__main__':
    main()
