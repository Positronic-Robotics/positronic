"""A stand-in for BFL's interpreter running `backend.py`. It answers as BFL's RoboLab server answers.

It writes its arguments, its pid and a summary of every request to a log, one JSON object per line.

Usage
  python bfl_stand_in.py <log> <serve|refuse> -P backend.py --checkpoint <checkpoint> --port <port> ...
"""

import enum
import json
import os
import sys
from typing import Any

import numpy as np
from positronic_model_server.serialization import deserialize, serialize
from websockets.sync.server import ServerConnection, serve

# The chunk that answers every request: 32 actions of seven joint positions and a gripper.
CHUNK = np.arange(32 * 8, dtype=np.float32).reshape(32, 8) / 1000
# What BFL's server sends before it closes the connection on an exception.
REFUSAL = 'Traceback (most recent call last): the policy refused the request'


class Mode(enum.Enum):
    SERVE = 'serve'
    REFUSE = 'refuse'


def _summary(value: Any) -> Any:
    """The shape of an image, and the value of anything else."""
    if isinstance(value, np.ndarray):
        return list(value.shape) if value.ndim == 3 else value.tolist()
    return value


def main() -> None:
    log_path, mode_token, *argv = sys.argv[1:]
    mode = Mode(mode_token)

    def record(entry: dict[str, Any]) -> None:
        with open(log_path, 'a') as log:
            log.write(json.dumps(entry) + '\n')

    def handle(connection: ServerConnection) -> None:
        connection.send(serialize({}))
        for message in connection:
            record({'request': {key: _summary(value) for key, value in deserialize(message).items()}})
            if mode is Mode.REFUSE:
                connection.send(REFUSAL)
                connection.close(1011)
                return
            connection.send(serialize({'action': CHUNK}))

    record({'argv': argv, 'pid': os.getpid()})
    port = int(argv[argv.index('--port') + 1])
    with serve(handle, '127.0.0.1', port, compression=None, max_size=None) as backend:
        backend.serve_forever()


if __name__ == '__main__':
    main()
