import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def socket_path():
    # A Unix socket path has a short length limit, so it does not go under the long pytest directory.
    with tempfile.TemporaryDirectory(prefix='teleop-') as directory:
        yield Path(directory) / 'teleop.sock'
