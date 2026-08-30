"""Sync helpers for ``ModelSource.load`` implementations.

``load`` runs in a worker thread while the connected client sits in the websocket handshake with a
30s per-message timeout; these helpers pump ``on_progress`` every few seconds so long downloads,
subprocess boots and first inferences keep that handshake alive.
"""

import logging
import threading
import time
from collections.abc import Callable
from functools import partial
from typing import Any

from positronic.policy import INFER, Policy

logger = logging.getLogger(__name__)


def run_with_progress(fn: Callable[[], Any], description: str, on_progress: Callable[[str], None] | None) -> Any:
    """Run blocking ``fn``, reporting ``description`` with elapsed time every few seconds."""
    if on_progress is None:
        return fn()
    done = threading.Event()
    start = time.monotonic()

    def tick():
        while not done.wait(5.0):
            on_progress(f'{description}... ({time.monotonic() - start:.0f}s elapsed)')

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        return fn()
    finally:
        done.set()
        ticker.join()


def warmup(policy: Policy, obs: dict[str, Any], on_progress: Callable[[str], None] | None = None) -> None:
    """Run one inference through ``policy``, so a backend's first-call cost is paid before it serves.

    ``obs`` has to be an observation the loaded backend accepts.
    """
    with policy.episode() as fns:
        run_with_progress(partial(fns[INFER], obs), 'Running warmup inference', on_progress)


def wait_for_subprocess_ready(
    check_ready: Callable[[], bool],
    check_crashed: Callable[[], tuple[bool, int | None]],
    description: str,
    on_progress: Callable[[str], None] | None = None,
    max_wait: float = 300.0,
    update_interval: float = 5.0,
) -> None:
    """Poll a subprocess until ready, reporting progress so long boots don't starve the client handshake."""
    start = time.monotonic()
    last_update = start
    while time.monotonic() - start < max_wait:
        crashed, exit_code = check_crashed()
        if crashed:
            raise RuntimeError(f'{description} exited with code {exit_code}')
        if check_ready():
            logger.info(f'{description} ready after {time.monotonic() - start:.0f}s')
            return
        if on_progress is not None and time.monotonic() - last_update >= update_interval:
            on_progress(f'Starting {description}... ({time.monotonic() - start:.0f}s elapsed)')
            last_update = time.monotonic()
        time.sleep(1.0)
    raise RuntimeError(f'{description} did not become ready within {max_wait}s')
