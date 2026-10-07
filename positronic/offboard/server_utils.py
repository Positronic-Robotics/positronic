"""Sync helpers a model's build runs: a long download, a subprocess boot, a first inference.

Each logs how long it has run every few seconds, so a load that takes minutes shows in the server log.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from positronic.offboard.spec import Model

logger = logging.getLogger(__name__)

PROGRESS_LOG_INTERVAL_S = 5.0


def run_with_progress(fn: Callable[[], Any], description: str) -> Any:
    """Run blocking ``fn``, logging ``description`` with elapsed time every few seconds."""
    done = threading.Event()
    start = time.monotonic()

    def tick():
        while not done.wait(PROGRESS_LOG_INTERVAL_S):
            logger.info(f'{description}... ({time.monotonic() - start:.0f}s elapsed)')

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        return fn()
    finally:
        done.set()
        ticker.join()


def warmup(policy: Model, obs: dict[str, Any]) -> None:
    """Run one inference through ``policy``, so a backend's first-call cost is paid before it serves.

    ``obs`` has to be an observation the loaded backend accepts.
    """
    session_id = uuid4().hex
    try:
        run_with_progress(lambda: policy(obs, session_id=session_id), 'Running warmup inference')
    finally:
        policy.end_session(session_id)


def wait_for_subprocess_ready(
    check_ready: Callable[[], bool],
    check_crashed: Callable[[], tuple[bool, int | None]],
    description: str,
    max_wait: float = 300.0,
) -> None:
    """Poll a subprocess until ready, logging how long it has run."""
    start = time.monotonic()
    last_update = start
    while time.monotonic() - start < max_wait:
        crashed, exit_code = check_crashed()
        if crashed:
            raise RuntimeError(f'{description} exited with code {exit_code}')
        if check_ready():
            logger.info(f'{description} ready after {time.monotonic() - start:.0f}s')
            return
        if time.monotonic() - last_update >= PROGRESS_LOG_INTERVAL_S:
            logger.info(f'Starting {description}... ({time.monotonic() - start:.0f}s elapsed)')
            last_update = time.monotonic()
        time.sleep(1.0)
    raise RuntimeError(f'{description} did not become ready within {max_wait}s')
