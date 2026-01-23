"""Shared utilities for inference servers.

This module provides common functionality for servers that need to wait for
subprocess-based model loading with periodic status updates to prevent
WebSocket keepalive timeouts.

Limitations:
- These utilities only work for subprocess-based loading where the loading
  happens out-of-process, allowing the async loop to remain responsive.
- For in-process synchronous loading (e.g., basic_server, lerobot_server),
  blocking calls freeze the async loop, preventing periodic status updates.
  Those servers must assume fast (<20s) loading to avoid keepalive timeouts.
"""

import asyncio
import logging
import time
from collections.abc import Callable

from fastapi import WebSocket

from positronic.utils.serialization import serialise

logger = logging.getLogger(__name__)


async def monitor_async_task(
    task: asyncio.Task, description: str, websocket: WebSocket | None = None, update_interval: float = 5.0
) -> None:
    """Monitor an async task and send periodic status updates.

    Args:
        task: The async task to monitor.
        description: Description to include in status messages.
        websocket: Optional WebSocket to send status updates to.
        update_interval: Seconds between status update messages.
    """
    if websocket is None:
        await task
        return

    start_time = time.time()
    last_update_time = start_time

    while not task.done():
        elapsed = int(time.time() - start_time)

        # Send periodic status updates
        if time.time() - last_update_time >= update_interval:
            await websocket.send_bytes(
                serialise({'status': 'loading', 'message': f'{description}... ({elapsed}s elapsed)'})
            )
            last_update_time = time.time()

        await asyncio.sleep(1.0)

    # Get result (will raise if task failed)
    await task


async def _poll_subprocess_ready(
    check_ready: Callable[[], bool],
    check_crashed: Callable[[], tuple[bool, int | None]],
    description: str,
    max_wait: float = 300.0,
    ready_check_timeout: float = 2.0,
) -> None:
    """Poll a subprocess until it's ready (internal, no status updates).

    The check_ready function is called with a timeout to prevent it from blocking
    the async loop for too long.

    Args:
        check_ready: Function that returns True when subprocess is ready.
                    Called in a thread pool with ready_check_timeout.
        check_crashed: Function that returns (crashed: bool, exit_code: int | None).
        description: Human-readable description of the subprocess (e.g., "OpenPI subprocess").
        max_wait: Maximum time in seconds to wait before timing out.
        ready_check_timeout: Timeout for each check_ready call (prevents blocking).

    Raises:
        RuntimeError: If subprocess crashes or doesn't become ready within max_wait.
    """
    start_time = time.time()

    while time.time() - start_time < max_wait:
        elapsed = int(time.time() - start_time)

        # Check if subprocess crashed
        crashed, exit_code = check_crashed()
        if crashed:
            raise RuntimeError(f'{description} exited with code {exit_code}')

        # Check if subprocess is ready (with timeout to avoid blocking the async loop)
        try:
            ready = await asyncio.wait_for(asyncio.to_thread(check_ready), timeout=ready_check_timeout)
            if ready:
                logger.info(f'{description} ready after {elapsed}s')
                return
        except TimeoutError:
            # Check timed out, subprocess not ready yet
            pass

        await asyncio.sleep(1.0)

    raise RuntimeError(f'{description} did not become ready within {max_wait}s')


async def wait_for_subprocess_ready(
    check_ready: Callable[[], bool],
    check_crashed: Callable[[], tuple[bool, int | None]],
    description: str,
    websocket: WebSocket | None = None,
    max_wait: float = 300.0,
    update_interval: float = 5.0,
    ready_check_timeout: float = 2.0,
) -> None:
    """Wait for a subprocess to become ready, with optional periodic status updates.

    This function polls a subprocess until it's ready, optionally sending periodic
    WebSocket status updates to prevent keepalive timeouts during long startups.

    The check_ready function is called with a timeout to prevent it from blocking
    the async loop for too long (which would prevent status updates from being sent).

    Args:
        check_ready: Function that returns True when subprocess is ready.
                    Called in a thread pool with ready_check_timeout.
        check_crashed: Function that returns (crashed: bool, exit_code: int | None).
        description: Human-readable description of the subprocess (e.g., "OpenPI subprocess").
        websocket: Optional WebSocket to send status updates to. If None, runs silently.
        max_wait: Maximum time in seconds to wait before timing out.
        update_interval: Seconds between status update messages (if websocket provided).
        ready_check_timeout: Timeout for each check_ready call (prevents blocking).

    Raises:
        RuntimeError: If subprocess crashes or doesn't become ready within max_wait.

    Example:
        >>> await wait_for_subprocess_ready(
        ...     check_ready=lambda: client.ping(),
        ...     check_crashed=lambda: (process.poll() is not None, process.returncode),
        ...     description='GR00T subprocess',
        ...     websocket=websocket,
        ...     max_wait=120.0,
        ... )
    """
    task = asyncio.create_task(
        _poll_subprocess_ready(check_ready, check_crashed, description, max_wait, ready_check_timeout)
    )
    await monitor_async_task(
        task, description=f'Starting {description}', websocket=websocket, update_interval=update_interval
    )
