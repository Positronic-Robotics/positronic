"""The in-process runtime: a call starts the work off the caller's thread and hands back an ``Answer``."""

import concurrent.futures
import contextvars
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any

from positronic.policy.base import Answer, Fn, NotAnswered, Runtime


class Executor(Runtime):
    """Serves a set of functions on worker threads of its own, ``max_workers`` calls at a time.

    A call runs under a copy of the context it was made in, so telemetry recorded inside it anchors where
    it was asked for.
    """

    class _Answer(Answer):
        def __init__(self, call: Future[Any]):
            self._call = call

        def done(self) -> bool:
            return self._call.done()

        def result(self) -> Any:
            if not self._call.done():
                raise NotAnswered('The call is not answered yet')
            return self._call.result()

    def __init__(self, functions: Mapping[str, Callable[..., Any]], *, max_workers: int = 1):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='policy-fn')
        self._fns: Mapping[str, Fn] = {name: partial(self._start, fn) for name, fn in functions.items()}
        # Every call made and not answered yet, read from the caller's thread while the workers answer. The
        # lock is reentrant because a call that answers before it is registered runs ``_answered`` inline.
        self._pending: set[Future[Any]] = set()
        self._lock = threading.RLock()

    @property
    def fns(self) -> Mapping[str, Fn]:
        return self._fns

    @property
    def in_flight(self) -> bool:
        """Whether any call is still to answer."""
        with self._lock:
            return bool(self._pending)

    def wait(self, timeout: float | None = None) -> None:
        """Block until every call made so far has answered, or until ``timeout`` seconds pass."""
        with self._lock:
            pending = set(self._pending)
        concurrent.futures.wait(pending, timeout=timeout)

    def _start(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Answer:
        context = contextvars.copy_context()
        call = self._pool.submit(context.run, fn, *args, **kwargs)
        with self._lock:
            self._pending.add(call)
            call.add_done_callback(self._answered)
        return self._Answer(call)

    def _answered(self, call: Future[Any]) -> None:
        with self._lock:
            self._pending.discard(call)

    def close(self) -> None:
        """Drop the queued calls and wait out those in flight, which may still hold their caller's resources.
        A call made after close raises."""
        self._pool.shutdown(wait=True, cancel_futures=True)
