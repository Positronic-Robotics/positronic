"""Served functions: a call starts the work off the caller's thread and hands back a pimm ``Answer``."""

import contextvars
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any

import pimm

# Calling one starts the work and returns its ``Answer`` at once, never waiting.
Fn = Callable[..., pimm.calls.Answer[Any]]


class _PoolAnswer(pimm.calls.Answer[Any]):
    def __init__(self, call: Future[Any]):
        self._call = call

    def done(self) -> bool:
        return self._call.done()

    def result(self) -> Any:
        if not self._call.done():
            raise pimm.NoValueException('The call is not answered yet')
        return self._call.result()


class Executor:
    """Serves a set of functions on a worker thread of its own, one call at a time.

    A call runs under a copy of the context it was made in, so telemetry recorded inside it anchors where
    it was asked for.
    """

    def __init__(self, functions: Mapping[str, Callable[..., Any]]):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='policy-fn')
        self._fns: Mapping[str, Fn] = {name: partial(self._start, fn) for name, fn in functions.items()}

    @property
    def fns(self) -> Mapping[str, Fn]:
        return self._fns

    def _start(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> pimm.calls.Answer[Any]:
        context = contextvars.copy_context()
        return _PoolAnswer(self._pool.submit(context.run, fn, *args, **kwargs))

    def close(self) -> None:
        """Drop the queued calls and wait out the one in flight, which may still hold its caller's resources.
        A call made after close raises."""
        self._pool.shutdown(wait=True, cancel_futures=True)
