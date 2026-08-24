"""Served functions: a call starts the work off the caller's thread and hands back an ``Answer``.

``Answer`` is the policy API's own, not pimm's: the two are alike but not interchangeable.
"""

import contextvars
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any


class NotAnswered(RuntimeError):
    """The call has not answered yet, and reading it never waits for one."""


class Answer(ABC):
    """The caller's handle on one call: ``done()`` once the function has answered, then ``result()``."""

    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def result(self) -> Any:
        """What the function returned, re-raising what it raised. ``NotAnswered`` before it has answered."""


# Calling one starts the work and returns its ``Answer`` at once, never waiting.
Fn = Callable[..., Answer]


class Executor:
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

    @property
    def fns(self) -> Mapping[str, Fn]:
        return self._fns

    def _start(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Answer:
        context = contextvars.copy_context()
        return self._Answer(self._pool.submit(context.run, fn, *args, **kwargs))

    def close(self) -> None:
        """Drop the queued calls and wait out those in flight, which may still hold their caller's resources.
        A call made after close raises."""
        self._pool.shutdown(wait=True, cancel_futures=True)
