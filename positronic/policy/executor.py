"""The in-process runtime: a call starts the work on a worker thread and returns an ``Answer``."""

import concurrent.futures
import contextvars
import logging
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
        def __init__(self, name: str, call: Future[Any], read: Callable[['Executor._Answer'], None]):
            self.name = name
            self.call = call
            self._read = read

        def done(self) -> bool:
            return self.call.done()

        def result(self) -> Any:
            if not self.call.done():
                raise NotAnswered('The call is not answered yet')
            self._read(self)
            return self.call.result()

        def failure(self) -> BaseException | None:
            """What the call raised, once it has answered. ``None`` when it returned a value or was cancelled."""
            return None if self.call.cancelled() else self.call.exception()

    def __init__(self, functions: Mapping[str, Callable[..., Any]], *, max_workers: int = 1):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='policy-fn')
        self._fns: Mapping[str, Fn] = {name: partial(self._start, name, fn) for name, fn in functions.items()}
        # Every answer that no caller has read. A call that has still to answer is one of these, so
        # ``in_flight`` and ``owes_an_answer`` read this one set.
        self._unread: set[Executor._Answer] = set()
        self._lock = threading.Lock()

    @property
    def fns(self) -> Mapping[str, Fn]:
        return self._fns

    @property
    def in_flight(self) -> bool:
        """Whether any call is still to answer."""
        with self._lock:
            return any(not answer.done() for answer in self._unread)

    @property
    def owes_an_answer(self) -> bool:
        """Whether any call's answer has still to be read, whether or not that call has landed."""
        with self._lock:
            return bool(self._unread)

    def wait(self, timeout: float | None = None) -> None:
        """Block until every call made so far has answered, or until ``timeout`` seconds pass."""
        with self._lock:
            pending = [answer.call for answer in self._unread]
        concurrent.futures.wait(pending, timeout=timeout)

    def _start(self, name: str, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Answer:
        context = contextvars.copy_context()
        call = self._pool.submit(context.run, fn, *args, **kwargs)
        answer = self._Answer(name, call, self._read)
        with self._lock:
            self._unread.add(answer)
        return answer

    def _read(self, answer: '_Answer') -> None:
        with self._lock:
            self._unread.discard(answer)

    @staticmethod
    def _closed(*args: Any, **kwargs: Any) -> Answer:
        raise RuntimeError('The runtime is closed and serves nothing')

    def close(self) -> None:
        """Drop the queued calls and wait out those in flight, which may still hold their caller's resources.
        A call made after close raises.

        Reports what a call raised that no caller read: the session that asked for it has gone.
        """
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            # A function holds what it was declared with — model weights, a socket. Nothing reaches them
            # through this runtime after it closes.
            unread, self._unread = self._unread, set()
            self._fns = dict.fromkeys(self._fns, self._closed)
        for answer in unread:
            # rules-allow: swallowed-error — the caller dropped the answer, so there is nobody to raise to,
            # and the log is the only place the failure can go.
            if (exc := answer.failure()) is not None:
                logging.error(f'The function {answer.name} failed and no caller read its answer: {exc}')
