"""The in-process runtime, and the two ways code calls a served function.

A session holds one call at a time through a ``Caller``. A caller with no control loop of its own —
a server request, a warmup, a probe — runs a session call to its answer with ``call_until_answered``.
"""

import concurrent.futures
import contextvars
import logging
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any

from positronic.policy.base import Answer, Fn, NotAnswered, Runtime, Session


class Executor(Runtime):
    """Serves a set of functions on worker threads of its own, ``max_workers`` calls at a time.

    A call runs under a copy of the context it was made in, so telemetry recorded inside it anchors where
    it was asked for.
    """

    class _Answer(Answer):
        def __init__(self, name: str, call: Future[Any], read: Callable[['Executor._Answer'], None]):
            self.name = name
            self._call = call
            self._read = read

        def done(self) -> bool:
            return self._call.done()

        def result(self) -> Any:
            if not self._call.done():
                raise NotAnswered('The call is not answered yet')
            self._read(self)
            return self._call.result()

        def failure(self) -> BaseException | None:
            """What the call raised, once it has answered. ``None`` when it returned a value or was cancelled."""
            return None if self._call.cancelled() else self._call.exception()

    def __init__(self, functions: Mapping[str, Callable[..., Any]], *, max_workers: int = 1):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='policy-fn')
        self._fns: Mapping[str, Fn] = {name: partial(self._start, name, fn) for name, fn in functions.items()}
        # Every call made and not answered yet, read from the caller's thread while the workers answer. The
        # lock is reentrant because a call that answers before it is registered runs ``_answered`` inline.
        self._pending: set[Future[Any]] = set()
        # Every answer that no caller has read. What is left at close failed with nobody to raise to.
        self._unread: set[Executor._Answer] = set()
        self._lock = threading.RLock()

    @property
    def fns(self) -> Mapping[str, Fn]:
        return self._fns

    @property
    def in_flight(self) -> bool:
        """Whether any call is still to answer."""
        with self._lock:
            return bool(self._pending)

    @property
    def owes_an_answer(self) -> bool:
        """Whether any call's answer has still to be read, whether or not that call has landed."""
        with self._lock:
            return bool(self._unread)

    def wait(self, timeout: float | None = None) -> None:
        """Block until every call made so far has answered, or until ``timeout`` seconds pass."""
        with self._lock:
            pending = set(self._pending)
        concurrent.futures.wait(pending, timeout=timeout)

    def _start(self, name: str, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Answer:
        context = contextvars.copy_context()
        call = self._pool.submit(context.run, fn, *args, **kwargs)
        answer = self._Answer(name, call, self._read)
        with self._lock:
            self._pending.add(call)
            self._unread.add(answer)
            call.add_done_callback(self._answered)
        return answer

    def _answered(self, call: Future[Any]) -> None:
        with self._lock:
            self._pending.discard(call)

    def _read(self, answer: '_Answer') -> None:
        with self._lock:
            self._unread.discard(answer)

    def close(self) -> None:
        """Drop the queued calls and wait out those in flight, which may still hold their caller's resources.
        A call made after close raises.

        Reports what a call raised that no caller read: the session that asked for it has gone.
        """
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            unread, self._unread = self._unread, set()
        for answer in unread:
            # rules-allow: swallowed-error — the caller dropped the answer, so there is nobody to raise to,
            # and the log is the only place the failure can go.
            if (exc := answer.failure()) is not None:
                logging.error(f'The function {answer.name} failed and no caller read its answer: {exc}')


class Caller:
    """One session's use of one served function, with one call in flight at a time.

    A session starts a call and gives control back. It reads the answer on a later call, through ``take``.
    """

    def __init__(self, rt: Runtime, name: str):
        self._fn = rt.fns[name]
        self._answer: Answer | None = None
        self._cancelled = False

    @property
    def idle(self) -> bool:
        """Whether no call is held, so a new one can start."""
        return self._answer is None

    @property
    def in_flight(self) -> bool:
        """Whether the call that is held has still to answer."""
        return self._answer is not None and not self._answer.done()

    def start(self, *args: Any, **kwargs: Any) -> None:
        assert self._answer is None, 'a call is already in flight'
        self._answer = self._fn(*args, **kwargs)

    def take(self) -> Any:
        """What the call returned, and ``None`` while it is out or after a cancel."""
        assert self._answer is not None, 'no call is held, so there is no answer to take'
        if not self._answer.done():
            return None
        answer, cancelled = self._answer, self._cancelled
        # Both are cleared before the read, because ``result`` raises what the call raised. A cancel then
        # ends with the answer it was made against, and does not drop the next one.
        self._answer, self._cancelled = None, False
        result = answer.result()
        return None if cancelled else result

    def cancel(self) -> None:
        """Drop the result of the call that is held. ``take`` still raises what that call raised."""
        self._cancelled = self._answer is not None


def call_until_answered(session: Session, rt: Executor, obs: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """What ``session`` answers for ``obs``, over as many calls as the functions it starts take.

    For a caller that has no control loop to give the time back to.
    """
    actions = session(obs)
    # Not ``in_flight``: a call that lands before this test would leave its answer unread, and the session
    # reads an answer only on a later call.
    while actions is None and rt.owes_an_answer:
        rt.wait()
        actions = session(obs)
    return actions
