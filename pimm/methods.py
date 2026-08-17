"""Methods: request/reply between control systems, the counterpart of signals.

A control system declares a `MethodCaller` where it invokes another control system and a `MethodHandler` where
it serves invocations. `World.connect` binds a caller to a handler wherever the two run.

- A handler serves at most one caller; unbound, its `incoming()` yields nothing.
- Calling an unbound caller raises.
- The contract is the same in-process and across processes; across processes arguments, results and exceptions
  must be picklable.
- No call and no reply is dropped.
- Calls from one caller reach the handler in the order made; replies may return in any order.
- `incoming()` yields each call once.
- Each `Call` is answered once, with `set_result` or `set_exception`; answering again raises in the handler.
- `__call__` returns a `concurrent.futures.Future` completed by the handler's answer and by nothing else.
- The future never waits: `done()`, `result()` and `exception()` return at once. On an unanswered future
  `result()` and `exception()` raise `TimeoutError`; a positive `timeout` raises `NotImplementedError`.
- `cancel()` on the future raises.
- An exception set by the handler is what `result()` raises at the caller; its traceback does not survive a
  process boundary.
- All of the above holds when a caller and its futures are used from one thread, and a handler and its calls
  from one thread.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from typing import Any, Generic, ParamSpec, TypeVar

from .core import ControlSystem, ControlSystemEmitter, ControlSystemReceiver, SignalEmitter, SignalReceiver

P = ParamSpec('P')
R = TypeVar('R')
T = TypeVar('T')


class Call(ABC, Generic[P, R]):
    """One invocation awaiting its answer: the arguments, and the reply slot the caller's `Future` is bound to."""

    @property
    @abstractmethod
    def args(self) -> tuple[Any, ...]: ...

    @property
    @abstractmethod
    def kwargs(self) -> dict[str, Any]: ...

    @abstractmethod
    def set_result(self, value: R) -> None: ...

    @abstractmethod
    def set_exception(self, exc: BaseException) -> None: ...


class MethodCaller(ABC, Generic[P, R]):
    @abstractmethod
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Future[R]: ...


class MethodHandler(ABC, Generic[P, R]):
    @abstractmethod
    def incoming(self) -> Iterator[Call[P, R]]:
        """Calls not yet yielded; each may be answered now or on a later tick."""


@dataclass(frozen=True)
class _Request:
    id: int
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class _Result:
    id: int
    value: Any


@dataclass(frozen=True)
class _Failure:
    id: int
    exc: BaseException


def _drain(receiver: SignalReceiver[T]) -> Iterator[T]:
    """Every message queued on `receiver`, oldest first."""
    while (msg := receiver.read()) is not None and msg.updated:
        yield msg.data


class _ControlSystemCall(Call[P, R]):
    def __init__(self, request: _Request, replies: SignalEmitter[_Result | _Failure]):
        self._request = request
        self._replies = replies
        self._answered = False

    @property
    def args(self) -> tuple[Any, ...]:
        return self._request.args

    @property
    def kwargs(self) -> dict[str, Any]:
        return self._request.kwargs

    def set_result(self, value: R) -> None:
        self._answer(_Result(self._request.id, value))

    def set_exception(self, exc: BaseException) -> None:
        self._answer(_Failure(self._request.id, exc))

    def _answer(self, reply: _Result | _Failure) -> None:
        if self._answered:
            raise InvalidStateError(f'Call {self._request.id} is already answered')
        self._replies.emit(reply)
        self._answered = True


class ControlSystemHandler(MethodHandler[P, R]):
    """A control system's handler; `requests` and `replies` are the signal endpoints `World.connect` binds."""

    def __init__(self, owner: ControlSystem):
        self.requests = ControlSystemReceiver[_Request](owner, maxsize=0)
        self.replies = ControlSystemEmitter[_Result | _Failure](owner)

    def incoming(self) -> Iterator[Call[P, R]]:
        return iter([_ControlSystemCall(request, self.replies) for request in _drain(self.requests)])


class _ReplyFuture(Future[R]):
    """The caller's view of one call: a `Future` that pulls its answer from the caller's replies on inspection."""

    def __init__(self, deliver_replies: Callable[[], None]):
        super().__init__()
        self._deliver_replies = deliver_replies

    def done(self) -> bool:
        self._deliver_replies()
        return super().done()

    def result(self, timeout: float | None = None) -> R:
        self._reject_waiting(timeout)
        self._deliver_replies()
        return super().result(timeout=0)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        self._reject_waiting(timeout)
        self._deliver_replies()
        return super().exception(timeout=0)

    def cancel(self) -> bool:
        raise NotImplementedError('A method call cannot be cancelled')

    @staticmethod
    def _reject_waiting(timeout: float | None) -> None:
        if timeout:
            raise NotImplementedError('A method future cannot wait; poll `done()` between yields')


class ControlSystemCaller(MethodCaller[P, R]):
    """A control system's caller; `requests` and `replies` are the signal endpoints `World.connect` binds."""

    def __init__(self, owner: ControlSystem):
        self.requests = ControlSystemEmitter[_Request](owner)
        self.replies = ControlSystemReceiver[_Result | _Failure](owner, maxsize=0)
        self._pending: dict[int, _ReplyFuture[R]] = {}
        self._next_id = 0

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        if self.requests.num_bound == 0:
            raise RuntimeError('Caller is not connected to a handler')
        request = _Request(self._next_id, args, kwargs)
        self._next_id += 1
        future = _ReplyFuture(self._deliver_replies)
        self._pending[request.id] = future
        self.requests.emit(request)
        return future

    def _deliver_replies(self) -> None:
        for reply in _drain(self.replies):
            future = self._pending.pop(reply.id)
            if isinstance(reply, _Result):
                future.set_result(reply.value)
            else:
                future.set_exception(reply.exc)
