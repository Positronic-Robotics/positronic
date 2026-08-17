"""Calls: request/reply between control systems, the counterpart of signals.

A control system declares a `Caller` where it invokes another control system and a `Handler` where it serves
invocations. `World.connect` binds a caller to a handler wherever the two run.

- A handler serves at most one caller; unbound, its `incoming()` yields nothing.
- Calling an unbound caller raises.
- The contract is the same in-process and across processes; across processes requests, results and exceptions
  must be picklable.
- No call and no reply is dropped.
- Calls from one caller reach the handler in the order made; replies may return in any order.
- `incoming()` yields calls one at a time as it is advanced; each call is yielded once, and one it has not
  reached is yielded by a later `incoming()`.
- Each `Call` is answered once, with `set_result` or `set_exception`; answering again raises in the handler.
- `__call__` returns a `concurrent.futures.Future` completed by the handler's answer and by nothing else.
- The future never waits: `done()`, `result()` and `exception()` return at once. On an unanswered future
  `result()` and `exception()` raise `TimeoutError`; a positive `timeout` raises `NotImplementedError`;
  `concurrent.futures.wait()` and `as_completed()` never return for it.
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
from typing import Generic, TypeVar

from .core import ControlSystem, ControlSystemEmitter, ControlSystemReceiver, SignalEmitter, SignalReceiver

Req = TypeVar('Req')
Res = TypeVar('Res')
T = TypeVar('T')


class Call(ABC, Generic[Req, Res]):
    """One invocation awaiting its answer: the request, and the reply slot the caller's `Future` is bound to."""

    @property
    @abstractmethod
    def request(self) -> Req: ...

    @abstractmethod
    def set_result(self, value: Res) -> None: ...

    @abstractmethod
    def set_exception(self, exc: BaseException) -> None: ...


class Caller(ABC, Generic[Req, Res]):
    @abstractmethod
    def __call__(self, request: Req) -> Future[Res]: ...


class Handler(ABC, Generic[Req, Res]):
    @abstractmethod
    def incoming(self) -> Iterator[Call[Req, Res]]:
        """Calls not yet yielded; each may be answered now or on a later tick."""


@dataclass(frozen=True)
class _Request(Generic[Req]):
    id: int
    data: Req


@dataclass(frozen=True)
class _Result(Generic[Res]):
    id: int
    value: Res


@dataclass(frozen=True)
class _Failure:
    id: int
    exc: BaseException


def _drain(receiver: SignalReceiver[T]) -> Iterator[T]:
    """Every message queued on `receiver`, oldest first."""
    while (msg := receiver.read()) is not None and msg.updated:
        yield msg.data


class _ControlSystemCall(Call[Req, Res]):
    def __init__(self, request: _Request[Req], replies: SignalEmitter[_Result[Res] | _Failure]):
        self._request = request
        self._replies = replies
        self._answered = False

    @property
    def request(self) -> Req:
        return self._request.data

    def set_result(self, value: Res) -> None:
        self._answer(_Result(self._request.id, value))

    def set_exception(self, exc: BaseException) -> None:
        self._answer(_Failure(self._request.id, exc))

    def _answer(self, reply: _Result[Res] | _Failure) -> None:
        if self._answered:
            raise InvalidStateError(f'Call {self._request.id} is already answered')
        self._replies.emit(reply)
        self._answered = True


class ControlSystemHandler(Handler[Req, Res]):
    """A control system's handler; `requests` and `replies` are the signal endpoints `World.connect` binds."""

    def __init__(self, owner: ControlSystem):
        self.requests = ControlSystemReceiver[_Request[Req]](owner, maxsize=0)
        self.replies = ControlSystemEmitter[_Result[Res] | _Failure](owner)

    def incoming(self) -> Iterator[Call[Req, Res]]:
        for request in _drain(self.requests):
            yield _ControlSystemCall(request, self.replies)


class _ReplyFuture(Future[Res]):
    """The caller's view of one call: a `Future` that pulls its answer from the caller's replies on inspection."""

    def __init__(self, deliver_replies: Callable[[], None]):
        super().__init__()
        self._deliver_replies = deliver_replies

    def done(self) -> bool:
        self._deliver_replies()
        return super().done()

    def result(self, timeout: float | None = None) -> Res:
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


class ControlSystemCaller(Caller[Req, Res]):
    """A control system's caller; `requests` and `replies` are the signal endpoints `World.connect` binds."""

    def __init__(self, owner: ControlSystem):
        self.requests = ControlSystemEmitter[_Request[Req]](owner)
        self.replies = ControlSystemReceiver[_Result[Res] | _Failure](owner, maxsize=0)
        self._pending: dict[int, _ReplyFuture[Res]] = {}
        self._next_id = 0

    def __call__(self, request: Req) -> Future[Res]:
        if self.requests.num_bound == 0:
            raise RuntimeError('Caller is not connected to a handler')
        envelope = _Request(self._next_id, request)
        self._next_id += 1
        future = _ReplyFuture(self._deliver_replies)
        self._pending[envelope.id] = future
        self.requests.emit(envelope)
        return future

    def _deliver_replies(self) -> None:
        for reply in _drain(self.replies):
            future = self._pending.pop(reply.id)
            if isinstance(reply, _Result):
                future.set_result(reply.value)
            else:
                future.set_exception(reply.exc)
