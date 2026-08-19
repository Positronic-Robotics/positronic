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
- `__call__` returns an `Answer` completed by the handler's answer and by nothing else.
- An `Answer` never waits: `done()` and `result()` return at once; `result()` on an unanswered call raises
  `NoValueException`.
- An exception set by the handler is what `result()` raises at the caller; its traceback does not survive a
  process boundary.
- All of the above holds when a caller and its answers are used from one thread, and a handler and its calls
  from one thread.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

from .core import (
    ControlSystem,
    ControlSystemEmitter,
    ControlSystemReceiver,
    NoValueException,
    SignalEmitter,
    SignalReceiver,
)

Req = TypeVar('Req')
Res = TypeVar('Res')
T = TypeVar('T')


class Call(ABC, Generic[Req, Res]):
    """One invocation awaiting its answer: the request, and the reply slot the caller's `Answer` is bound to."""

    @property
    @abstractmethod
    def request(self) -> Req: ...

    @abstractmethod
    def set_result(self, value: Res) -> None: ...

    @abstractmethod
    def set_exception(self, exc: BaseException) -> None: ...


@contextmanager
def answering(call: Call[Req, Res]) -> Iterator[None]:
    """Answer `call` with whatever the block raises; its result, if any, the block sets itself."""
    try:
        yield
    # rules-allow: swallowed-error — the exception is not dropped but handed to the caller, who raises it
    except Exception as exc:
        call.set_exception(exc)


class Answer(ABC, Generic[Res]):
    """The caller's handle on one call: `done()` once the handler has answered, then `result()`."""

    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def result(self) -> Res: ...


class Caller(ABC, Generic[Req, Res]):
    @abstractmethod
    def __call__(self, request: Req) -> Answer[Res]: ...


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
        assert not self._answered, f'Call {self._request.id} is already answered'
        self._replies.emit(reply)
        self._answered = True


class ControlSystemHandler(Handler[Req, Res]):
    """A control system's handler; `requests` and `replies` are the signal endpoints `World.connect` binds."""

    def __init__(self, owner: ControlSystem):
        self.requests = ControlSystemReceiver[_Request[Req]](owner, maxsize=0)
        self.replies = ControlSystemEmitter[_Result[Res] | _Failure](owner)

    @property
    def owner(self) -> ControlSystem:
        return self.requests.owner

    def incoming(self) -> Iterator[Call[Req, Res]]:
        for request in _drain(self.requests):
            yield _ControlSystemCall(request, self.replies)


class _ControlSystemAnswer(Answer[Res]):
    """Pulls its reply from the caller's replies on inspection."""

    def __init__(self, deliver_replies: Callable[[], None]):
        self._deliver_replies = deliver_replies
        self._reply: _Result[Res] | _Failure | None = None

    def done(self) -> bool:
        return self._pull() is not None

    def result(self) -> Res:
        match self._pull():
            case None:
                raise NoValueException('The call is not answered yet')
            case _Failure(exc=exc):
                raise exc
            case _Result(value=value):
                return value

    def _pull(self) -> _Result[Res] | _Failure | None:
        if self._reply is None:
            self._deliver_replies()
        return self._reply


class ControlSystemCaller(Caller[Req, Res]):
    """A control system's caller; `requests` and `replies` are the signal endpoints `World.connect` binds."""

    def __init__(self, owner: ControlSystem):
        self.requests = ControlSystemEmitter[_Request[Req]](owner)
        self.replies = ControlSystemReceiver[_Result[Res] | _Failure](owner, maxsize=0)
        self._pending: dict[int, _ControlSystemAnswer[Res]] = {}
        self._next_id = 0

    @property
    def owner(self) -> ControlSystem:
        return self.requests.owner

    def __call__(self, request: Req) -> Answer[Res]:
        if self.requests.num_bound == 0:
            raise RuntimeError('Caller is not connected to a handler')
        envelope = _Request(self._next_id, request)
        self._next_id += 1
        answer = _ControlSystemAnswer(self._deliver_replies)
        self._pending[envelope.id] = answer
        self.requests.emit(envelope)
        return answer

    def _deliver_replies(self) -> None:
        for reply in _drain(self.replies):
            self._pending.pop(reply.id)._reply = reply
