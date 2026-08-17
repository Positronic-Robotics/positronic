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
- `cancel()` on the future raises.
- An exception set by the handler is what `result()` raises at the caller; its traceback does not survive a
  process boundary.
- All of the above holds when a caller and its futures are used from one thread, and a handler and its calls
  from one thread.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import Future
from typing import Any, Generic, ParamSpec, TypeVar

P = ParamSpec('P')
R = TypeVar('R')


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
        """Calls that arrived since the previous drain; each may be answered now or on a later tick."""
