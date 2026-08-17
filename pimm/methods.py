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
