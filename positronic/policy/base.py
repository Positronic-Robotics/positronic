"""Typed processors, their construction recipes, and the runtime that serves them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, ClassVar, Generic, ParamSpec, TypeVar

from attr import dataclass

# Structural keys of the wire spec for sequential and parallel composition.
SEQ = 'seq'
PAR = 'par'


class NotAnswered(RuntimeError):
    """The call has not answered yet. A read of an ``Answer`` raises this rather than waiting."""


T = TypeVar('T')


class Answer(ABC, Generic[T]):
    """The caller's handle on one call.

    pimm has an ``Answer`` of the same shape. The two are not interchangeable.
    """

    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def result(self) -> T:
        """What the function returned. Raises what the function raised, or ``NotAnswered`` before it answers."""

    def cancel(self):
        """Cancel the call if anything can be cancelled."""
        pass


Obs = Mapping[str, Any]
Commands = Mapping[str, Any]


@dataclass
class Step:
    commands: Commands
    resume_at_ns: int


P = ParamSpec('P')


class Runtime(ABC):
    """What the framework offers one session. Every session gets its own.

    Closed before the session it serves: a call still in flight is using what that session holds.

    TODO: Expose per-processor and submitted-call timings through the runtime.
    """

    @property
    @abstractmethod
    def time_ns(self) -> int:
        """The current time in nanoseconds."""
        pass

    @property
    @abstractmethod
    def tick(self) -> int:
        """The current tick number."""
        pass

    @abstractmethod
    def submit(self, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Answer[T]: ...


InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class Processor(ABC, Generic[InputT, OutputT]):
    """An episode-scoped computation with typed inputs and outputs."""

    # A receiver resolves this name through its registry of installed processor classes.
    WIRE_NAME: ClassVar[str | None] = None

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime

    @abstractmethod
    def __call__(self, value: InputT) -> OutputT: ...

    def close(self) -> None:
        return None

    def meta(self) -> dict[str, Any]:
        """What this processor reports about its computation and its episode."""
        return {}


Policy = Processor[Obs, Step]


ProcessorT = TypeVar('ProcessorT', bound=Processor[Any, Any])


class Factory(Generic[ProcessorT]):
    """A processor constructor and its configuration, reusable across episodes.

    Configured arguments are copied for each build. Supply live dependencies, such as child
    processors or inference callables, to ``build`` after the runtime; they are passed by reference.
    ``to_spec`` describes only the constructor and configuration, without creating episode state.

    TODO: Have the wire-spec loader resolve registered names into factories before building a stack.
    """

    def __init__(self, processor: type[ProcessorT], /, **kwargs: Any) -> None:
        self._processor = processor
        self._kwargs = deepcopy(kwargs)

    def build(self, runtime: Runtime, /, *dependencies: Any) -> ProcessorT:
        """Create a fresh processor with positional dependencies followed by configured keywords."""
        return self._processor(runtime, *dependencies, **deepcopy(self._kwargs))

    @staticmethod
    def _spec_value(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        if isinstance(value, (tuple, list)):
            return [Factory._spec_value(item) for item in value]
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError('Wire-spec argument dictionaries must have string keys')
            return {key: Factory._spec_value(item) for key, item in value.items()}
        raise TypeError(f'{type(value).__name__} is not a wire-spec argument')

    def to_spec(self) -> dict[str, Any]:
        """Describe construction using a registered name and plain-data keyword arguments."""
        name = self._processor.WIRE_NAME
        if name is None:
            raise TypeError(f'{self._processor.__name__} has no wire name')
        spec: dict[str, Any] = {'name': name}
        if self._kwargs:
            spec['args'] = self._spec_value(self._kwargs)
        return spec


class Sequential(Generic[ProcessorT]):
    """Nest processor factories, with the first outermost and a child supplied at build time.

    ``Sequential(Factory(A), Factory(B)).build(runtime, child)`` constructs
    ``A(runtime, B(runtime, child))``. Each processor controls calls to its child.
    """

    def __init__(
        self, first: Factory[ProcessorT] | Sequential[ProcessorT], /, *rest: Factory[Any] | Sequential[Any]
    ) -> None:
        self._first = first
        self._rest = rest

    def build(self, runtime: Runtime, inner: Callable[..., Any]) -> ProcessorT:
        """Build inside out, passing each newly created processor to its parent."""
        for factory in reversed(self._rest):
            inner = factory.build(runtime, inner)
        return self._first.build(runtime, inner)

    def to_spec(self) -> dict[str, Any]:
        return {SEQ: [factory.to_spec() for factory in (self._first, *self._rest)]}
