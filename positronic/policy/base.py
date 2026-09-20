"""Reusable processors, their episode generators, and the runtime that serves them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping
from typing import Any, ClassVar, Generic, ParamSpec, TypeVar

from attr import dataclass
from typing_extensions import TypeAliasType

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
    """Commands to emit now and the next requested call time.

    Any submitted call becoming ready may resume the policy sooner, including at the same clock time.
    """

    # TODO: Allow selecting which answers can wake the policy early, alongside its time deadline.
    commands: Commands
    resume_at_ns: int


P = ParamSpec('P')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

ProcessorRun = TypeAliasType('ProcessorRun', Generator[OutputT | None, InputT, None], type_params=(InputT, OutputT))


class Runtime(ABC):
    """What the framework offers one episode. Every episode gets its own.

    At episode shutdown, drain submitted work before closing live generators whose resources it may use.

    TODO: Define how generators report episode metadata.
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

    def start(
        self, processor: Processor[InputT, OutputT], /, *args: Any, **kwargs: Any
    ) -> ProcessorRun[InputT, OutputT]:
        """Create and prime an episode generator. The caller owns its closure."""
        run = processor.run(self, *args, **kwargs)
        initial = next(run)
        if initial is not None:
            run.close()
            raise AssertionError('a processor must yield None before receiving its first input')
        return run

    @abstractmethod
    def submit(self, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Answer[T]: ...


class Processor(ABC, Generic[InputT, OutputT]):
    """A reusable computation definition with typed inputs and outputs.

    Constructors hold configuration. Each ``run`` creates one episode's generator, keeping mutable
    episode state in its locals. ``Runtime.start`` advances it to its first yield, which must be
    ``None``, after which ``send(input)`` returns one output. Dependencies are supplied live and already
    primed when they are generators. Whoever starts a generator owns its closure.
    """

    # A receiver resolves this name through its registry of installed processor classes.
    WIRE_NAME: ClassVar[str]

    @abstractmethod
    def run(self, runtime: Runtime, *args: Any, **kwargs: Any) -> ProcessorRun[InputT, OutputT]: ...

    def meta(self) -> dict[str, Any]:
        """Model and configuration metadata shared across episodes."""
        return {}

    def to_spec(self) -> dict[str, Any]:
        """A registered name and plain-data constructor arguments, for deliverable processors."""
        raise NotImplementedError(f'{type(self).__name__} has no wire spec')


Policy = Processor[Obs, Step]
PolicyRun = ProcessorRun[Obs, Step]
